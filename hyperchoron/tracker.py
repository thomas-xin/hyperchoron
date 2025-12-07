from dataclasses import dataclass
import functools
from math import ceil, hypot, log10
import os
from types import SimpleNamespace
from .mappings import (
	org_instrument_selection, org_instrument_mapping,
	instrument_names, midi_instrument_selection,
	c4, c1, percussion_mats,
)
from .util import create_reader, log2lin, priority_ordering, temp_dir, in_sample_rate, event_types


@dataclass(slots=True)
class OrgNote:
	tick: int
	pitch: int
	length: int
	volume: int
	panning: int

	@property
	def aligned(self):
		return self.tick + self.length

def build_org(notes, instrument_activities, speed_info, ctx):
	orig_ms_per_clock, real_ms_per_clock, scale, orig_step_ms, _orig_tempo = speed_info
	speed_ratio = real_ms_per_clock / scale / orig_ms_per_clock
	wait = max(1, round(orig_step_ms / speed_ratio))
	activities = list(map(list, instrument_activities.items()))
	instruments = []
	if not ctx.extended_ranges and sum(t[1][1] for t in activities) >= 12:
		instruments.append(SimpleNamespace(
			id=60,
			index=0,
			type=12,
			notes=[],
		))
	while len(instruments) < 8:
		activities.sort(key=lambda t: t[1][0], reverse=True)
		curr = activities[0]
		if ctx.extended_ranges:
			curr[1][0] /= 4096
		else:
			curr[1][0] /= 3
		typeid = curr[0]
		itype = org_instrument_selection[typeid]
		if itype < 0:
			itype = 7
		instruments.append(SimpleNamespace(
			id=itype,
			index=len(instruments),
			type=typeid,
			notes=[],
		))
	for i, drum in enumerate([0, 2, 5, 6, 4, 8, 0, 0]):
		instruments.append(SimpleNamespace(
			id=drum,
			index=len(instruments),
			type=-1,
			notes=[],
		))
	note_count = sum(map(len, notes))
	conservative = False
	if note_count > (65536 * 8 * 4 if ctx.extended_ranges else 4096 * 8 * 4):
		print(f"High note count {note_count} detected, using conservative note sustains...")
		conservative = True
	max_vol = 1
	active = {}
	for i, beat in enumerate(notes):
		taken = {}
		next_active = {}
		if beat and len(beat) > 1 and not ctx.extended_ranges:
			for note in beat:
				if note.priority < 0:
					note.priority = 0
			ordered = priority_ordering(beat, n_largest=12, active=active)
			lowest = min((note.instrument_class == -1, note) for note in beat)[-1]
			if sum(note.instrument_class != -1 for note in ordered) > 8 and org_instrument_selection[lowest.instrument_class] >= 0:
				lowest_to_remove = True
				for j, note in enumerate(ordered):
					if note.instrument_class == -1:
						continue
					if note == lowest and lowest_to_remove:
						ordered.pop(j)
						lowest_to_remove = False
					elif note.pitch == lowest.pitch + 12:
						vel = hypot(note.velocity, lowest.velocity) * 3 / 2
						tvel = min(max_vol, vel)
						keep = lowest.velocity - tvel * 2 / 3
						note.pitch = lowest.pitch
						note.velocity = keep
						lowest.priority = max(note.priority, lowest.priority)
						lowest.velocity = tvel
						lowest.instrument_class = 12
						lowest.panning = (note.panning + lowest.panning) / 2
						break
				if lowest.instrument_class != 12 and lowest.pitch < c4 - 12:
					lowest.instrument_class = 12
					lowest.velocity = min(max_vol, lowest.velocity * 3 / 2)
				ordered.insert(0, lowest)
			elif len(ordered) > 1:
				try:
					ordered.remove(lowest)
				except ValueError:
					pass
				else:
					ordered.insert(0, lowest)
		else:
			ordered = list(beat)
		loudest = hypot(*(log2lin(note.velocity) for note in beat))
		for note in ordered:
			itype, pitch, priority, pan = note.instrument_class, note.pitch, note.priority, note.panning
			pitch = round(pitch)
			vel = log2lin(note.velocity)
			if vel < loudest * 0.01:
				continue
			vel = (round(vel * 4) * 64 if conservative else round(vel * 32) * 8) / 256
			if vel > 0:
				vel = 255 * (1 + log10(vel))
			volume = max(0, min(254, round(vel)))
			panning = round(pan * 6 + 6)
			ins = org_instrument_selection[itype]
			if ins < 0:
				try:
					mat, rpitch = percussion_mats[pitch]
				except KeyError:
					continue
				if mat == "PLACEHOLDER":
					continue
				pitch = 0
				ins = instrument_names[mat]
				match ins:
					case "basedrum":
						if mat in ("netherrack", "cobblestone"):
							iid = 8
							pitch = rpitch - 12
						else:
							iid = 12
							pitch = rpitch - 18
					case "snare" if rpitch >= 18:
						iid = 11
						pitch = rpitch - 12
					case "snare":
						iid = 9
					case "hat":
						iid = 10
					case "creeper":
						iid = 11
					case "xylophone" if itype == -1:
						iid = 13
						pitch = rpitch
					case _:
						iid = 12
						pitch = rpitch
				if iid in taken and not ctx.extended_ranges:
					continue
				new_pitch = pitch + c4 - c1
				note = OrgNote(
					tick=i,
					pitch=new_pitch,
					length=1,
					volume=volume,
					panning=panning,
				)
				instruments[iid].notes.append(note)
				taken[iid] = taken.get(iid, 0) + 1
				continue
			new_pitch = pitch - c1
			if new_pitch < 0:
				new_pitch += 12
			new_pitch = max(0, min(95, new_pitch))
			h = (itype, new_pitch)
			if (priority < 2 or conservative) and h in active:
				try:
					for iid in active[h]:
						if iid in taken and not ctx.extended_ranges:
							continue
						instrument = instruments[iid]
						for x in range(1, len(instrument.notes)):
							target = instrument.notes[-x]
							if target.pitch == 255:
								continue
							if target.aligned < i - 1:
								break
							if target.aligned != i:
								continue
							if target.length >= 192:
								continue
							if target.volume != volume or (target.panning != panning and not conservative):
								instrument.notes.append(OrgNote(
									tick=i,
									pitch=255,
									length=1,
									volume=volume,
									panning=target.panning if conservative else panning,
								))
							target.length += 1
							taken[iid] = taken.get(iid, 0) + 1
							next_active.setdefault(h, []).append(instrument.index)
							raise StopIteration
				except StopIteration:
					continue
			if ctx.extended_ranges:
				instrument = max(instruments[:8], key=lambda instrument: (len(instrument.notes) < 65504, instrument.id == ins, -taken.get(instrument.index, 0), -(len(instrument.notes) // 4096)))
			else:
				instrument = max(instruments[:8], key=lambda instrument: (len(instrument.notes) < 65504, -taken.get(instrument.index, 0), len(instrument.notes) < 3072, instrument.id == ins, instrument.id != 60, -(len(instrument.notes) // 1024)))
			iid = instrument.index
			if iid in taken and not ctx.extended_ranges:
				continue
			if instrument.id == 60 and ins != 60 and pitch >= 12:
				pitch -= 12
			instrument.notes.append(OrgNote(
				tick=i,
				pitch=new_pitch,
				length=1,
				volume=volume,
				panning=panning,
			))
			taken[iid] = taken.get(iid, 0) + 1
			try:
				next_active[h].append(instrument.index)
			except KeyError:
				next_active[h] = [instrument.index]
		active = next_active
	for instrument in instruments:
		if len(instrument.notes) > 4096:
			instrument.notes = [note for i, note in enumerate(instrument.notes) if note.pitch != 255 or note.volume in (0, 64, 128, 192, 254)]
		if len(instrument.notes) >= 65536:
			instrument.notes = instrument.notes[:65535]
	return instruments, wait


def read_org(fi):
	if isinstance(fi, str):
		fi = open(fi, "rb")
	with fi:
		read = create_reader(fi)
		wait = read(6, 2)
		end = read(14, 4)
		instruments = []
		for i in range(16):
			finetune = read(18 + i * 6, 2)
			iid = read(18 + i * 6 + 2, 1)
			sus = i < 8 and not read(18 + i * 6 + 3, 1)
			count = read(18 + i * 6 + 4, 2)
			instruments.append(SimpleNamespace(
				id=iid,
				events=[SimpleNamespace(
					tick=0,
					instrument=i,
					pitch=0,
					length=0,
					volume=0,
					panning=0,
				) for x in range(count)],
				finetune=finetune,
				sustain=sus,
				drum=i >= 8,
			))
		offsets = 114
		for i, ins in enumerate(instruments):
			for j, e in enumerate(ins.events):
				e.tick = read(offsets + j * 4, 4)
			offsets += len(ins.events) * 4
			for j, e in enumerate(ins.events):
				e.pitch = read(offsets + j, 1)
			offsets += len(ins.events)
			for j, e in enumerate(ins.events):
				e.length = read(offsets + j, 1)
			offsets += len(ins.events)
			for j, e in enumerate(ins.events):
				e.volume = read(offsets + j, 1)
			offsets += len(ins.events)
			for j, e in enumerate(ins.events):
				e.panning = read(offsets + j, 1)
			offsets += len(ins.events)
	for ins in instruments:
		ins.events.sort(key=lambda e: (e.tick, e.volume == 255))
	return SimpleNamespace(
		wait=wait,
		end=end,
		instruments=instruments,
	)

def load_org(file):
	print("Importing ORG...")
	min_vol = 16
	orgfile = read_org(file)
	events = [
		[0, 0, event_types.HEADER, 1, len(orgfile.instruments) + 1, 1, 0, 16],
		[1, 0, event_types.TEMPO, orgfile.wait * 1000],
	]
	for i, ins in enumerate(orgfile.instruments):
		if i not in range(8, 16):
			events.append([i + 2, 0, event_types.PROGRAM_C, i, midi_instrument_selection[org_instrument_mapping[ins.id]], ins.id])
			channel = i
			pitch = None
		else:
			channel = 9
			if i == 8:
				events.append([i + 2, 0, event_types.PROGRAM_C, channel, 0])
			pitch = [35, 38, 42, 46, 47, 73, 35, 35][i - 8]
		for j, e in enumerate(ins.events):
			if e.panning != 255:
				events.append([i + 2, e.tick, event_types.CONTROL_C, channel, 10, (e.panning - 6) / 6 * 63 + 64])
			if e.volume != 255:
				events.append([i + 2, e.tick, event_types.CONTROL_C, channel, 7, e.volume / 254 * (127 - min_vol) + min_vol])
			if e.pitch != 255:
				events.append([i + 2, e.tick, event_types.NOTE_ON_C, channel, pitch or e.pitch + c1, 127, 1, e.length])
	return events


# TODO: Finish implementation with filters and instrument detection
def load_xm(file):
	print("Importing XM...")

	import librosa
	import numpy as np
	import soundfile as sf
	from .pcm import separate_audio
	path = os.path.abspath(file) if isinstance(file, str) else str(hash(file))
	tmpl = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]

	if isinstance(file, str):
		file = open(file, "rb")
	with file:
		read = create_reader(file)
		offs = 60
		header_size = read(offs, 4)
		song_length = read(offs + 4, 2)
		loop_start = read(offs + 6, 2)
		channel_count = read(offs + 8, 2)
		pattern_count = read(offs + 10, 2)
		instrument_count = read(offs + 12, 2)
		flags = read(offs + 14, 2)
		is_amiga = not flags & 1
		# print(is_amiga)
		tempo = read(offs + 16, 2)
		bpm = read(offs + 18, 2)
		ordering = read(offs + 20, 256, decode=False)[:song_length]
		offs += header_size

		patterns = []
		pattern_lengths = []
		for i in range(pattern_count):
			pattern_head_size = read(offs, 4)
			pattern_rows = read(offs + 5, 2)
			pattern_size = read(offs + 7, 2)
			offs += pattern_head_size
			pattern = read(offs, pattern_size, decode=False)
			patterns.append(pattern)
			pattern_lengths.append(pattern_rows)
			offs += pattern_size

		instruments = []
		for i in range(instrument_count):
			instrument_head_size = read(offs, 4)
			name = read(offs + 4, 22, decode=False).rstrip(b"\x00").decode("utf-8").strip()
			instrument = SimpleNamespace(
				name=name,
				sample_count=0,
				midi=[46] * 96,
				pitches=[0] * 96,
			)
			instruments.append(instrument)
			sample_count = read(offs + 27, 2)
			if sample_count > 0:
				sample_header_size = read(offs + 29, 4)
			instrument.sample_count = sample_count
			sample_mapping = read(offs + 33, 96, decode=False) if sample_count > 0 else [0] * 96
			offs += instrument_head_size
			sample_sustains = []
			sample_sizes = []
			sample_pitches = []
			sample_bits = []
			for j in range(sample_count):
				sample_size = read(offs, 4)
				sample_loops = read(offs + 8, 4)
				sample_vol = read(offs + 12, 1)
				sample_sizes.append(sample_size)
				sample_fine = read(offs + 13, 1)
				if sample_fine >= 128:
					sample_fine -= 256
				sample_type = read(offs + 14, 1)
				sample_pitch = read(offs + 16, 1)
				if sample_pitch >= 128:
					sample_pitch -= 256
				sample_relative = sample_pitch + sample_fine / 128
				sample_pitches.append(sample_relative)
				sample_reserved = read(offs + 17, 1)
				sample_bit = 4 if sample_reserved == 0xad else 16 if sample_type & 16 else 8
				sample_bits.append(sample_bit)
				name = read(offs + 18, 22, decode=False).rstrip(b"\x00").decode("utf-8", "replace").strip()
				if sample_type & 3 and sample_loops:
					sample_sustains.append(True)
				else:
					sample_sustains.append(False)
				offs += sample_header_size
			for j in range(sample_count):
				size = sample_sizes[j]
				if not size:
					continue
				sample_delta = read(offs, size, decode=False)
				bits = sample_bits[j]
				if bits == 16:
					sample = np.clip(np.cumsum(np.frombuffer(sample_delta, dtype=np.int16), dtype=np.int16) / 32767, -1, 1)
				elif bits == 8:
					sample = np.clip(np.cumsum(np.frombuffer(sample_delta, dtype=np.int8), dtype=np.int8) / 127, -1, 1)
				else:
					raise NotImplementedError(bits)
				pitch = 2 ** (sample_pitches[j] / 12)
				sr = round(pitch * 8363)
				# print(name, sample_sustains[j], len(sample), sr / 2)
				if sample_sustains[j] and len(sample) < sr / 2:
					sample = np.tile(sample, ceil(sr / 2 / len(sample)))
				adjusted = librosa.resample(sample, orig_sr=sr, target_sr=in_sample_rate)

				if len(adjusted) >= in_sample_rate / 4 or not sample_sustains[j]:
					# XM carries NO reliable information about the instruments themselves; this means we have to analyse the samples to decide if they're percussion, and if they're not, analyse the actual pitch they play at!
					D = librosa.stft(adjusted)
					H, P = librosa.decompose.hpss(D)
					rms_harm = librosa.feature.rms(S=H, hop_length=len(adjusted), center=False)
					rms_perc = librosa.feature.rms(S=P, hop_length=len(adjusted), center=False)
					harm = np.sum(rms_harm)
					perc = np.sum(rms_perc)
					print(name, harm, perc, len(adjusted))
					if perc >= harm / 8:
						sample_file = f"{temp_dir}{tmpl}-{i}-{j}.flac"
						sf.write(sample_file, adjusted, in_sample_rate)
						if perc < harm / 3:
							# Instrument may still be percussion; call stem separation to verify
							output_names = {k: tmpl + f"-{i}-{j}-" + v for k, v in dict(
								Vocals="V",
								Drums="D",
								Bass="B",
								Other="O",
							).items()}
							separate_audio("htdemucs_ft.yaml", sample_file, output_names)
							drums, *_ = librosa.load(temp_dir + output_names["Drums"] + ".flac", sr=in_sample_rate, mono=True)
							rms_orig = librosa.feature.rms(y=adjusted, frame_length=len(adjusted), hop_length=len(adjusted))
							rms_drum = librosa.feature.rms(y=drums, frame_length=len(drums), hop_length=len(drums))
							orig = np.sum(rms_orig)
							drum = np.sum(rms_drum)
						else:
							drum, orig = perc, harm
						if drum >= orig / 3:
							# Instrument is more likely percussion; call further stem separation to identify
							output_names = {k: tmpl + f"-{i}-{j}-" + v for k, v in dict(
								Kick="K",
								Snare="S",
								Toms="T",
								HH="H",
								Ride="R",
								Crash="C",
							).items()}
							separate_audio("MDX23C-DrumSep-aufr33-jarredou.ckpt", sample_file, output_names)
							outputs = {k: librosa.load(temp_dir + v + ".flac", sr=in_sample_rate, mono=True)[0] for k, v in output_names.items()}
							scales = [0.5, 1, 1, 1.5, 2, 2]
							output_volumes = {k: scales[i] * np.sum(librosa.feature.rms(y=v, frame_length=len(v), hop_length=len(v))) for i, (k, v) in enumerate(outputs.items())}
							highest = np.argmax(tuple(output_volumes.values()))
							# target = tuple(output_volumes)[highest]
							notes = [35, 38, 47, 42, 46, 49]
							for k, p in enumerate(sample_mapping):
								if p == j:
									instrument.midi[k] = -1
									instrument.pitches[k] = notes[highest]
							offs += size
							continue
				# Even for melodic instruments, xm does not guarantee the notes being played or the relative/fine tuning would produce the actual pitch a human would hear. Therefore, we must once again analyse the instrument samples, this time for their individual fundamental frequency.
				f0, voiced_flag, _ = librosa.pyin(
					adjusted,
					sr=in_sample_rate,
					fmin=librosa.note_to_hz("C1"),
					fmax=librosa.note_to_hz("C8"),
					resolution=1,
					switch_prob=1,
					fill_na=0,
				)
				try:
					average_note = round(np.mean(librosa.hz_to_midi(f0[f0 != 0])).item())
				except ValueError:
					average_note = 60
				for k, p in enumerate(sample_mapping):
					if p == j:
						instrument.midi[k] = 73 if sample_sustains[j] else 46
						instrument.pitches[k] = k + average_note - 60 + 11
				offs += size
	events = [
		[0, 0, event_types.HEADER, 1, len(instruments) + 1, tempo],
		[1, 0, event_types.TEMPO, 2500000 / bpm * tempo],
	]
	@functools.lru_cache(maxsize=pattern_count)
	def decode_pattern(p):
		output = []
		tick = 0
		channel = 0
		i = 0
		last_params = {}

		def note_off(channel, original_pitch=0):
			if last_params.get(channel):
				pitch = last_params[channel]["pitch"]
				ins = last_params[channel]["ins"]
				instrument = instruments[ins - 1]
				iid = instrument.midi[pitch]
				output.append([channel, tick, event_types.NOTE_OFF_C, 9 if iid == -1 else channel + 10, instrument.pitches[pitch], 0])
				return pitch == original_pitch

		breakafter = False
		while i < len(p):
			priority = 2
			pitch = ins = vcol = eff = efp = 0
			dbyt = p[i]
			if dbyt >= 128:
				if dbyt & 1:
					i += 1
					pitch = p[i]
				if dbyt & 2:
					i += 1
					ins = p[i]
				if dbyt & 4:
					i += 1
					vcol = p[i]
				if dbyt & 8:
					i += 1
					eff = p[i]
				if dbyt & 16:
					i += 1
					efp = p[i]
				i += 1
			else:
				pitch = dbyt
				ins, vcol, eff, efp = p[i + 1:i + 5]
				i += 5
			if vcol < 16:
				try:
					vel = last_params[channel]["vel"]
				except KeyError:
					vel = 127
			elif vcol <= 96:
				vel = min(127, (vcol - 16) * 2)
			else:
				raise NotImplementedError(vel)
			if eff == 0xd:
				breakafter = True
			if pitch == 97:
				note_off(channel)
			elif pitch == 0:
				if ins == 0:
					try:
						ins = last_params[channel]["ins"]
					except KeyError:
						ins = 1
				instrument = instruments[ins - 1]
				iid = instrument.midi[pitch]
				output.append([channel, tick, event_types.CONTROL_C, 9 if iid == -1 else channel + 10, 7, vel])
			else:
				if pitch == 0:
					try:
						pitch = last_params[channel]["pitch"]
					except KeyError:
						pitch = 37
				if ins == 0:
					try:
						ins = last_params[channel]["ins"]
					except KeyError:
						ins = 1
				if note_off(channel, pitch):
					priority = 1
				instrument = instruments[ins - 1]
				iid = instrument.midi[pitch]
				events.append([channel, tick, event_types.PROGRAM_C, 9 if iid == -1 else channel + 10, iid])
				output.append([channel, tick, event_types.CONTROL_C, 9 if iid == -1 else channel + 10, 7, vel])
				output.append([channel, tick, event_types.NOTE_ON_C, 9 if iid == -1 else channel + 10, instrument.pitches[pitch], 127, priority])
				last_params[channel] = {"pitch": pitch, "ins": ins, "vel": vel}
			channel += 1
			if channel >= channel_count:
				channel = 0
				tick += tempo
				if breakafter:
					break
		for channel in range(channel_count):
			note_off(channel)
		return output, tick
	tick = 0
	for b in ordering:
		output, t = decode_pattern(patterns[b])
		for event in output:
			adjusted = event.copy()
			adjusted[1] += tick
			events.append(adjusted)
		tick += t
	return events


def save_org(transport, output, instrument_activities, speed_info, ctx, **void):
	print("Exporting ORG...")
	import struct
	instruments, wait = list(build_org(transport, instrument_activities, speed_info, ctx=ctx))
	nc = 0
	with open(output, "wb") as org:
		org.write(b"\x4f\x72\x67\x2d\x30\x32")
		org.write(struct.pack("<H", wait))
		org.write(b"\x04\x08" if wait >= 40 else b"\x08\x08")
		org.write(struct.pack("<L", 0))
		org.write(struct.pack("<L", ceil(len(transport) / 4) * 4))
		print([len(ins.notes) for ins in instruments])
		for i, ins in enumerate(instruments):
			org.write(struct.pack("<H", 1000 + (i + 1 >> 1) * (70 if i & 1 else -70)))
			org.write(struct.pack("B", ins.id))
			org.write(b"\x00")
			org.write(struct.pack("<H", len(ins.notes)))
			nc += len(ins.notes)
		for i, ins in enumerate(instruments):
			for note in ins.notes:
				org.write(struct.pack("<L", note.tick))
			for note in ins.notes:
				org.write(struct.pack("B", note.pitch))
			for note in ins.notes:
				org.write(struct.pack("B", note.length))
			for note in ins.notes:
				org.write(struct.pack("B", note.volume))
			for note in ins.notes:
				org.write(struct.pack("B", note.panning))
	return nc