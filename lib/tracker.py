from dataclasses import dataclass
import functools
from math import ceil, hypot
from types import SimpleNamespace
from .mappings import (
	org_instrument_selection, org_instrument_mapping,
	instrument_names, midi_instrument_selection,
	percussion_mats,
	c4, c1,
)
from .util import create_reader, transport_note_priority


@dataclass(slots=True)
class OrgNote:
	tick: int
	pitch: int
	length: int
	volume: int
	panning: int

def render_org(notes, instrument_activities, speed_info, ctx):
	orig_ms_per_clock, real_ms_per_clock, scale, orig_step_ms, _orig_tempo = speed_info
	speed_ratio = real_ms_per_clock / scale / orig_ms_per_clock
	wait = round(orig_step_ms / speed_ratio)
	activities = list(map(list, instrument_activities.items()))
	instruments = []
	if sum(t[1][1] for t in activities) >= 12:
		instruments.append(SimpleNamespace(
			id=60,
			index=0,
			type=9,
			notes=[],
		))
	while len(instruments) < 8:
		activities.sort(key=lambda t: t[1][0], reverse=True)
		curr = activities[0]
		curr[1][0] /= 2
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
	for i, drum in enumerate([0, 2, 5, 6, 4, 0, 0, 0]):
		instruments.append(SimpleNamespace(
			id=drum,
			index=len(instruments),
			type=-1,
			notes=[],
		))
	note_count = sum(map(len, notes))
	conservative = False
	if note_count > 4096 * 8 * 4:
		print(f"High note count {note_count} detected, using conservative note sustains...")
		conservative = True
	min_vol = 16
	max_vol = 127 + min_vol
	active = {}
	for i, beat in enumerate(notes):
		taken = []
		next_active = {}
		if beat:
			ordered = sorted(beat, key=lambda note: (transport_note_priority(note, sustained=(note[0], note[1] - c1) in active), note[1]), reverse=True)
			lowest = min((note[0] == -1, note[1], note) for note in beat)[-1]
			if len(ordered) >= 7 and org_instrument_selection[lowest[0]] >= 0:
				lowest_to_remove = True
				for j, note in enumerate(ordered):
					if note[0] == -1:
						continue
					if note == lowest and lowest_to_remove:
						ordered.pop(j)
						lowest_to_remove = False
					elif note[1] == lowest[1] + 12:
						vel = hypot(note[4], lowest[4]) * 3 / 2
						tvel = min(max_vol, vel)
						keep = lowest[4] - tvel * 2 / 3
						lowest = (9, lowest[1], max(note[2], lowest[2]), True, tvel, (note[5] + lowest[5]) / 2)
						ordered[j - 1 + lowest_to_remove] = (note[0], note[1], min(1, note[2]), note[3], max(1, keep), note[5])
						break
				if lowest[0] != 9 and lowest[1] < c4 - 12:
					pitch = lowest[1]
					lowest = (9, pitch, lowest[2], lowest[3], min(max_vol, lowest[4] * 3 / 2), lowest[5])
				ordered.sort(key=lambda note: (note[2] + round(note[4] * 8 / max_vol), note[1]), reverse=True)
				ordered.insert(0, lowest)
			elif len(ordered) > 1:
				ordered.remove(lowest)
				ordered.insert(0, lowest)
		else:
			ordered = list(beat)
		for note in ordered:
			itype, pitch, priority, _long, vel, pan = note
			vel -= min_vol
			volume = max(0, min(254, round(vel * 2 / 64) * 64 if conservative else round(vel * 2 / 8) * 8))
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
						if mat == "netherrack":
							iid = 8
							pitch = rpitch - 12
						else:
							iid = 12
							pitch = rpitch - 18
					case "snare":
						iid = 9
					case "hat":
						iid = 10
					case "creeper":
						iid = 11
					case _:
						iid = 12
						pitch = rpitch
				if iid in taken:
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
				taken.append(iid)
				continue
			new_pitch = pitch - c1
			if new_pitch < 0:
				new_pitch += 12
			if new_pitch < 0:
				new_pitch = 0
			if new_pitch > 95:
				new_pitch = 95
			h = (itype, new_pitch)
			if (priority < 2 or conservative) and h in active:
				try:
					for iid in active[h]:
						if iid in taken:
							continue
						instrument = instruments[iid]
						last_vol, last_pan = instrument.notes[-1].volume, instrument.notes[-1].panning
						idx = -1
						while (last_note := instrument.notes[idx]) and last_note.pitch == 255:
							idx -= 1
						if last_note.length < 192:
							if last_vol != volume or (last_pan != panning and not conservative):
								instrument.notes.append(OrgNote(
									tick=i,
									pitch=255,
									length=1,
									volume=volume,
									panning=last_pan if conservative else panning,
								))
							last_note.length += 1
							taken.append(iid)
							try:
								next_active[h].append(instrument.index)
							except KeyError:
								next_active[h] = [instrument.index]
							raise StopIteration
				except StopIteration:
					continue
			choices = sorted(instruments[:8], key=lambda instrument: (instrument.index not in taken, len(instrument.notes) < 3072, instrument.id == ins, instrument.id != 60, -(len(instrument.notes) // 1024)), reverse=True)
			instrument = choices[0]
			if instrument.index in taken:
				if len(taken) >= len(instruments):
					break
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
			taken.append(instrument.index)
			try:
				next_active[h].append(instrument.index)
			except KeyError:
				next_active[h] = [instrument.index]
		active = next_active
	return instruments, wait


def load_org(file):
	print("Importing ORG...")
	if isinstance(file, str):
		file = open(file, "rb")
	min_vol = 16
	with file:
		read = create_reader(file)
		wait = read(6, 2)
		# end = read(14, 4)
		instruments = []
		total_count = 0
		for i in range(16):
			iid = read(18 + i * 6 + 2, 2)
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
				sustain=sus,
			))
			total_count += count
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
	events = [
		[0, 0, "header", 1, len(instruments) + 1, 1],
		[1, 0, "tempo", wait * 1000],
	]
	for i, ins in enumerate(instruments):
		if i not in range(8, 16):
			events.append([i + 2, 0, "program_c", i, midi_instrument_selection[org_instrument_mapping[ins.id]]])
			channel = i
			pitch = None
		else:
			channel = 9
			if i == 8:
				events.append([i + 2, 0, "program_c", channel, 0])
			pitch = [35, 38, 42, 46, 47, 73, 35, 35][i - 8]
		for j, e in enumerate(ins.events):
			if e.panning != 255:
				events.append([i + 2, e.tick, "control_c", channel, 10, (e.panning - 6) / 6 * 63 + 64])
			if e.volume != 255:
				events.append([i + 2, e.tick, "control_c", channel, 7, e.volume / 254 * (127 - min_vol) + min_vol])
			if e.pitch != 255:
				events.append([i + 2, e.tick, "note_on_c", channel, pitch or e.pitch + c1, 127, 1, e.length])
	return events

# TODO: Finish implementation
def load_xm(file):
	print("Importing XM...")
	if isinstance(file, str):
		file = open(file, "rb")
	with file:
		read = create_reader(file)
		offs = 60
		header_size = read(offs, 4)
		song_length = read(offs + 4, 2)
		channel_count = read(offs + 8, 2)
		pattern_count = read(offs + 10, 2)
		instrument_count = read(offs + 12, 2)
		tempo = read(offs + 16, 2)
		ordering = read(offs + 20, 256, decode=False)
		offs += header_size

		patterns = []
		for i in range(pattern_count):
			pattern_head_size = read(offs, 4)
			pattern_rows = read(offs + 5, 2)
			pattern_size = read(offs + 7, 2)
			print(i, pattern_rows, pattern_size)
			offs += pattern_head_size
			pattern = read(offs, pattern_size, decode=False)
			patterns.append(pattern)
			offs += pattern_size

		instruments = []
		for i in range(instrument_count):
			instrument_head_size = read(offs, 4)
			name = read(offs + 4, 22, decode=False).rstrip(b"\x00").decode("utf-8").strip()
			instrument = SimpleNamespace(
				name=name,
				sustain=False,
			)
			instruments.append(instrument)
			sample_count = read(offs + 27, 2)
			if sample_count > 0:
				sample_header_size = read(offs + 29, 4)
			offs += instrument_head_size
			sample_sizes = []
			for j in range(sample_count):
				sample_size = read(offs, 4)
				sample_type = read(offs + 14, 1)
				sample_sizes.append(sample_size)
				name = read(offs + 18, 22, decode=False).rstrip(b"\x00").decode("utf-8").strip()
				if sample_type & 3:
					instrument.sustain = True
				offs += sample_header_size
			offs += sum(sample_sizes)
	events = [
		[0, 0, "header", 1, len(instruments) + 1, 8],
		[1, 0, "tempo", tempo * 1000 * 8],
	]
	@functools.lru_cache(maxsize=pattern_count)
	def decode_pattern(p):
		pattern = []
		i = 0
		while i < len(p):
			pitch = p[i]
			assert pitch < 128, "Pattern compression not yet implemented!"
			ins, vcol, eff, efp = p[i + 1:i + 5]
			i += 5
			pattern.append([0, len(pattern), "note_on_c", channel, pitch or e.pitch + c1, 127, 1, e.length])
	for b in ordering:
		for event in decode_pattern(patterns[b]):
			pass
	for i, ins in enumerate(instruments):
		if i not in range(8, 16):
			events.append([i + 2, 0, "program_c", i, midi_instrument_selection[org_instrument_mapping[ins.id]]])
			channel = i
			pitch = None
		else:
			channel = 9
			if i == 8:
				events.append([i + 2, 0, "program_c", channel, 0])
			pitch = [35, 38, 42, 46, 47, 73, 35, 35][i - 8]
		for j, e in enumerate(ins.events):
			if e.panning != 255:
				events.append([i + 2, e.tick, "control_c", channel, 10, (e.panning - 6) / 6 * 63 + 64])
			if e.volume != 255:
				events.append([i + 2, e.tick, "control_c", channel, 7, e.volume / 254 * (127 - min_vol) + min_vol])
			if e.pitch != 255:
				events.append([i + 2, e.tick, "note_on_c", channel, pitch or e.pitch + c1, 127, 1, e.length])


def save_org(transport, output, instrument_activities, speed_info, ctx):
	print("Exporting ORG...")
	import struct
	instruments, wait = list(render_org(list(transport), instrument_activities, speed_info, ctx=ctx))
	nc = 0
	with open(output, "wb") as org:
		org.write(b"\x4f\x72\x67\x2d\x30\x32")
		org.write(struct.pack("<H", wait))
		org.write(b"\x04\x08" if wait >= 40 else b"\x08\x08")
		org.write(struct.pack("<L", 0))
		org.write(struct.pack("<L", ceil(len(transport) / 4) * 4))
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