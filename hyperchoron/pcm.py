import bisect
import contextlib
from dataclasses import dataclass
from math import ceil, isfinite
import functools
import os
import shutil
import subprocess
from types import SimpleNamespace
import numpy as np
from .mappings import c1, c4, a4, nbs_raws
from .util import base_path, temp_dir, ts_us, in_sample_rate, out_sample_rate, fluidsynth, create_reader, orgexport, ensure_synths, get_sf2, lin2log, log2lin, round_random


writer_sr = 36000

bps = 2
bpo = bps * 12
octaves = 7
semitone = 2 ** (1 / 12)

@dataclass(slots=True)
class PCMNote:
	pitch: int
	velocity: float
	tick: int

def separate_audio(model, file, outputs):
	global separator
	for o in outputs.values():
		target = temp_dir + o + ".flac"
		if not os.path.exists(target):
			print(target)
			break
	else:
		return
	if not globals().get("separator"):
		from audio_separator.separator import Separator
		separator = Separator(output_dir=temp_dir, output_format="FLAC", sample_rate=in_sample_rate, use_soundfile=True)
	separator.load_model(model)
	with contextlib.chdir(temp_dir):
		return separator.separate(file, outputs)


def load_raw(file):
	print(f"Importing {file.rsplit('.', 1)[-1].upper()}...")
	path = os.path.abspath(file)
	tmpl = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]

	import librosa
	song, *_ = librosa.load(path, sr=in_sample_rate, mono=True, dtype=np.float32)
	bpm_, *_ = librosa.beat.beat_track(y=song.astype(np.float16), sr=in_sample_rate)
	bpm = bpm_[0]
	print("Detected BPM:", bpm)

	output_names = {k: tmpl + "-" + v for k, v in {
		"No Reverb": "N",
		"Reverb": "_R",
	}.items()}
	separate_audio("UVR-DeEcho-DeReverb.pth", path, output_names)
	dry = temp_dir + output_names["No Reverb"] + ".flac"

	output_names = {k: tmpl + "-" + v for k, v in dict(
		Vocals="V",
		Instrumental="_I",
	).items()}
	separate_audio("model_bs_roformer_ep_317_sdr_12.9755.ckpt", dry, output_names)
	instrumentals = temp_dir + output_names["Instrumental"] + ".flac"

	output_names = {k: tmpl + "-" + v for k, v in dict(
		Vocals="_V",
		Drums="_D",
		Bass="B",
		Other="O",
	).items()}
	separate_audio("htdemucs_ft.yaml", instrumentals, output_names)
	# others = output_names["Other"] + ".flac"
	drums = temp_dir + output_names["Drums"] + ".flac"

	output_names = {k: tmpl + "-" + v for k, v in dict(
		Kick="K",
		Snare="S",
		Toms="T",
		HH="H",
		Ride="R",
		Crash="C",
	).items()}
	separate_audio("MDX23C-DrumSep-aufr33-jarredou.ckpt", drums, output_names)

	# TODO: Find a better model to break down the remaining instruments
	# output_names = {k: tmpl + "-" + v for k, v in dict(
	# 	Vocals="_V2",
	# 	Drums="_D2",
	# 	Bass="_B",
	# 	Guitar="G",
	# 	Piano="P",
	# 	Other="O",
	# ).items()}
	# separate_audio("htdemucs_6s.yaml", others, output_names)

	bufsize = round(in_sample_rate / (bpm / 60) / 12)

	def decompose_stem(fn, instrument=0, pitch=None, monophonic=True, pitch_range=("C1", "C7"), tolerance=256, mult=1):
		ins = instrument if instrument != -1 else 9
		events = [[ins, 0, "program_c", ins, instrument]]
		song, *_ = librosa.load(temp_dir + fn + ".flac", sr=in_sample_rate, mono=False, dtype=np.float32)
		mono = librosa.to_mono(song)
		volumes = librosa.feature.rms(y=mono, frame_length=bufsize * 4, hop_length=bufsize)[0]
		max_volume = np.max(volumes)
		if monophonic:
			if pitch is not None:
				for tick, v in enumerate(volumes):
					if v < max_volume / tolerance:
						continue
					if volumes[tick - 1] < v * 1 / 2:
						if tick < len(volumes) - 1 and volumes[tick + 1] >= v and volumes[tick + 1] < v * 2:
							v = volumes[tick + 1]
						v = lin2log(min(1, v * mult))
						events.append([ins, tick, "note_on_c", ins, pitch, v * 127, 2, 1])
				return events
			left, right = song[0], song[1]
			pannings = np.array([(R - L) / max(L, R) if max(
				(R := np.mean(np.abs(right[i:i + bufsize]))),
				(L := np.mean(np.abs(left[i:i + bufsize]))),
			) > 0 else 0 for i in range(0, len(right), bufsize)])
			f0, voiced_flag, _ = librosa.pyin(
				mono,
				sr=in_sample_rate,
				fmin=librosa.note_to_hz(pitch_range[0]),
				fmax=librosa.note_to_hz(pitch_range[1]),
				frame_length=bufsize * 4,
				hop_length=bufsize,
				resolution=1 / 3,
				switch_prob=0.5,
				fill_na=0,
			)
			notes = np.round(librosa.hz_to_midi(f0) * 6) / 6
			for tick, note in enumerate(notes):
				v = volumes[tick]
				if v < max_volume / tolerance:
					continue
				if not isfinite(note):
					continue
				if tick == 0 or volumes[tick - 1] < v * 2 / 3:
					priority = 2
				elif notes[tick - 1] != note:
					priority = 1
				else:
					priority = 0
				v = lin2log(min(1, v * mult))
				events.append([ins, tick, "note_on_c", ins, note, v * 127, priority, 1, pannings[tick]])
			return events
		c = librosa.hybrid_cqt(
			mono,
			sr=in_sample_rate,
			fmin=librosa.note_to_hz(pitch_range[0]),
			n_bins=bpo * octaves,
			bins_per_octave=bpo,
			hop_length=bufsize,
		)
		amp = np.abs(c.T, dtype=np.float32)
		eq = np.concatenate([np.arange(1, 1 + len(amp[0]) // 2), np.arange(len(amp[0]), len(amp[0]) // 2, -1) / 2])
		eq **= 2
		eq *= 0.5 / np.max(eq)
		eq += 0.5
		amp *= np.tile(eq, (len(amp), 1))
		active_notes = {}
		for tick, bins in enumerate(amp):
			volume = volumes[tick] if tick < len(volumes) else volumes[-1]
			chord = bins
			high = np.max(chord)
			if high <= 1 / 256:
				continue
			chord *= volume / max_volume / high * mult
			clipped = np.clip(chord, 0, 1)
			inds = list(np.argsort(clipped))
			for attempt in range(32):
				i = inds.pop(-1)
				v = clipped[i]
				if v < 1 / tolerance:
					break
				if active_notes.get(i, 0) >= v * 1.0625:
					priority = -1
				elif active_notes.get(i, 0) <= v * 0.25:
					priority = 2
				else:
					priority = 1
				active_notes[i] = v
				p = i / bps + c1
				v = lin2log(min(1, v * mult))
				events.append([ins, tick, "note_on_c", ins, p, v * 127, priority, 1])
		return events

	events = [
		[0, 0, "header", 1, 1 + 1, 1],
		[1, 0, "tempo", bufsize / in_sample_rate * 1000 * 1000 * 1],
	]
	print("Decomposing Bass...")
	events.extend(decompose_stem(tmpl + "-B", 46, monophonic=True, pitch_range=("C0", "C6"), mult=3))
	print("Decomposing Voice...")
	events.extend(decompose_stem(tmpl + "-V", 52, monophonic=False, pitch_range=("C1", "C8"), tolerance=16, mult=1.5))
	# events.extend(decompose_stem(tmpl + "-G", 48, monophonic=True, pitch_range=("C2", "C7")))
	print("Decomposing Others...")
	events.extend(decompose_stem(tmpl + "-O", 11, monophonic=False, pitch_range=("C2", "C9"), tolerance=12, mult=2))
	# events.extend(decompose_stem(tmpl + "-P", 0, monophonic=True, pitch_range=("C1", "C8")))
	print("Decomposing Drums...")
	events.extend(decompose_stem(tmpl + "-K", -1, monophonic=True, pitch=35, tolerance=24, mult=4))
	events.extend(decompose_stem(tmpl + "-S", -1, monophonic=True, pitch=38, tolerance=24, mult=4))
	events.extend(decompose_stem(tmpl + "-T", -1, monophonic=True, pitch=47, tolerance=24, mult=4))
	events.extend(decompose_stem(tmpl + "-H", -1, monophonic=True, pitch=42, tolerance=24, mult=4))
	events.extend(decompose_stem(tmpl + "-R", -1, monophonic=True, pitch=46, tolerance=24, mult=4))
	events.extend(decompose_stem(tmpl + "-C", -1, monophonic=True, pitch=49, tolerance=24, mult=4))
	return events


def ffmpeg_output(fo):
	ext = fo.rsplit(".", 1)[-1]
	extra = ["-ar", str(out_sample_rate)]
	match ext:
		case "ogg" | "opus":
			ba = "160k"
			extra = ["-f", ext, "-c:a", "libopus"]
		case "aac":
			ba = "192k"
			extra = ["-cutoff", "16000", "-c:a", "libfdk_aac"]
		case "mp3":
			ba = "224k"
			extra = ["-c:a", "libmp3lame"]
		case _:
			return ["-af", "alimiter", fo]
	return extra + ["-b:a", ba, "-vbr", "on", "-af", "alimiter", fo]


def render_midi(inputs, outputs, fmt="flac"):
	sf2 = get_sf2()
	convert = fmt not in ("wav", "flac")
	if convert:
		intermediate = [temp_dir + str(ts_us()) + str(i) + ".flac" for i, fo in enumerate(outputs)]
	else:
		intermediate = outputs
	procs = []
	for fi, fo in zip(inputs, intermediate):
		args = [fluidsynth, "-g", "0.25" if convert else "1", "-F", fo, "-c", "64", "-o", "synth.polyphony=32767", "-r", str(writer_sr if convert else out_sample_rate), "-n", sf2, fi]
		proc = subprocess.Popen(args, stdin=subprocess.DEVNULL)
		procs.append(proc)
	for proc in procs:
		proc.wait()
	procs.clear()
	if not convert:
		assert all(os.path.exists(fo) and os.path.getsize(fo) for fo in outputs)
		return outputs
	import imageio_ffmpeg as ii
	ffmpeg = ii.get_ffmpeg_exe()
	for fi, fo in zip(intermediate, outputs):
		args = [ffmpeg, "-y", "-i", fi, "-af", "volume=4", *ffmpeg_output(fo)]
		proc = subprocess.Popen(args, stdin=subprocess.DEVNULL)
		procs.append(proc)
	for proc in procs:
		proc.wait()
	assert all(os.path.exists(fo) and os.path.getsize(fo) for fo in outputs)
	return outputs


def render_nbs(inputs, outputs, fmt="flac"):
	import concurrent.futures
	import zipfile
	from audiotsm import phasevocoder
	from audiotsm.io.array import ArrayReader, ArrayWriter
	from librosa import load, resample
	from soundfile import SoundFile
	from .minecraft import segment_nbs
	@functools.lru_cache(maxsize=192)
	def load_audio(fn):
		with zipfile.ZipFile(f"{base_path}templates/Notes.zip", "r") as z:
			with z.open(fn, "r") as f:
				a = load(f, sr=writer_sr)[0]
		# Cut silence at end of audio to minimise later processing
		last = len(a) - 1 - np.argmax(a[::-1] != 0)
		a = a[:last]
		a *= 0.5
		return a
	@functools.lru_cache(maxsize=1024)
	def _pitch_audio(fn, p):
		a = load_audio(f"{fn}.ogg")
		speed = 1 / 2 ** (p / 12)
		if -12 <= p <= 12:
			a = resample(a, orig_sr=writer_sr, target_sr=round(writer_sr * speed), res_type="soxr_hq", fix=False)
		elif p < 0:
			p += 12
			r = ArrayReader(a.reshape((1, len(a))))
			w = ArrayWriter(1)
			tsm = phasevocoder(r.channels, speed=speed)
			tsm.run(r, w)
			a = w.data.ravel()
			a = resample(a, orig_sr=writer_sr, target_sr=round(writer_sr * speed), res_type="soxr_hq", fix=False)
		else:
			a = resample(a, orig_sr=writer_sr, target_sr=round(writer_sr * speed), res_type="soxr_hq", fix=False)
			p -= 12
			r = ArrayReader(a.reshape((1, len(a))))
			w = ArrayWriter(1)
			tsm = phasevocoder(r.channels, speed=speed)
			tsm.run(r, w)
			a = w.data.ravel()
		return a.astype(np.float16)
	@functools.lru_cache(maxsize=192)
	def pitch_audio(fn, p):
		return _pitch_audio(fn, p).astype(np.float32)
	for fi, fo in zip(inputs, outputs):
		buf_seconds = 2
		bufsize = buf_seconds * writer_sr
		# We keep an additional 3 seconds of buffer space to account for sounds that last a while
		buffer = np.zeros((bufsize + 3 * writer_sr, 2), dtype=np.float32)
		sides = [buffer[:, 0], buffer[:, 1]]
		queue = [[], []]
		# Use a reader and writer thread to minimise overhead of file IO + sound resampling
		with concurrent.futures.ThreadPoolExecutor(max_workers=1) as wx:
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as rx:
				def render_note(side, idx, rendered, vel=1):
					sides[side][idx:idx + len(rendered)] += (rendered if vel == 1 else rendered * vel)
				def render_note_block(side, idx, instrument, pitch, velocity, panning):
					fn = nbs_raws[3 if instrument > 15 else instrument]
					p = pitch / 100 - 33 - 12
					if p < -60:
						p = -60 + p % 12
					elif p >= 72:
						p = 60 + p % 12
					a = pitch_audio(fn, p)
					t = (panning / 100 + 1) * np.pi / 4
					mult = velocity * (np.sin(t) if side else np.cos(t))
					render_note(side, idx, a, mult)
				def update_queue(side):
					for (idx, instrument, pitch, velocity, panning) in queue[side]:
						render_note_block(side, idx, instrument, pitch, velocity, panning)
					queue[side].clear()
				with SoundFile(fo, "w", writer_sr, channels=2, format=fmt) as f:
					futs = []
					for tempo, segment in segment_nbs(fi):
						pos = segment[0][0]
						for tick, chord in segment:
							while (tick - pos) / tempo >= buf_seconds:
								fut = rx.submit(update_queue, 0)
								update_queue(1)
								fut.result()
								for fut in futs:
									fut.result()
								futs.clear()
								fut = wx.submit(f.write, buffer[:bufsize].copy())
								futs.clear()
								futs.append(fut)
								buffer[:-bufsize] = buffer[bufsize:]
								buffer[-bufsize:] = 0
								pos += buf_seconds * tempo
							stats = {}
							for note in chord:
								if note.velocity == 0:
									continue
								pitch = note.key * 100 + note.pitch
								tup = (note.instrument, pitch, note.panning)
								stats[tup] = stats.get(tup, 0) + note.velocity / 100
							for (instrument, pitch, panning), velocity in stats.items():
								idx = round_random(writer_sr * (tick - pos) / tempo)
								queue[0].append((idx, instrument, pitch, velocity, panning))
								idx = round_random(writer_sr * (tick - pos) / tempo)
								queue[1].append((idx, instrument, pitch, velocity, panning))
						tick += 1
						extra_buff = round_random((tick - pos) / tempo * writer_sr)
						if extra_buff:
							fut = rx.submit(update_queue, 0)
							update_queue(1)
							fut.result()
							for fut in futs:
								fut.result()
							futs.clear()
							fut = wx.submit(f.write, buffer[:extra_buff].copy())
							futs.append(fut)
							buffer[:-extra_buff] = buffer[extra_buff:]
							buffer[-extra_buff:] = 0


def render_org(inputs, outputs, fmt="wav"):
	from .tracker import read_org
	from librosa import resample
	drum_sr = 22050
	WDB = None
	def load_wdb():
		nonlocal WDB
		if WDB:
			return WDB
		nwaves = 100
		wdb = dict(
			wave100=np.empty((nwaves, 256), dtype=np.float32),
			drums=[],
		)
		import struct
		import zipfile
		with zipfile.ZipFile(f"{base_path}templates/soundbank.zip", "r") as z:
			with z.open(z.filelist[0], "r") as f:
				data = memoryview(f.read())
		for i in range(nwaves):
			wdb["wave100"][i] = np.frombuffer(data[i * 256:(i + 1) * 256], dtype=np.int8)
		wdb["wave100"] *= 1 / 127 / 4
		data = data[nwaves * 256:]
		while data:
			length = struct.unpack("<L", data[:4])[0]
			drumdata = data[4:4 + length]
			data = data[4 + length:]
			assert len(drumdata) == length, "Incomplete drum samples in wdb file!"
			drumsample = (np.frombuffer(drumdata, dtype=np.int8) ^ 127).astype(np.float32) * (1 / 127 / 4)
			wdb["drums"].append(drumsample)
		WDB = wdb
		return wdb
	@functools.lru_cache(maxsize=1024)
	def render_org_wave(iid, freq, length, target_length):
		sample = WDB["wave100"][iid]
		target_len = len(sample) * freq * length
		wave = np.tile(sample, ceil(target_len / len(sample)))[:round(target_len)]
		for idx in range(len(sample) // 2):
			if wave[idx] in range(-1, 2):
				break
		declick_start = wave[:idx]
		declick_start *= (1 + np.tanh(np.linspace(-np.pi, np.pi, num=len(declick_start), dtype=np.float32))) * 0.5
		for idx in range(len(wave) - 1, len(wave) - 1 - len(sample) // 2, -1):
			if wave[idx] in range(-1, 2):
				break
		declick_end = wave[idx:]
		declick_end *= (1 - np.tanh(np.linspace(-np.pi, np.pi, num=len(declick_end), dtype=np.float32))) * 0.5
		wave = resample(wave, orig_sr=len(wave), target_sr=target_length, res_type="soxr_hq", fix=False)
		return wave.astype(np.float16)
	@functools.lru_cache(maxsize=1024)
	def render_org_drum(iid, freq, length, target_length):
		wave = WDB["drums"][iid]
		orig_sr = freq
		# print(orig_sr, writer_sr)
		wave = resample(wave, orig_sr=orig_sr, target_sr=writer_sr, res_type="soxr_hq", fix=False)
		return wave.astype(np.float16)
	volume_table = np.arange(256, dtype=np.float32)
	volume_table *= 1 / 255
	volume_table -= 1
	np.power(10, volume_table, out=volume_table)
	@functools.lru_cache(maxsize=1024)
	def resamp_vols(volumes, target_length):
		volumes = volume_table[np.uint8(volumes)]
		x_old = np.linspace(0, 1, len(volumes))
		x_new = np.linspace(0, 1, target_length)
		return np.interp(x_new, x_old, volumes)
	load_wdb()
	procs = []
	for fi, fo in zip(inputs, outputs):
		orgfile = read_org(fi)
		final_notes = []
		active_notes = []
		for ins in orgfile.instruments:
			for j, e in enumerate(ins.events):
				if e.pitch != 255:
					if ins.drum:
						e.freq = (e.pitch + ins.finetune / 1000) * 800 + 100
						min_length = ceil(e.freq / writer_sr / orgfile.wait * 1000)
					else:
						e.freq = 1760 * 2 ** ((e.pitch + ins.finetune / 1000 - 1 - a4) / 12)
						min_length = 1
					e.length = max(e.length, min_length) if ins.sustain else 1
					e.volumes = np.full(e.length, e.volume, dtype=np.uint8)
					for k in range(len(active_notes)):
						e2 = active_notes[-k - 1]
						if e.tick >= e2.tick + e2.length:
							final_notes.extend(active_notes[:-k - 1])
							active_notes = active_notes[-k - 1:]
							break
					bisect.insort_right(active_notes, e, key=lambda ev: ev.tick + ev.length)
					active_notes.append(e)
				else:
					if e.volume != 255:
						for k in range(len(active_notes)):
							e2 = active_notes[-k - 1]
							if e.tick >= e2.tick + e2.length:
								final_notes.extend(active_notes[:-k - 1])
								active_notes = active_notes[-k - 1:]
								break
							if e.instrument != e2.instrument:
								continue
							diff = e.tick - e2.tick
							e2.volumes[diff:] = e.volume
		final_notes.extend(active_notes)
		buffer = np.zeros((ceil(orgfile.end * orgfile.wait / 1000 * writer_sr), 2), dtype=np.float32)
		stacks = np.zeros(len(buffer), dtype=np.uint8)
		for e in final_notes:
			pos = round(e.tick * orgfile.wait / 1000 * writer_sr)
			stop = round((e.tick + e.length) * orgfile.wait / 1000 * writer_sr)
			target_length = stop - pos
			if target_length <= 0:
				continue
			ins = orgfile.instruments[e.instrument]
			if ins.drum:
				wave = render_org_drum(ins.id, e.freq, e.length * orgfile.wait / 1000, target_length)
			else:
				wave = render_org_wave(ins.id, e.freq, e.length * orgfile.wait / 1000, target_length)
			stop = min(len(buffer), pos + len(wave))
			if stop <= pos:
				continue
			wave = wave[:stop - pos]
			vols = resamp_vols(tuple(e.volumes), target_length=target_length)[:len(wave)]
			wave = wave.astype(np.float32)
			wave[:len(vols)] *= vols
			if len(vols) and vols[-1] != 1:
				wave[len(vols):] *= vols[-1]
			left = right = wave
			buffer[pos:stop, 0] += left
			buffer[pos:stop, 1] += right
			stacks[pos:stop] += 1
		import imageio_ffmpeg as ii
		ffmpeg = ii.get_ffmpeg_exe()
		args = [ffmpeg, "-y", "-f", "f32le", "-ar", str(writer_sr), "-ac", "2", "-i", "-", *ffmpeg_output(fo)]
		proc = subprocess.Popen(args, stdin=subprocess.PIPE)
		proc.communicate(buffer.data)
		procs.append(proc)
	for proc in procs:
		proc.wait()
	assert all(os.path.exists(fo) and os.path.getsize(fo) for fo in outputs)
	return outputs


def render_xm(inputs, outputs, fmt="flac"):
	import imageio_ffmpeg as ii
	ffmpeg = ii.get_ffmpeg_exe()
	procs = []
	for fi, fo in zip(inputs, outputs):
		args = [ffmpeg, "-y", "-i", fi, *ffmpeg_output(fo)]
		proc = subprocess.Popen(args, stdin=subprocess.DEVNULL)
		procs.append(proc)
	for proc in procs:
		proc.wait()
	assert all(os.path.exists(fo) and os.path.getsize(fo) for fo in outputs)
	return outputs


def mix_raw(inputs, output):
	if len(inputs) == 1 and inputs[0].rsplit(".", 1)[-1] == output.rsplit(".", 1)[-1]:
		try:
			os.replace(inputs[0], output)
		except (OSError, PermissionError):
			shutil.copyfile(inputs[0], output)
		return output
	import imageio_ffmpeg as ii
	ffmpeg = ii.get_ffmpeg_exe()
	args = [ffmpeg, "-y"]
	for fi in inputs:
		args.extend(("-i", fi))
	if len(inputs) > 1:
		args.extend(("-filter_complex", "".join(f"[{i}:a]" for i in range(len(inputs))) + f"amix=inputs={len(inputs)}:duration=longest"))
	args.extend(ffmpeg_output(output))
	subprocess.check_output(args)
	assert os.path.exists(output) and os.path.getsize(output)
	return output


def save_raw(transport, output, ctx, speed_info, instrument_activities, **void):
	print(f"Exporting {output.rsplit('.', 1)[-1].upper()}...")
	from .util import Transport
	modalities = {}
	for beat in transport:
		for t in modalities.values():
			t.append([])
		for note in beat:
			m = note.modality
			try:
				modalities[m][-1].append(note)
			except KeyError:
				modalities[m] = Transport(tick_delay=transport.tick_delay)
				modalities[m].append([note])
	ofmt = output.rsplit(".", 1)[-1]
	ofi = ".flac" if ofmt != "wav" else "." + ofmt
	nc = 0
	outputs = []
	if 0 in modalities:
		m_transport = modalities.pop(0)
		tmpl = temp_dir + str(ts_us()) + "0"
		m_out = tmpl + ".mid"
		from . import midi
		m_out_f = tmpl + ofi
		nc += midi.save_midi(m_transport, m_out, speed_info=speed_info, instrument_activities=instrument_activities, ctx=ctx)
		render_midi([m_out], [m_out_f], fmt=ofi[1:])
		outputs.append(m_out_f)
	if 1 in modalities:
		m_transport = modalities.pop(1)
		tmpl = temp_dir + str(ts_us()) + "1"
		m_out = tmpl + ".nbs"
		from . import minecraft
		m_out_f = tmpl + ofi
		nc += minecraft.save_nbs(m_transport, m_out, speed_info=speed_info, instrument_activities=instrument_activities, ctx=ctx)
		render_nbs([m_out], [m_out_f], fmt=ofi[1:])
		outputs.append(m_out_f)
	if 16 in modalities:
		m_transport = modalities.pop(16)
		tmpl = temp_dir + str(ts_us()) + "16"
		m_out = tmpl + ".org"
		from . import tracker
		m_out_f = tmpl + ofi
		nc += tracker.save_org(m_transport, m_out, speed_info=speed_info, instrument_activities=instrument_activities, ctx=ctx)
		render_org([m_out], [m_out_f], fmt=ofi[1:])
		outputs.append(m_out_f)
	if modalities:
		raise NotImplementedError(tuple(modalities))
	mix_raw(outputs, output)
	return nc