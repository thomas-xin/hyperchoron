import contextlib
from dataclasses import dataclass
from math import isfinite
import os
import librosa
import numpy as np
from .mappings import c1
from .util import temp_dir, sample_rate


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
		separator = Separator(output_dir=temp_dir, output_format="FLAC", sample_rate=sample_rate, use_soundfile=True)
	separator.load_model(model)
	with contextlib.chdir(temp_dir):
		return separator.separate(file, outputs)


def load_wav(file, ctx):
	print(f"Importing {file.rsplit('.', 1)[-1].upper()}...")
	path = os.path.abspath(file)
	tmpl = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]

	if ctx.mc_legal:
		bpm = 20 * 5
	else:
		song, *_ = librosa.load(path, sr=sample_rate, mono=True, dtype=np.float32)
		bpm_, *_ = librosa.beat.beat_track(y=song.astype(np.float16), sr=sample_rate)
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

	bufsize = round(sample_rate / (bpm / 60) / 12)

	def decompose_stem(fn, instrument=0, pitch=None, monophonic=True, pitch_range=("C1", "C7"), tolerance=256, mult=1):
		ins = instrument if instrument != -1 else 9
		events = [[ins, 0, "program_c", ins, instrument]]
		song, *_ = librosa.load(temp_dir + fn + ".flac", sr=sample_rate, mono=False, dtype=np.float32)
		mono = librosa.to_mono(song)
		# volumes = np.array([np.mean(np.abs(mono[i:i + bufsize])) for i in range(0, len(mono), bufsize)])
		volumes = librosa.feature.rms(y=mono, frame_length=bufsize * 4, hop_length=bufsize)[0]
		max_volume = np.max(volumes)
		if monophonic:
			if pitch is not None:
				for tick, v in enumerate(volumes):
					# if tick <= 0:
					# 	continue
					if v < max_volume / tolerance:
						continue
					if volumes[tick - 1] < v * 1 / 2:
						if tick < len(volumes) - 1 and volumes[tick + 1] >= v and volumes[tick + 1] < v * 2:
							v = volumes[tick + 1]
						v = min(1, v * mult)
						events.append([ins, tick, "note_on_c", ins, pitch, v * 127, 2, 1])
				return events
			left, right = song[0], song[1]
			pannings = np.array([(R - L) / max(L, R) if max(
				(R := np.mean(np.abs(right[i:i + bufsize]))),
				(L := np.mean(np.abs(left[i:i + bufsize]))),
			) > 0 else 0 for i in range(0, len(right), bufsize)])
			f0, voiced_flag, _ = librosa.pyin(
				mono,
				sr=sample_rate,
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
				v = min(1, v * mult)
				events.append([ins, tick, "note_on_c", ins, note, v * 127, priority, 1, pannings[tick]])
			return events
		c = librosa.hybrid_cqt(
			mono,
			sr=sample_rate,
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
				events.append([ins, tick, "note_on_c", ins, p, v * 127, priority, 1])
		return events

	events = [
		[0, 0, "header", 1, 1 + 1, 1],
		[1, 0, "tempo", bufsize / sample_rate * 1000 * 1000 * 1],
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