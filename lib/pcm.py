from dataclasses import dataclass
import io
import math
import wave
import librosa
import numpy as np
import pywt
from scipy import signal
from .mappings import c1, harmonics


# BUFSIZE = 750
bpo = 24
octaves = 7
semitone = 2 ** (1 / 12)

# def load_templates():
# 	instruments = {}
# 	for fn in os.listdir("lib/wav"):
# 		instrument = SimpleNamespace(name=fn.rsplit(".", 1)[0], base_freq=0, sample=None)
# 		data = io.BytesIO()
# 		with wave.open("lib/wav/" + fn, "rb") as w:
# 			assert (sr := w.getframerate()) == 48000
# 			assert (_nc := w.getnchannels()) == 2
# 			while True:
# 				b = w.readframes(sr)
# 				if not b:
# 					break
# 				data.write(b)
# 		buffer = data.getbuffer()
# 		a = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
# 		merged = a[::2] + a[1::2]
# 		merged *= 1 / 32767
# 		np.clip(merged, -1, None, out=merged)
# 		c = librosa.cqt(merged, sr=sr, fmin=librosa.note_to_hz("C1"), n_bins=bpo * octaves, bins_per_octave=bpo, hop_length=BUFSIZE)
# 		amp = np.abs(c, dtype=np.float64)
# 		if not np.any(amp):
# 			continue
# 		instrument.base_freq = np.argmax(amp.T[0])
# 		instrument.sample = amp / amp.T[0][instrument.base_freq]
# 		instruments[instrument.name] = instrument
# 	return instruments
# INSTRUMENTS = load_templates()

# Code below adapted from: https://github.com/scaperot/the-BPM-detector-python
# simple peak detection
def peak_detect(data):
	max_val = np.amax(abs(data))
	peak_ndx = np.where(data == max_val)
	if len(peak_ndx[0]) == 0:  # if nothing found then the max must be negative
		peak_ndx = np.where(data == -max_val)
	return peak_ndx

def bpm_detector(data, fs):
	cA = []
	cD = []
	correl = []
	cD_sum = []
	levels = 4
	max_decimation = 2 ** (levels - 1)
	min_ndx = math.floor(60.0 / 220 * (fs / max_decimation))
	max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))

	for loop in range(0, levels):
		cD = []
		# 1) DWT
		if loop == 0:
			[cA, cD] = pywt.dwt(data, "db4")
			cD_minlen = len(cD) / max_decimation + 1
			cD_sum = np.zeros(math.floor(cD_minlen))
		else:
			[cA, cD] = pywt.dwt(cA, "db4")

		# 2) Filter
		cD = signal.lfilter([0.01], [1 - 0.99], cD)
		# 4) Subtract out the mean.
		# 5) Decimate for reconstruction later.
		cD = abs(cD[:: (2 ** (levels - loop - 1))])
		cD = cD - np.mean(cD)

		# 6) Recombine the signal before ACF
		#    Essentially, each level the detail coefs (i.e. the HPF values) are concatenated to the beginning of the array
		cD_sum = cD[0 : math.floor(cD_minlen)] + cD_sum

	if [b for b in cA if b != 0.0] == []:
		return None, None

	# Adding in the approximate data as well...
	cA = signal.lfilter([0.01], [1 - 0.99], cA)
	cA = abs(cA)
	cA = cA - np.mean(cA)
	cD_sum = cA[0 : math.floor(cD_minlen)] + cD_sum

	# ACF
	correl = np.correlate(cD_sum, cD_sum, "full")

	midpoint = math.floor(len(correl) / 2)
	correl_midpoint_tmp = correl[midpoint:]
	peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])
	if len(peak_ndx) > 1:
		return None, None
	peak_ndx_adjusted = peak_ndx[0] + min_ndx
	bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)
	return bpm, correl

def detect_bpm(wav, sample_rate=48000, window=3):
	bpm = 0
	nsamps = len(wav)
	window_samps = int(window * sample_rate)
	samps_ndx = 0  # First sample in window_ndx
	max_window_ndx = math.floor(nsamps / window_samps)
	bpms = np.zeros(max_window_ndx)

	# Iterate through all windows
	for window_ndx in range(0, max_window_ndx):
		# Get a new set of samples
		# print(n,":",len(bpms),":",max_window_ndx_int,":",fs,":",nsamps,":",samps_ndx)
		data = wav[samps_ndx : samps_ndx + window_samps]
		if not ((len(data) % window_samps) == 0):
			raise AssertionError(str(len(data)))
		bpm, _correl_temp = bpm_detector(data, sample_rate)
		if bpm is None:
			continue
		bpms[window_ndx] = bpm
		# correl = correl_temp
		# Iterate at the end of the loop
		samps_ndx = samps_ndx + window_samps
	return np.median(bpms)

@dataclass(slots=True)
class PCMNote:
	pitch: int
	velocity: float
	tick: int

def load_wav(file):
	print("Importing WAV...")
	data = io.BytesIO()
	with wave.open(file, "rb") as w:
		assert (sr := w.getframerate()) == 48000
		assert (_nc := w.getnchannels()) == 2
		while True:
			b = w.readframes(w.getframerate())
			if not b:
				break
			data.write(b)
	buffer = data.getbuffer()
	a = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
	merged = a[::2] + a[1::2]
	merged *= 1 / 32767
	start = np.nonzero(merged)[0][0]
	merged = merged[start:]
	np.clip(merged, -1, None, out=merged)
	bpm = detect_bpm(merged, sr)
	bufsize = round(sr / (bpm / 60) / 24)
	c = librosa.hybrid_cqt(merged, sr=sr, fmin=librosa.note_to_hz("C1"), n_bins=bpo * octaves, bins_per_octave=bpo, hop_length=bufsize)
	amp = np.abs(c.T, dtype=np.float64)
	print(amp.shape, c.shape, sr)
	events = [
		[0, 0, "header", 1, 1 + 1, 1],
		[1, 0, "tempo", bufsize / sr * 1000 * 1000 * 1],
	]
	events.append([2, 0, "program_c", 0, 46])
	events.append([2, 0, "program_c", 1, 1])
	events.append([2, 0, "program_c", 2, 2])
	events.append([2, 0, "program_c", 3, 80])
	events.append([2, 0, "program_c", 4, 81])
	amp *= np.tile(np.arange(1, 1 + len(amp[0])), (len(amp), 1))
	loudest = np.max(amp) / 4
	active_notes = {}
	overtone_cost = 0.25
	overtone_value = 2
	for tick, bins in enumerate(amp):
		chord = bins[::2]
		note_instruments = [0] * len(chord)
		for i, v in enumerate(chord):
			if v <= loudest / 4096:
				continue
			overtone_matches = dict.fromkeys(harmonics, 0)
			for k, values in harmonics.items():
				for j, harmonic in values:
					if i + j < len(chord):
						overtone_matches[k] += abs(chord[i + j] - v * harmonic) / len(values)
			k, overtone_match = sorted(overtone_matches.items(), key=lambda t: t[1])[-1]
			if overtone_match <= overtone_cost:
				for j, harmonic in harmonics[k]:
					if i + j < len(chord):
						v2 = chord[i + j] - v * harmonic
						chord[i + j] = 0 if v2 <= chord[i + j] * overtone_cost else v2
				chord[i] = v * overtone_value
				match k:
					case "default":
						ins = 1
					case "triangle":
						ins = 2
					case "square":
						ins = 3
					case "saw":
						ins = 4
					case _:
						ins = 0
				note_instruments[i] = ins
		# betweens = bins[1::2] * 0.5
		# chord -= betweens
		# chord[1:] -= betweens[:-1]
		high = np.max(chord)
		if high <= loudest / 4096:
			continue
		norm = np.clip(chord * (1 / loudest), 0, 1)
		inds = list(np.argsort(norm))
		for attempt in range(24):
			i = inds.pop(-1)
			v = norm[i]
			if v < 1 / 4096:
				break
			if active_notes.get(i, 0) >= v * 1.0625:
				priority = -1
			elif active_notes.get(i, 0) <= v * 0.25:
				priority = 2
			else:
				priority = 1
			active_notes[i] = v
			events.append([2, tick, "note_on_c", note_instruments[i], i + c1, v * 127, priority, 1])
	return events