from collections import deque, namedtuple
import csv
from dataclasses import dataclass
import fractions
from math import ceil, sqrt
import os
from types import SimpleNamespace
try:
	import tqdm
except ImportError:
	import contextlib
	tqdm = None
else:
	import warnings
	warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
from .mappings import (
	instrument_mapping, org_instrument_mapping, midi_instrument_selection,
	material_map, sustain_map, instrument_codelist,
	fs1, c_1,
)
from .util import round_min, sync_tempo, transport_note_priority, estimate_filesize, DEFAULT_NAME, DEFAULT_DESCRIPTION

if os.name == "nt" and os.path.exists("Midicsv.exe") and os.path.exists("Csvmidi.exe"):
	use_py_midicsv = False
else:
	use_py_midicsv = True


def midi2csv(file):
	if use_py_midicsv or estimate_filesize(file) <= 65536:
		import py_midicsv
		csv_list = py_midicsv.midi_to_csv(file)
	else:
		import subprocess
		if isinstance(file, str):
			csv_list = subprocess.check_output(["Midicsv.exe", file, "-"]).decode("utf-8", "replace").splitlines()
		else:
			p = subprocess.Popen(["Midicsv.exe", "-", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
			b = file.read()
			csv_list = p.communicate(b)[0].decode("utf-8", "replace").splitlines()
	return csv_list
def csv2midi(file, output):
	if use_py_midicsv or estimate_filesize(file) <= 65536:
		import py_midicsv
		midi = py_midicsv.csv_to_midi(file)
		with open("example_converted.mid", "wb") as f:
			writer = py_midicsv.FileWriter(f)
			writer.write(midi)
	else:
		import subprocess
		p = subprocess.Popen(["Csvmidi.exe", "-", output], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
		b = file.read().encode("utf-8")
		csv_list = p.communicate(b)[0].decode("utf-8", "replace").splitlines()
	return csv_list


def get_step_speed(midi_events, ctx=None):
	orig_tempo = 0
	clocks_per_crotchet = 0
	milliseconds_per_clock = 0
	timestamps = {}
	time_diffs = {}
	tempos = {}
	since_tempo = 0
	has_bends = False
	for event in midi_events:
		mode = event[2]
		timestamp = event[1]
		if mode == "note_on_c":
			if int(event[5]) == 1:
				continue
			targets = [timestamp]
			channel = int(event[3])
			for last_timestamp in time_diffs.get(channel, (0,)):
				target = timestamp - last_timestamp
				if target <= 0:
					continue
				targets.append(target)
			td = time_diffs.setdefault(channel, deque(maxlen=4))
			if not td or timestamp != td[-1]:
				td.append(timestamp)
		elif mode == "note_off_c":
			channel = int(event[3])
		else:
			targets = [timestamp]
		for target in targets:
			try:
				timestamps[target] += 1
			except KeyError:
				timestamps[target] = 1
		if mode == "header":
			# Header tempo directly specifies clock pulses per quarter note
			clocks_per_crotchet = int(event[5])
		elif mode == "tempo":
			# Tempo event specifies microseconds per quarter note
			new_tempo = int(event[3])
			if orig_tempo and orig_tempo != new_tempo:
				tempos[orig_tempo] = tempos.get(orig_tempo, 0) + timestamp - since_tempo
			since_tempo = timestamp
			orig_tempo = new_tempo
			milliseconds_per_clock = orig_tempo / 1000 / clocks_per_crotchet
		elif mode == "pitch_bend_c":
			has_bends = True
	if tempos:
		tempos[orig_tempo] = tempos.get(orig_tempo, 0) + int(midi_events[-1][1]) - since_tempo
		tempo_list = list(tempos.items())
		tempo_list.sort(key=lambda t: t[1], reverse=True)
		orig_tempo = tempo_list[0][0]
		milliseconds_per_clock = orig_tempo / 1000 / clocks_per_crotchet
		print("Multiple tempos detected! Auto-selecting most common from:", min(tempo_list), max(tempo_list))
	if not clocks_per_crotchet:
		clocks_per_crotchet = 4 # Default to 4/4 time signature
		print("No time signature found! Defaulting to 4/4...")
	if not milliseconds_per_clock:
		orig_tempo = orig_tempo or 500 * 1000
		milliseconds_per_clock = orig_tempo / 1000 / clocks_per_crotchet # Default to 120 BPM
		print("No BPM found! Defaulting to 120...")
	return sync_tempo(timestamps, milliseconds_per_clock, clocks_per_crotchet, ctx.resolution / ctx.speed, orig_tempo, has_bends, ctx=ctx)

def preprocess(midi_events, ctx):
	title = None
	copyright = None
	is_org = False
	for event in midi_events[:64]:
		mode = event[2]
		match mode:
			case "title_t":
				title = event[3].strip(" \t\r\n\"'")
				if copyright:
					break
			case "copyright_t":
				copyright = ",".join(event[3:]).strip(" \t\r\n\"'")
				if title:
					break
	try:
		display = repr(title)
		print("Title:", display)
	except UnicodeEncodeError:
		display = repr(title.encode("utf-8", "ignore"))
		print("Title:", display)
	if title and title.startswith("Organya Symphony No. 1") and copyright == "(C) AUTHOR xxxxx, 2014":
		print("Using Org mapping...")
		is_org = True
	channel_stats = ChannelStats()
	note_lengths = {}
	temp_active = {}
	discard = set()
	midi_events.sort(key=lambda x: (x[1], x[2] == "note_on_c")) # Sort events by timestamp, keep note events last
	max_vel = 0
	last_timestamp = 0
	for i, e in enumerate(midi_events):
		m = e[2]
		match m:
			case "note_on_c":
				ti = e[0]
				t = e[1]
				c = int(e[3])
				p = int(e[4])
				h = (ti, c, p)
				velocity = int(e[5])
				volume = velocity * channel_stats[c].volume
				max_vel = max(max_vel, volume)
				if velocity < 1:
					try:
						actives = temp_active[h]
					except KeyError:
						continue
					t2 = actives.pop(0)
					h2 = (t2, ti, c, p)
					note_lengths[h2] = max(1, t - t2)
					if not actives:
						temp_active.pop(h)
					discard.add(i)
				else:
					try:
						temp_active[h].append(t)
					except KeyError:
						temp_active[h] = [t]
				last_timestamp = max(last_timestamp, t)
			case "note_off_c":
				ti = e[0]
				t = e[1]
				c = int(e[3])
				p = int(e[4])
				h = (ti, c, p)
				try:
					actives = temp_active[h]
				except KeyError:
					continue
				t2 = actives.pop(0)
				h2 = (t2, ti, c, p)
				note_lengths[h2] = max(1, t - t2)
				if not actives:
					temp_active.pop(h)
				discard.add(i)
				last_timestamp = max(last_timestamp, t)
			case "control_c":
				control = int(e[4])
				if control == 7:
					c = int(e[3])
					value = int(e[5])
					volume = value / 127
					channel_stats[c].volume = volume
	for k, v in temp_active.items():
		for t in v:
			h2 = (t, *k)
			note_lengths.setdefault(h2, 1)
	print("Max volume:", max_vel)
	midi_events = [e for i, e in enumerate(midi_events) if i not in discard]
	return midi_events, note_lengths, max_vel, last_timestamp, is_org

@dataclass(slots=True)
class TransportNote:
	pitch: int
	velocity: float
	start: float
	length: float
	channel: int
	sustain: bool
	# Priority is the value used by downstream encoders to determine how the note should be handled. The highest value is 2, where a note should be created regardless of any other factors. A note with priority 1 may be joined with a previous note ending on the same tick, including through pitchbends. A note with priority 0 will always be joined with any notes currently active, and a note with less than 0 priority will not be rendered in discrete note segment outputs (e.g. minecraft or nbs).
	priority: int
	volume: float
	panning: float
	period: int
	offset: int

NoteSegment = namedtuple("NoteSegment", ("instrument", "pitch", "priority", "long", "velocity", "panning"))

class ChannelStats:
	__slots__ = ("c")
	@dataclass(slots=True)
	class StatTypes:
		volume: float
		bend: float
		pan: float
		bend_range: float

	def __init__(self):
		self.c = []

	def __getitem__(self, i):
		try:
			return self.c[i]
		except IndexError:
			pass
		while i >= len(self.c):
			self.c.append(self.StatTypes(1, 0, 0, 2))
		return self.c[i]

def deconstruct(midi_events, speed_info, ctx=None):
	played_notes = []
	max_pitch = 101 if not ctx.mc_legal else 84
	active_notes = {i: [] for i in range(len(material_map))}
	active_notes[-1] = []
	active_nc = 0
	instrument_activities = {}
	latest_timestamp = timestamp_approx = timestamp = 0
	loud = 0
	note_candidates = 0
	_orig_ms_per_clock, real_ms_per_clock, scale, orig_step_ms, orig_tempo = speed_info
	step_ms = orig_step_ms
	midi_events, note_lengths, max_vel, last_timestamp, is_org = preprocess(midi_events, ctx=ctx)
	instrument_map = {}
	channel_stats = ChannelStats()
	print("Processing quantised notes...")
	bar_format = "{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
	timescale = real_ms_per_clock / scale
	progress = tqdm.tqdm(total=ceil(last_timestamp * timescale / 1000), bar_format=bar_format) if tqdm else contextlib.nullcontext()
	global_index = 0
	curr_frac = round(step_ms)
	leaked_notes = 0
	allowed_leakage = 1
	low_precision = len(midi_events) > 262144
	with progress as bar:
		while global_index < len(midi_events):
			event = midi_events[global_index]
			event_timestamp = event[1]
			curr_step = step_ms
			time = event_timestamp * timescale
			if latest_timestamp - time >= (curr_step / 3 * (1 - leaked_notes / allowed_leakage)):
				if event_timestamp > last_timestamp:
					break
				# Process all events at the current timestamp
				mode = event[2]
				match mode:
					case "program_c":
						channel = int(event[3])
						if channel == 9 and ctx.drums:
							pass
						else:
							value = int(event[4])
							instrument_map[channel] = org_instrument_mapping[value] if is_org else instrument_mapping[value]
						# print(instrument_map)
					case "note_on_c":
						channel = int(event[3])
						if channel not in instrument_map:
							instrument_map[channel] = -1 if channel == 9 and ctx.drums else 0
						instrument = instrument_map[channel]
						pitch = round_min(float(event[4]))
						velocity = int(event[5])
						panning = 0
						if velocity == 0:
							pass
						elif velocity < loud * 0.0625 and active_nc >= 48:
							note_candidates += 1
						else:
							note_candidates += 1
							priority = 2
							if instrument in (-1,):
								sustain = 0
							else:
								sustain = sustain_map[instrument] or (1 if is_org else 2 if not ctx.mc_legal else 0)
							length = 0
							if len(event) > 6:
								if len(event) > 7:
									if len(event) > 8:
										panning = event[8]
									length_ticks = event[7]
									length = (length_ticks + 0.25) * timescale
								priority = event[6]
							if length == 0:
								length = 0
								if sustain or channel_stats[channel]:
									track = int(event[0])
									h = (event_timestamp, track, channel, pitch)
									try:
										length = (note_lengths[h] + 0.25) * timescale
									except KeyError:
										length = 0
								min_sustain = curr_frac * 2.25 if sustain == 2 else curr_frac * 1.25
								if length < min_sustain:
									length = min_sustain
									if sustain == 1:
										sustain = 0
							note = TransportNote(
								pitch=pitch,
								velocity=velocity,
								start=timestamp_approx,
								length=length,
								channel=channel,
								sustain=sustain,
								priority=priority,
								volume=0,
								panning=panning,
								period=1,
								offset=0,
							)
							active_notes[instrument].append(note)
							active_nc += 1
							loud = max(loud, velocity)
					case "pitch_bend_c":
						channel = int(event[3])
						value = int(event[4])
						offset = round_min(round((value - 8192) / 8192 * channel_stats[channel].bend_range * 6) / 6)
						prev = channel_stats[channel].bend
						if offset != prev:
							note_candidates += 1
							channel_stats[channel].bend = offset
							if round(offset) != round(prev) and channel in instrument_map:
								instrument = instrument_map[channel]
								for note in active_notes[instrument]:
									if note.channel == channel:
										note.priority = max(note.priority, 1)
										note.length = max(note.length, float(timestamp_approx + curr_frac - note.start))
										note.sustain = True
					case "control_c" if int(event[4]) == 6:
						channel = int(event[3])
						channel_stats[channel].bend_range = int(event[5])
					case "control_c" if int(event[4]) == 7:
						channel = int(event[3])
						value = int(event[5])
						volume = value / 127
						orig_volume = channel_stats[channel].volume
						if volume >= orig_volume * 1.1:
							if note_candidates:
								note_candidates += 1
							if channel in instrument_map:
								instrument = instrument_map[channel]
								for note in active_notes[instrument]:
									if note.channel == channel:
										note.priority = max(note.priority, 1)
						channel_stats[channel].volume = volume
					case "control_c" if int(event[4]) == 10:
						channel = int(event[3])
						value = int(event[5])
						pan = max(-1, (value - 64) / 63)
						channel_stats[channel].pan = pan
					case "tempo":
						tempo = int(event[3])
						ratio = tempo / orig_tempo
						if max(ratio, 1 / ratio) - 1 < 1 / 16:
							ratio = 1
						elif ratio > 1:
							r2 = fractions.Fraction(ratio).limit_denominator(8)
							ratio = r2
						elif ratio < 1:
							r2 = fractions.Fraction(1 / ratio).limit_denominator(8)
							ratio = 1 / r2
						new_step = orig_step_ms / ratio
						if isinstance(new_step, fractions.Fraction):
							temp = float(new_step)
							if new_step == temp:
								new_step = temp
						if isinstance(new_step, float):
							if new_step.is_integer():
								new_step = int(new_step)
						step_ms = new_step
						curr_frac = float(step_ms) if isinstance(step_ms, fractions.Fraction) else step_ms
						if not note_candidates:
							timestamp_approx = timestamp = round(time)
				global_index += 1
			else:
				ticked = {}
				long_notes = 0
				max_volume = 0
				total_value = 0
				poly = 0
				for notes in active_notes.values():
					for note in notes:
						note.volume = channel_stats[note.channel].volume * note.velocity / max_vel * 127 * ctx.volume
						if note.volume > max_volume:
							max_volume = note.volume
						total_value += note.volume * min(4, note.length)
						poly += note.sustain
				for instrument, notes in active_notes.items():
					notes.reverse()
					for i in range(len(notes) - 1, -1, -1):
						note = notes[i]
						volume = note.volume
						length = note.length
						panning = note.panning + channel_stats[note.channel].pan
						end = note.start + note.length
						sms = curr_frac
						needs_sustain = note.sustain
						long = length >= sms * 2
						if note.start + sms * 3 / 4 > timestamp_approx or needs_sustain and timestamp_approx + sms <= end:
							pitch = channel_stats[note.channel].bend + note.pitch
							normalised = pitch + ctx.transpose - fs1
							if normalised > max_pitch:
								pitch = max_pitch - ctx.transpose + fs1
							elif normalised < -12:
								pitch = -12 - ctx.transpose + fs1
							priority = note.priority
							if priority > 0 and volume != 0:
								# For sections that are really loud or for sustained notes at a fast tempo; quantise note segments based on the square root of the ratio between the note's volume and total volume, multiplied by the ratio between note lengths
								period = note.period = round(min(8, max(1, 50 / sqrt(pitch + 12) / ctx.strum_affinity * sqrt(total_value / volume / min(4, length) + 8) / sms))) if long else 8
								offset = note.offset = long_notes % period if long else 0
							elif round((timestamp_approx - note.start) / sms) % note.period != note.offset:
								priority = -1
							bucket = (instrument, pitch)
							try:
								temp = ticked[bucket]
							except KeyError:
								temp = ticked[bucket] = [priority, long, volume ** 2, panning]
							else:
								temp[0] = max(temp[0], priority)
								temp[1] |= long
								temp[3] = temp[3] if temp[2] > volume ** 2 else panning
								temp[2] = temp[2] + volume ** 2
							long_notes += long
						if timestamp_approx >= end or len(notes) >= 64 and not needs_sustain:
							notes.pop(i)
							active_nc -= 1
						else:
							note.priority = 0 if note.sustain == 1 else -1
							if note.sustain == 2 and (length < sms * 3 or (timestamp_approx - note.start) % (sms * 2) >= sms):
								# Notes that do not have a sustain property should fade out based on their duration
								note.velocity = max(2, note.velocity - sms / (length + sms * 2) * note.velocity * 2)
					notes.reverse()
				beat = []
				poly = {}
				for k, v in ticked.items():
					instrument, pitch = k
					priority, long, volume, pan = v
					# After the per-note formula is applied, the result of multiple notes quantised into one will have a volume equal to the root mean square.
					volume = sqrt(volume)
					count = max(1, int(volume // 127))
					vel = max(1, min(127, round(volume / count)))
					block = NoteSegment(instrument, pitch, priority, long, vel, pan)
					for w in range(count):
						beat.append(block)
					if instrument != -1:
						try:
							poly[instrument] += count
						except KeyError:
							poly[instrument] = count
					else:
						poly.setdefault(instrument, 0)
					try:
						instrument_activities[instrument][0] += volume
						instrument_activities[instrument][1] = max(instrument_activities[instrument][1], poly[instrument])
					except KeyError:
						instrument_activities[instrument] = [volume, poly[instrument]]
				played_notes.append(beat)
				timestamp += curr_step
				if isinstance(timestamp, int) or timestamp.is_integer():
					timestamp_approx = timestamp = int(timestamp)
				else:
					timestamp_approx = float(timestamp)
				if low_precision:
					latest_timestamp = timestamp_approx
				else:
					latest_timestamp = timestamp
				if bar:
					bar.update(curr_step / 1000)
				loud *= 0.5
				leaked_notes = 0
				allowed_leakage = 0
				for i in range(global_index, len(midi_events)):
					e = midi_events[i]
					offs = latest_timestamp - e[1] * timescale
					if offs >= curr_step / 3:
						if e[2] != "note_on_c":
							continue
						if offs >= curr_step * 2 / 3:
							allowed_leakage += 2
						else:
							allowed_leakage += 1
					else:
						break
				allowed_leakage = max(1, allowed_leakage / 2)
	while played_notes and not played_notes[-1]:
		played_notes.pop(-1)
	return played_notes, note_candidates, is_org, instrument_activities, speed_info


@dataclass(slots=True)
class MidiNote:
	tick: int
	pitch: int
	length: int
	volume: int
	panning: int
	events: list
	aligned: int

def build_midi(notes, instrument_activities, speed_info, ctx):
	orig_ms_per_clock, real_ms_per_clock, scale, orig_step_ms, _orig_tempo = speed_info
	speed_ratio = real_ms_per_clock / scale / orig_ms_per_clock
	wait = round(orig_step_ms / speed_ratio * 1000)
	resolution = 6
	activities = list(map(list, instrument_activities.items()))
	instruments = [SimpleNamespace(
		id=midi_instrument_selection[curr[0]],
		type=curr[0],
		notes=[],
		name=instrument_codelist[curr[0]],
		channel=curr[0] if curr[0] >= 0 else 9,
		pitchbend_range=0,
		event_count=0,
		sustain=sustain_map[curr[0]],
	) for curr in activities for c in range(curr[1][1] if curr[0] != -1 else 1)]
	drums = None
	for i, ins in enumerate(instruments):
		ins.index = i
		if ins.type == -1:
			drums = ins
	active = {}
	for i, beat in enumerate(notes):
		tick = i * resolution
		taken = []
		next_active = {}
		if beat:
			ordered = sorted(beat, key=lambda note: (transport_note_priority(note, sustained=(note[0], note[1] - c_1) in active), note[0] == -1), reverse=True)
		else:
			ordered = list(beat)
		for note in ordered:
			itype, pitch, priority, _long, vel, pan = note
			pitch = round(pitch)
			volume = round(min(127, vel))
			panning = round(pan * 63 + 64)
			ins = midi_instrument_selection[itype]
			if ins < 0:
				new_pitch = pitch
				note = MidiNote(
					tick=tick,
					pitch=new_pitch,
					length=resolution,
					volume=volume,
					panning=panning,
					events=[],
					aligned=0,
				)
				if drums:
					drums.notes.append(note)
				continue
			new_pitch = pitch - c_1
			if new_pitch < 0:
				new_pitch += 12
			if new_pitch < 0:
				new_pitch = 0
			if new_pitch > 127:
				new_pitch = 127
			h = (itype, new_pitch)
			if priority < 2:
				try:
					for h2 in (h, (itype, new_pitch - 1), (itype, new_pitch + 1), (itype, new_pitch - 2), (itype, new_pitch + 2)):
						for iid in active.get(h2, ()):
							if iid in taken:
								continue
							instrument = instruments[iid]
							last_note = instrument.notes[-1]
							last_vol, last_pan = last_note.volume, last_note.panning
							if last_vol + 16 <= volume:
								continue
							if (last_vol < volume or last_note.length > resolution * 4) and not sustain_map[instrument.type]:
								continue
							if h != h2:
								if last_pan != panning or last_note.aligned > resolution * 3:
									continue
								diff = h[1] - h2[1]
								if last_vol + 8 <= volume:
									continue
								if abs(diff) > 1 and last_note.aligned > resolution:
									continue
								if last_note.aligned <= resolution * 3:
									if not any(e[1] == "pitch" for e in last_note.events):
										last_note.events.append((last_note.tick, "pitch", last_note.pitch))
									last_note.pitch = new_pitch
									last_note.aligned = 0
								for j in range(resolution + 2):
									offset = j / (resolution + 1) * diff
									timing = tick - resolution // 2 + j - 1
									last_note.events.append((timing, "pitch", h2[1] + offset))
									instrument.event_count += 1
								instrument.pitchbend_range = max(instrument.pitchbend_range, ceil(max(abs(e[2] - last_note.pitch) for e in last_note.events if e[1] == "pitch") / 12) * 12)
							if last_vol >= volume and not sustain_map[instrument.type]:
								pass
							else:
								if last_vol != volume:
									last_note.events.append((tick, "volume", min(127, round(volume / last_vol * 100))))
									instrument.event_count += 1
								if last_pan != panning:
									last_note.events.append((tick, "panning", panning))
									instrument.event_count += 1
							last_note.length += resolution
							last_note.aligned += resolution
							taken.append(iid)
							try:
								next_active[h].append(instrument.index)
							except KeyError:
								next_active[h] = [instrument.index]
							raise StopIteration
				except StopIteration:
					continue
			choices = sorted(instruments, key=lambda instrument: (instrument.index not in taken, instrument.id == ins), reverse=True)
			instrument = choices[0]
			events = []
			instrument.notes.append(MidiNote(
				tick=tick,
				pitch=new_pitch,
				length=resolution,
				volume=volume,
				panning=panning,
				events=events,
				aligned=resolution,
			))
			taken.append(instrument.index)
			try:
				next_active[h].append(instrument.index)
			except KeyError:
				next_active[h] = [instrument.index]
		active = next_active
	print([ins.pitchbend_range for ins in instruments])
	return instruments, wait, resolution


def load_csv(file):
	print("Importing CSV...")
	with open(file, "r", encoding="utf-8") as f:
		csv_list = f.read().splitlines()
	return list(csv.reader(csv_list))

def load_midi(file):
	print("Importing MIDI...")
	csv_list = midi2csv(file)
	if not isinstance(file, str):
		file.close()
	return list(csv.reader(csv_list))


def save_midi(transport, output, instrument_activities, speed_info, ctx):
	is_csv = output.casefold().endswith(".csv")
	if is_csv:
		print("Exporting CSV...")
	else:
		print("Exporting MIDI...")
	out_name = output.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
	instruments, wait, resolution = list(build_midi(transport, instrument_activities, speed_info, ctx=ctx))
	import io
	b = open(output, "w", newline="", encoding="utf-8") if is_csv else io.StringIO()
	nc = 0
	with b:
		instruments.sort(key=lambda ins: ins.event_count)
		writer = csv.writer(b)
		writer.writerows([
			[0, 0, "header", 1, len(instruments) + 1, resolution * 24],
			[1, 0, "start_track"],
			[1, 0, "title_t", out_name],
			[1, 0, "copyright_t", DEFAULT_NAME],
			[1, 0, "text_t", DEFAULT_DESCRIPTION],
			[1, 0, "time_signature", 4, 4, 8, 8],
			[1, 0, "tempo", wait * 24],
			[1, 0, "end_track"],
		])
		covered_channels = {}
		extra_channels = 0
		for i, ins in enumerate(instruments, 2):
			notes = deque(ins.notes)
			nc += len(notes)
			start = 0
			if ins.pitchbend_range and ins.channel in covered_channels:
				if extra_channels > 5:
					for k, v in covered_channels.items():
						if k != ins.channel and v == ins.id:
							ins.channel = k
							break
				else:
					ins.channel = extra_channels + 10
					extra_channels += 1
			writer.writerows([
				[i, start, "start_track"],
				[i, start, "title_t", ins.name],
				[i, start, "program_c", ins.channel, ins.id if ins.id >= 0 else 0],
				[i, start, "control_c", ins.channel, 10, 64],
				[i, start, "control_c", ins.channel, 7, 100],
				[i, start, "control_c", ins.channel, 101, 0],
				[i, start, "control_c", ins.channel, 100, 0],
				[i, start, "control_c", ins.channel, 6, ins.pitchbend_range or 12],
				[i, start, "control_c", ins.channel, 100, 127],
				[i, start, "control_c", ins.channel, 101, 127],
			])
			covered_channels[ins.channel] = ins.id
			pan = 64
			vol = 100
			bent = False
			instrument = []
			end = start
			while notes:
				note = notes.popleft()
				if note.panning != pan:
					pan = note.panning
					instrument.append([i, note.tick, "control_c", ins.channel, 10, pan])
				if vol != 100:
					vol = 100
					instrument.append([i, note.tick, "control_c", ins.channel, 7, vol])
				if bent:
					instrument.append([i, note.tick, "pitch_bend_c", ins.channel, 8192])
					bent = False
				instrument.extend((
					[i, note.tick, "note_on_c", ins.channel, note.pitch, note.volume],
					[i, note.tick + note.length, "note_off_c", ins.channel, note.pitch, 0],
				))
				for tick, mode, value in note.events:
					match mode:
						case "panning":
							mode_i = 10
							if pan == value:
								continue
							pan = value
						case "volume":
							mode_i = 7
							if vol == value:
								continue
							vol = value
						case "pitch":
							bend = round(8191 * (value - note.pitch) / ins.pitchbend_range) + 8192
							instrument.append([i, tick, "pitch_bend_c", ins.channel, bend])
							bent = bend != 8192
							continue
						case _:
							raise NotImplementedError(mode)
					instrument.append([i, tick, "control_c", ins.channel, mode_i, value])
				end = max(end, note.tick + note.length)
			instrument.sort(key=lambda t: (t[1], t[2] == "note_on_c"))
			instrument.append([i, end, "end_track"])
			writer.writerows(instrument)
		writer.writerows([[0, 0, "end_of_file"]])
		if not is_csv:
			b.seek(0)
			csv2midi(b, output)
	return nc