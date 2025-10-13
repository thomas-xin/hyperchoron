from collections import deque
import csv
from dataclasses import dataclass
import fractions
from math import ceil, log2, hypot
from types import SimpleNamespace
import numpy as np
try:
	import tqdm
except ImportError:
	import contextlib
	tqdm = None
else:
	import warnings
	warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
from .mappings import (
	instrument_mapping, midi_instrument_selection,
	material_map, sustain_map, instrument_codelist,
	fs1, c_1,
)
from .util import (
	round_min, log2lin, lin2log, sync_tempo, transport_note_priority, as_int,
	event_types, DEFAULT_NAME, DEFAULT_DESCRIPTION, NoteSegment, Transport
)


# def midi2csv(file):
# 	import py_midicsv
# 	csv_list = csv.reader(py_midicsv.midi_to_csv(file))
# 	return csv_list
def csv2midi(file, output):
	import py_midicsv
	midi = py_midicsv.csv_to_midi(file)
	with open(output, "wb") as f:
		writer = py_midicsv.FileWriter(f)
		writer.write(midi)
	return output

def merge_uniques(vals1, counts1, vals2, counts2):
    vals = np.concatenate([vals1, vals2])
    counts = np.concatenate([counts1, counts2])
    vals_merged, inv = np.unique(vals, return_inverse=True)
    counts_merged = np.zeros_like(vals_merged, dtype=counts.dtype)
    np.add.at(counts_merged, inv, counts)
    return vals_merged, counts_merged

def clamp_uniques(tup, lim=1024):
	return tup[0][:lim], tup[1][:lim]

def stack_midis(inputs):
	gen = iter(inputs)
	first = next(gen)
	outs = [first]
	assert first[0][2] == event_types.HEADER, f"Invalid MIDI Header: {first[0]}"
	cpc = int(first[0][5])
	for i, second in enumerate(gen, 1):
		assert second[0][2] == event_types.HEADER, f"Invalid MIDI Header: {second[0]}"
		cpc2 = second[0][5]
		if cpc != cpc2:
			ss = cpc / cpc2
			second = np.asanyarray(second, dtype=object)
			second[:, 1] = np.float32(second[:, 1]) * ss
		second = np.delete(second, second[:, 2] == event_types.TEMPO, axis=0)
		second[:, 3] += i * 256
		outs.append(second)
	if len(outs) == 1:
		return outs
	max_len = max((len(a[0]) for a in outs))
	for i, events in enumerate(outs):
		if len(events[0]) < max_len:
			events = np.pad(events, ((0, 0), (0, max_len - len(events[0]))), mode='constant')
			outs[i] = events
	return np.concatenate(outs, dtype=object)

def get_step_speed(midi_events, ctx=None):
	assert midi_events[0][2] == event_types.HEADER, f"Invalid MIDI Header: {midi_events[0]}"
	clocks_per_crotchet = int(midi_events[0][5])

	has_bends = np.any(midi_events[:, 2] == event_types.PITCH_BEND_C)
	note_ons, onc = clamp_uniques(np.unique(midi_events[midi_events[:, 2] == event_types.NOTE_ON_C][:, 1], return_counts=True), 1536)
	note_offs, offc = clamp_uniques(np.unique(midi_events[midi_events[:, 2] == event_types.NOTE_OFF_C][:, 1], return_counts=True), 1536)
	onc *= 2
	elements, counts = clamp_uniques(merge_uniques(note_ons, onc, note_offs, offc))
	timestamps = np.repeat(elements, counts)

	orig_tempo = 0
	milliseconds_per_clock = 0
	tempos = {}
	tempoc = midi_events[midi_events[:, 2] == event_types.TEMPO]
	tempot = list(tempoc[:, 1]) + [midi_events[-1][1]]
	deltas = np.diff(tempot)
	changes = tempoc[:, 3]
	for c, t in zip(changes, deltas):
		tempos[c] = tempos.get(c, 0) + t
	if tempos:
		tempo_list = list(tempos.items())
		tempo_list.sort(key=lambda t: t[1], reverse=True)
		orig_tempo = tempo_list[0][0]
		milliseconds_per_clock = orig_tempo / 1000 / clocks_per_crotchet
		if len(tempo_list) > 1:
			print("Multiple tempos detected! Auto-selecting most common from:", tempo_list[:5])
	if not clocks_per_crotchet:
		clocks_per_crotchet = 4 # Default to 4/4 time signature
		print("No time signature found! Defaulting to 4/4...")
	if not milliseconds_per_clock:
		orig_tempo = orig_tempo or 500 * 1000
		milliseconds_per_clock = orig_tempo / 1000 / clocks_per_crotchet # Default to 120 BPM
		print("No BPM found! Defaulting to 120...")
	return sync_tempo(timestamps, milliseconds_per_clock, clocks_per_crotchet, ctx.resolution / ctx.speed, orig_tempo, has_bends, ctx=ctx)

def preprocess(midi_events):
	"Preprocesses a MIDI file to determine the lengths of all notes based on their corresponding note_end_c events, as well as the maximum note velocity and timestamp"
	print(f"Preprocessing ({len(midi_events)} events total)...")
	modality = midi_events[0][7] if len(midi_events[0]) > 7 else 0
	note_lengths = {}
	temp_active = {}
	discard = []
	last_timestamp = midi_events[-1][1]
	for i, e in enumerate(midi_events):
		m = e[2]
		match m:
			case event_types.NOTE_ON_C:
				ti = e[0]
				t = e[1]
				c = int(e[3])
				p = int(e[4])
				h = (ti, c, p)
				velocity = int(e[5])
				if velocity < 1:
					# Treat notes with 0 velocity as note offs
					try:
						actives = temp_active[h]
					except KeyError:
						continue
					t2 = actives.pop(0)
					h2 = (t2, ti, c, p)
					note_lengths[h2] = max(1, t - t2)
					if not actives:
						temp_active.pop(h)
					discard.append(i)
				else:
					temp_active.setdefault(h, []).append(t)
					if len(e) > 7:
						break
			case event_types.NOTE_OFF_C:
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
				discard.append(i)
	for k, v in temp_active.items():
		for t in v:
			h2 = (t, *k)
			note_lengths.setdefault(h2, 1)
	midi_events = np.delete(midi_events, discard, axis=0)
	try:
		volume = np.max(midi_events[:, 5][(midi_events[:, 2] == event_types.CONTROL_C) & (midi_events[:, 4] == 7)]) / 127
	except ValueError:
		volume = 1
	try:
		max_vol = np.max(midi_events[:, 5][midi_events[:, 2] == event_types.NOTE_ON_C]) * volume
	except ValueError:
		max_vol = 1
	return modality, midi_events, note_lengths, max_vol, last_timestamp

@dataclass(slots=True)
class TransportNote:
	modality: int
	instrument_id: int
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
		while len(self.c) <= i:
			self.c.append(self.StatTypes(1, 0, 0, 2))
		return self.c[i]

def deconstruct(midi_events, speed_info, ctx=None):
	max_pitch = 101
	allow_stacks = ctx.strict_tempo or not ctx.apply_volumes
	active_notes = {i: [] for i in range(len(material_map))}
	active_notes[-1] = []
	active_nc = 0
	instrument_activities = {}
	latest_timestamp = timestamp_approx = timestamp = 0
	loud = 0
	note_candidates = 0
	_orig_ms_per_clock, real_ms_per_clock, scale, orig_step_ms, orig_tempo = speed_info
	step_ms = orig_step_ms
	modality, midi_events, note_lengths, max_vol, last_timestamp = preprocess(midi_events)
	speed_ratio = real_ms_per_clock / scale / _orig_ms_per_clock
	wait = round(orig_step_ms / speed_ratio, 9)
	print("Tick delay:", wait)
	played_notes = Transport(tick_delay=fractions.Fraction(wait) / 1000)
	instrument_ids = {}
	instrument_map = {}
	channel_stats = ChannelStats()
	print("Processing quantised notes...")
	bar_format = "{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
	timescale = real_ms_per_clock / scale
	progress = tqdm.tqdm(total=ceil(last_timestamp * timescale / 1000), bar_format=bar_format) if tqdm else contextlib.nullcontext()
	global_index = 0
	curr_frac = round(step_ms)
	tempo = orig_tempo
	leaked_notes = 0
	allowed_leakage = 8
	low_precision = len(midi_events) > 262144
	early_triggers = (event_types.PITCH_BEND_C, event_types.CONTROL_C)
	min_duration = 75

	def parse_note(event):
		channel = event[3]
		c = 0
		if channel & 255 == 9:
			if not ctx.drums:
				return
			c = -1
		if channel not in instrument_map:
			instrument_ids[channel] = c
			instrument_map[channel] = c
		instrument = instrument_map[channel]
		pitch = event[4]
		panning = 0
		priority = 2
		offset = 0
		instrument_id = None
		sustain = sustain_map[instrument]
		if instrument not in (-1, 6):
			sustain = sustain or 2
		_modality = modality
		length = 0
		if len(event) > 6:
			if len(event) > 7:
				if len(event) > 8:
					if len(event) > 9:
						if len(event) > 10:
							if len(event) > 11:
								instrument_id = event[11]
							_modality = event[10] or modality
						offset = event[9]
					panning = event[8]
				length_ticks = event[7]
				length = (length_ticks + 0.25) * timescale
			priority = event[6]
		if length == 0:
			length = 0
			if sustain or channel_stats[channel]:
				track = event[0]
				h = (event_timestamp, track, channel, pitch)
				try:
					length = (note_lengths[h] + 0.25) * timescale
				except KeyError:
					raise
					length = 0
			min_sustain = curr_frac * 2.25 if sustain == 2 else curr_frac * 1.25
			if length < min_sustain:
				length = min_sustain
				if sustain == 1:
					sustain = 0
		return TransportNote(
			modality=_modality,
			instrument_id=instrument_id if instrument_id is not None else instrument_ids[channel],
			pitch=pitch,
			velocity=event[5],
			start=timestamp_approx,
			length=length,
			channel=channel,
			sustain=sustain,
			priority=priority,
			volume=0,
			panning=panning,
			period=1,
			offset=offset,
		)

	def tick_notes(active_nc):
		ticked = {}
		long_notes = 0
		max_volume = 0
		total_value = 0
		for notes in active_notes.values():
			for note in notes:
				note.volume = channel_stats[note.channel].volume * note.velocity / max_vol * 127
				if note.volume > max_volume:
					max_volume = note.volume
				total_value += note.volume * min(4, note.length)
		for instrument, notes in active_notes.items():
			notes.reverse()
			for i in range(len(notes) - 1, -1, -1):
				note = notes[i]
				volume = note.volume
				if note.length < min_duration:
					note.length = min_duration
					note.sustain = note.sustain and 2
				length = note.length
				panning = max(-1, min(1, note.panning + channel_stats[note.channel].pan))
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
					if not allow_stacks:
						if ticked and len(next(iter(ticked))) == 2:
							bucket = (instrument, pitch)
						elif len(ticked) >= 36:
							ticked2 = {}
							for k, v in ticked.items():
								k2 = k[:2]
								try:
									temp = ticked2[k2]
								except KeyError:
									temp = ticked2[k2] = v
								else:
									temp[0] = max(temp[0], v[0])
									temp[2] = temp[2] if temp[1] > v[1] else v[2]
									temp[1] = temp[1] + v[1]
							ticked = ticked2
							bucket = (instrument, pitch)
						else:
							bucket = (instrument, pitch, len(ticked))
					else:
						bucket = (instrument, pitch)
					try:
						temp = ticked[bucket]
					except KeyError:
						temp = ticked[bucket] = [priority, log2lin(volume / 127), panning, note.modality, note.instrument_id, note.offset or len(ticked)]
					else:
						v = log2lin(volume / 127)
						temp[0] = max(temp[0], priority)
						temp[2] = temp[2] if temp[1] > v else panning
						temp[1] = temp[1] + v if abs(timestamp_approx - time) < 1 / 24000 else hypot(temp[1], v)
					long_notes += long
				if timestamp_approx >= end or len(notes) >= 64 and not needs_sustain:
					note.priority = -1
					notes.pop(i)
					active_nc -= 1
				else:
					note.priority = 0 if note.sustain == 1 else -1
					if note.sustain == 2 and (length < sms * 3 or (timestamp_approx - note.start) % (sms * 2) >= sms):
						# Notes that do not have a sustain property should fade out based on their duration
						note.velocity = max(2, note.velocity - sms / (length + sms * 2) * note.velocity * 2)
			notes.reverse()
		return ticked, active_nc

	def to_segments(ticked):
		beat = []
		poly = {}
		for k, v in ticked.items():
			instrument, pitch, *_ = k
			priority, volume, pan, modality, i_id, timing = v
			volume *= ctx.volume
			count = max(1, int(volume))
			vel = lin2log(volume / count)
			block = NoteSegment(priority, modality, i_id, instrument, pitch, vel, pan, timing)
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
		return beat

	with progress as bar:
		while global_index < len(midi_events):
			event = midi_events[global_index]
			event_timestamp = event[1]
			curr_step = step_ms
			time = event_timestamp * timescale
			mode = event[2]
			if mode in early_triggers or tempo == orig_tempo:
				condition = latest_timestamp - time >= -curr_step / 2
			else:
				condition = latest_timestamp - time >= curr_step / 3 * (leaked_notes >= allowed_leakage) - curr_step / 3
			if condition:
				if event_timestamp > last_timestamp:
					break
				# Process all events at the current timestamp
				match mode:
					case event_types.PROGRAM_C:
						channel = event[3]
						if channel & 255 == 9:
							pass
						else:
							value = int(event[4])
							instrument_ids[channel] = value
							instrument_map[channel] = instrument_mapping[value]
					case event_types.NOTE_ON_C:
						velocity = event[5]
						if velocity == 0:
							pass
						elif velocity < loud * 0.0625 and active_nc >= 48:
							note_candidates += 1
						else:
							note_candidates += 1
							note = parse_note(event)
							if not note:
								continue
							instrument = instrument_map[note.channel]
							active_notes[instrument].append(note)
							active_nc += 1
							loud = max(loud, velocity)
						leaked_notes += 1
					case event_types.PITCH_BEND_C:
						channel = event[3]
						value = as_int(event[4])
						offset = round_min(round((value - 8192) / 8192 * channel_stats[channel].bend_range * 1000) / 1000)
						prev = channel_stats[channel].bend
						if offset != prev:
							note_candidates += 1
							channel_stats[channel].bend = offset
							if round(offset) != round(prev) and channel in instrument_map:
								instrument = instrument_map[channel]
								found = []
								for note in active_notes[instrument]:
									if note.channel == channel:
										new_length = float(timestamp_approx + curr_frac * 1.5 - note.start)
										if new_length <= note.length:
											found.append(note)
										if note.instrument_id != -1:
											note.priority = max(note.priority, 1)
											note.sustain = True
								if not found and active_notes[instrument]:
									for note in active_notes[instrument]:
										if note.channel == channel:
											note.length = max(note.length, float(timestamp_approx + curr_frac - note.start))
					case event_types.CONTROL_C if event[4] == 6:
						channel = event[3]
						bend_range = as_int(event[5])
						prev = channel_stats[channel].bend_range
						if bend_range != prev:
							note_candidates += 1
							channel_stats[channel].bend_range = bend_range
							if round(bend_range) != round(prev) and channel in instrument_map:
								instrument = instrument_map[channel]
								found = []
								for note in active_notes[instrument]:
									if note.channel == channel:
										new_length = float(timestamp_approx + curr_frac * 1.5 - note.start)
										if new_length <= note.length:
											found.append(note)
										if not note.sustain:
											note.priority = max(note.priority, 1)
											note.sustain = True
								if not found and active_notes[instrument]:
									for note in active_notes[instrument]:
										if note.channel == channel:
											note.length = max(note.length, float(timestamp_approx + curr_frac - note.start))
					case event_types.CONTROL_C if event[4] == 7:
						channel = event[3]
						value = as_int(event[5])
						volume = value / 127
						orig_volume = channel_stats[channel].volume
						if volume >= orig_volume * 1.1 and instrument_map.get(channel) != -1:
							if note_candidates:
								note_candidates += 1
							if channel in instrument_map:
								instrument = instrument_map[channel]
								for note in active_notes[instrument]:
									if note.channel == channel:
										note.priority = max(note.priority, 1)
						channel_stats[channel].volume = volume
					case event_types.CONTROL_C if int(event[4]) == 10:
						channel = event[3]
						value = as_int(event[5])
						pan = max(-1, (value - 64) / 63)
						channel_stats[channel].pan = pan
					case event_types.TEMPO:
						tempo = event[3]
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
						if type(new_step) is fractions.Fraction:
							temp = float(new_step)
							if new_step == temp:
								new_step = temp
						if type(new_step) is float:
							if new_step.is_integer():
								new_step = int(new_step)
						step_ms = new_step
						curr_frac = float(step_ms) if type(step_ms) is fractions.Fraction else step_ms
						if not note_candidates:
							timestamp_approx = timestamp = round(time)
				global_index += 1
			else:
				ticked, active_nc = tick_notes(active_nc)
				beat = to_segments(ticked)
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
				loud *= 0.5
				allowed_leakage = (allowed_leakage - leaked_notes) * 2
				leaked_notes = 0
				for e in midi_events[global_index:]:
					offs = latest_timestamp - e[1] * timescale
					if offs >= -curr_step / 3:
						if e[2] != event_types.NOTE_ON_C:
							continue
						if offs >= 0:
							allowed_leakage += 2
						else:
							allowed_leakage += 1
					else:
						break
				allowed_leakage = max(1, allowed_leakage + 1 >> 1)
				if bar:
					bar.update(curr_step / 1000)
	while played_notes and not played_notes[-1]:
		played_notes.pop(-1)
	while played_notes and not played_notes[0]:
		played_notes.pop(0)
	return played_notes, note_candidates, instrument_activities, speed_info


@dataclass(slots=True)
class MidiNote:
	instrument_id: int
	instrument_type: int
	tick: int
	pitch: int
	length: int
	volume: int
	max_volume: int
	panning: int
	events: list
	aligned: int

def build_midi(notes, instrument_activities, ctx):
	wait = max(1, round(notes.tick_delay * 1000000))
	resolution = 6
	activities = list(map(list, instrument_activities.items()))
	instruments = [SimpleNamespace(
		id=midi_instrument_selection[curr[0]],
		id_override=None,
		type=curr[0],
		notes=[],
		name=instrument_codelist[curr[0]],
		channel=curr[0] + 1 if curr[0] >= 9 else curr[0] if curr[0] >= 0 else 9,
		pitchbend_range=0,
		event_count=0,
		sustain=sustain_map[curr[0]],
	) for curr in activities]
	drums = None
	for i, ins in enumerate(instruments):
		ins.index = i
		if ins.type == -1:
			drums = ins
	active = []
	next_active = []
	next_notes = set()
	next_drums = []
	for i, beat in enumerate(notes):
		tick = i * resolution
		taken = []
		if len(next_active) >= 36:
			next_active.clear()
		active, next_active = [n for n in active if n.aligned >= tick - resolution * 2] + next_active, []
		last_notes, next_notes = next_notes, set()
		last_drums, next_drums = next_drums, []
		if beat:
			ordered = sorted(beat, key=lambda note: (transport_note_priority(note, sustained=(note.instrument_class, note.pitch - c_1) in last_notes), note.instrument_class == -1), reverse=True)
		else:
			ordered = list(beat)
		for note in ordered:
			volume = round(min(127, note.velocity * 127))
			panning = round(note.panning * 63 + 64)
			ins = midi_instrument_selection[note.instrument_class]
			if ins < 0:
				if not drums:
					continue
				new_pitch = note.pitch
				if last_drums:
					for note in last_drums:
						note.length = min(resolution * 4, max(resolution, tick - note.tick))
						note.aligned = note.tick + note.length
				note = MidiNote(
					instrument_id=drums.index,
					instrument_type=-1,
					tick=tick,
					pitch=new_pitch,
					length=resolution * 4,
					volume=volume,
					max_volume=volume,
					panning=panning,
					events=[],
					aligned=tick + resolution * 4,
				)
				drums.notes.append(note)
				next_drums.append(note)
				continue
			new_pitch = note.pitch - c_1
			if new_pitch < 0:
				new_pitch += 12
			if new_pitch < 0:
				new_pitch = 0
			if new_pitch > 127:
				new_pitch = 127
			itype = note.instrument_class
			priority = note.priority
			h = (itype, new_pitch)
			if priority < 2:
				closest = 2
				target = None
				for j, last_note in enumerate(active):
					if last_note.instrument_type != itype:
						continue
					instrument = instruments[last_note.instrument_id]
					h2 = (last_note.instrument_id, last_note.pitch)
					last_vol, last_pan = last_note.volume, last_note.panning
					if last_vol + 16 <= volume:
						continue
					if last_vol < volume and not sustain_map[instrument.type]:
						continue
					diff = h[1] - h2[1]
					if abs(diff) > closest:
						continue
					if diff != 0:
						if abs(last_pan - panning) > 1 / 8 and priority >= 1:
							continue
						if abs(diff) > 2:
							continue
						if abs(diff) > 1 and not sustain_map[instrument.type]:
							continue
						if last_vol + 8 <= volume:
							continue
						if abs(diff) > 1 and priority >= 1 and last_note.aligned - last_note.tick > resolution * 3 and not last_note.events:
							continue
					target = last_note
					closest = abs(diff)
				if target:
					instrument = instruments[target.instrument_id]
					h2 = (target.instrument_id, target.pitch)
					last_vol, last_pan = target.volume, target.panning
					if h[1] != h2[1]:
						diff = h[1] - h2[1]
						if not any(e[1] == "pitch" for e in target.events):
							target.events.append((target.tick, "pitch", target.pitch))
						target.pitch = new_pitch
						target.aligned = tick
						for j in range(resolution + 2):
							offset = j / (resolution + 1) * diff
							timing = tick - resolution // 2 + j - 1
							target.events.append((timing, "pitch", h2[1] + offset))
							instrument.event_count += 1
						instrument.pitchbend_range = max(instrument.pitchbend_range, ceil(max(abs(e[2] - target.pitch) for e in target.events if e[1] == "pitch") / 12) * 12)
					if last_vol >= volume and not sustain_map[instrument.type]:
						pass
					else:
						if last_vol != volume:
							target.events.append((tick, "volume", min(127, round(volume / last_vol * 100))))
							target.max_volume = max(target.events[-1][-1], target.max_volume)
							instrument.event_count += 1
						if last_pan != panning:
							target.events.append((tick, "panning", panning))
							instrument.event_count += 1
					target.aligned = tick + resolution
					target.length = target.aligned - target.tick
					taken.append(instrument.index)
					active.remove(target)
					next_active.append(target)
					next_notes.add(h)
					continue
			choices = sorted(instruments, key=lambda instrument: instrument.id == ins, reverse=True)
			instrument = choices[0]
			if not instrument.id_override and note.modality == 0 and instrument.id != note.instrument_id:
				instrument.id_override = note.instrument_id
			length = resolution if sustain_map[instrument.type] else resolution * 2
			midi_note = MidiNote(
				instrument_id=instrument.index,
				instrument_type=itype,
				tick=tick,
				pitch=new_pitch,
				length=length,
				volume=volume,
				max_volume=volume,
				panning=panning,
				events=[],
				aligned=tick + length,
			)
			if type(new_pitch) is not int and not new_pitch.is_integer() and not instrument.pitchbend_range:
				instrument.pitchbend_range = 1
			instrument.notes.append(midi_note)
			taken.append(instrument.index)
			next_active.append(midi_note)
			next_notes.add(h)
	for ins in instruments:
		if ins.id_override:
			ins.id = ins.id_override
	return instruments, wait, resolution


def load_csv(file):
	print("Importing CSV...")
	with open(file, "r", encoding="utf-8") as f:
		csv_list = f.read().splitlines()
	return csv.reader(csv_list)

def load_midi(file):
	print("Importing MIDI...")
	from . import fastmidi
	transfer = fastmidi.parse_midi_events(file)
	events = np.pad(transfer, ((0, 0), (1, 1)), mode='constant')
	events[:, 7] = events[:, 6]
	events[:, 6] = 2
	return events


def proceed_save_midi(output, out_name, is_csv, instruments, wait, resolution):
	instruments = instruments.copy()
	remaining = set(range(16))
	for ins in instruments:
		ins.ccount = log2(ins.event_count) if ins.event_count > 0 else -1
		remaining.remove(ins.channel)
		ins.channels = [ins.channel]
		ins.taken = [0]
		ins.poly = [0]
		ins.eff = [0]
	remaining.discard(9)
	for r in remaining:
		temp = sorted((ins for ins in instruments if ins.type >= 0), key=lambda ins: ins.ccount, reverse=True)
		if not temp:
			break
		ins = temp[0]
		ins.channels.append(r)
		ins.taken.append(0)
		ins.poly.append(0)
		ins.eff.append(0)
		ins.ccount -= 2
	instruments.sort(key=lambda ins: ins.id)
	pans = [64] * 16
	vols = [127] * 16
	bends = [0] * 16
	highest_track = len(instruments)
	rows = []
	nc = 0
	for idx, ins in enumerate(instruments, 1):
		notes = deque(ins.notes)
		nc += len(notes)
		start = 0
		instrument = []
		for c in ins.channels:
			i = c + 1
			instrument.extend([
				[i, start, "Title_t", ins.name],
				[i, start, "Program_c", c, ins.id if ins.id >= 0 else 0],
				[i, start, "Control_c", c, 10, 64],
				[i, start, "Control_c", c, 7, 127],
			])
			if ins.pitchbend_range:
				instrument.extend([
					[i, start, "Control_c", c, 101, 0],
					[i, start, "Control_c", c, 100, 0],
					[i, start, "Control_c", c, 6, ins.pitchbend_range],
					[i, start, "Control_c", c, 100, 127],
					[i, start, "Control_c", c, 101, 127],
				])
		highest_track = max(highest_track, max(ins.channels) + 1)
		end = start
		while notes:
			note = notes.popleft()
			pitch = round(note.pitch)
			fine = note.pitch - pitch
			nend = note.tick + note.length
			if note.events or fine != 0:
				n = 0 if len(ins.channels) <= 1 else min((t, i) for i, t in enumerate(ins.taken[1:], 1))[1]
				if nend > ins.eff[n]:
					ins.eff[n] = nend
				if nend > ins.taken[n]:
					ins.poly[n] = 1
					ins.taken[n] = nend
				else:
					ins.poly[n] += 1
				c = ins.channels[n]
				i = c + 1
				if note.panning != pans[c]:
					pans[c] = note.panning
					instrument.append([i, note.tick, "Control_c", c, 10, pans[c]])
				target_vol = round(note.volume / note.max_volume * 127) if note.volume != note.max_volume else 127
				if vols[c] != 127 or vols[c] != target_vol:
					vols[c] = target_vol
					instrument.append([i, note.tick, "Control_c", c, 7, vols[c]])
				bend = 8192 if fine == 0 or not ins.pitchbend_range else round(8191 * fine / ins.pitchbend_range) + 8192
				if bends[c] != -1 or bend != 8192:
					instrument.append([i, note.tick, "Pitch_bend_c", c, bend])
					if bend != 8192:
						bends[c] = note.tick
					elif bends[c] < note.tick:
						bends[c] = -1
			else:
				try:
					n = next(i for i, t in enumerate(ins.taken) if t <= note.tick and ins.eff[i] <= note.tick)
				except StopIteration:
					n = min((t, i) for i, t in enumerate(ins.poly))[1]
				if ins.eff[n] > note.tick:
					n = 0
				if nend > ins.taken[n]:
					ins.poly[n] = 1
					ins.taken[n] = nend
				else:
					ins.poly[n] += 1
				c = ins.channels[n]
				i = c + 1
				target_vol = 127
			instrument.extend((
				[i, note.tick, "Note_on_c", c, pitch, 127 if target_vol != 127 else round(note.max_volume)],
				[i, note.tick + note.length, "Note_off_c", c, pitch, 0],
			))
			for tick, mode, value in note.events:
				match mode:
					case "panning":
						mode_i = 10
						if pans[c] == value:
							continue
						pans[c] = value
					case "volume":
						mode_i = 7
						if vols[c] == value:
							continue
						vols[c] = value = round(value / note.max_volume * 127)
					case "pitch":
						bend = round(8191 * (value - pitch) / ins.pitchbend_range) + 8192
						instrument.append([i, tick, "Pitch_bend_c", c, bend])
						if bend != 8192 and tick > bends[c]:
							bends[c] = tick
						continue
					case _:
						raise NotImplementedError(mode)
				instrument.append([i, tick, "Control_c", c, mode_i, value])
			end = max(end, note.tick + note.length)
		rows.extend(instrument)
	max_event = max((e[1] for e in rows), default=1)
	for i in range(1, highest_track):
		rows.insert(0, [i + 1, 0, "Start_track"])
	for i in range(highest_track):
		rows.append([i + 1, max_event, "End_track"])
	rows.sort(key=lambda e: (e[0], e[1]))
	import io
	b = open(output, "w", newline="", encoding="utf-8") if is_csv else io.StringIO()
	with b:
		writer = csv.writer(b)
		writer.writerows([
			[0, 0, "Header", 1, highest_track, resolution * 24],
			[1, 0, "Start_track"],
			*([[1, 0, "Title_t", out_name]] if out_name else ()),
			[1, 0, "Copyright_t", DEFAULT_NAME],
			[1, 0, "Text_t", DEFAULT_DESCRIPTION],
			[1, 0, "Time_signature", 4, 4, 8, 8],
			[1, 0, "Tempo", wait * 24],
		])
		writer.writerows(rows)
		writer.writerows([[0, 0, "End_of_file"]])
		if not is_csv:
			b.seek(0)
			csv2midi(b, output)
	return nc

def save_midi(transport, output, instrument_activities, ctx, **void):
	is_csv = output.casefold().endswith(".csv")
	if is_csv:
		print("Exporting CSV...")
	else:
		print("Exporting MIDI...")
	out_name = output.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
	instruments, wait, resolution = list(build_midi(transport, instrument_activities, ctx=ctx))
	return proceed_save_midi(output, out_name, is_csv, instruments, wait, resolution)