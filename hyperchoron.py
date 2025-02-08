from collections import deque
import csv
import functools
import itertools
from math import ceil, inf, isqrt, sqrt, gcd
from types import SimpleNamespace
import py_midicsv
import tqdm


TRANSPOSE = 0
SPEED_MULTIPLIER = 1
STRUM_AFFINITY = 1
NO_DRUMS = 0

# Predefined list attempting to match instruments across pitch ranges
material_map = [
	["bamboo_planks", "black_wool", "black_wool+", "snow_block+", "gold_block", "gold_block+"], # Plucked
	["bamboo_planks", "bamboo_planks+", "glowstone", "glowstone+", "gold_block", "gold_block+"], # Keyboards
	["pumpkin", "pumpkin+", "snow_block", "clay", "clay+", "packed_ice+"], # Wind
	["pumpkin", "pumpkin+", "emerald_block", "emerald_block+", "gold_block", "gold_block+"], # Synth
	["bamboo_planks", "bamboo_planks+", "iron_block", "iron_block+", "gold_block", "gold_block+"], # Pitched Percussion
	["bamboo_planks", "black_wool", "snow_block", "snow_block+", "packed_ice", "packed_ice+"], # Bell
	["cobblestone", "cobblestone+", "red_stained_glass", "red_stained_glass+", "heavy_core", "heavy_core+"], # Unpitched Percussion
	["bamboo_planks", "black_wool", "hay_block", "hay_block+", "bone_block", "bone_block+"], # String
	None # Drumset
]
instrument_names = dict(
	snow_block="harp",
	bamboo_planks="bass",
	heavy_core="snare",
	black_concrete_powder="snare",
	blue_stained_glass="hat",
	red_stained_glass="hat",
	obsidian="basedrum",
	netherrack="basedrum",
	cobblestone="basedrum",
	gold_block="bell",
	clay="flute",
	packed_ice="chime",
	black_wool="guitar",
	bone_block="xylophone",
	iron_block="iron_xylophone",
	soul_sand="cow_bell",
	pumpkin="didgeridoo",
	emerald_block="bit",
	hay_block="banjo",
	glowstone="pling",
	skeleton_skull="skeleton",
	wither_skeleton_skull="wither_skeleton",
	zombie_head="zombie",
	creeper_head="creeper",
	piglin_head="piglin",
)
sustain_map = [
	0,
	0,
	2,
	1,
	0,
	0,
	0,
	1,
	0,
]
instrument_mapping = [
	1, 1, 1, 4, 3, 3, 0, 1, # Piano
	5, 5, 5, 4, 4, 5, 5, 1, # CP
	2, 1, 2, 2, 7, 7, 7, 3, # Organ
	0, 0, 0, 1, 1, 3, 3, 7, # Guitar
	0, 0, 0, 0, 4, 4, 3, 3, # Bass
	7, 7, 7, 7, 3, 0, 0, 6, # Strings
	2, 2, 2, 2, 2, 2, 2, 3, # Ensemble
	2, 2, 2, 2, 2, 2, 2, 2, # Brass
	7, 7, 7, 7, 7, 2, 2, 3, # Reed
	2, 2, 2, 2, 2, 2, 2, 2, # Pipe
	3, 3, 3, 3, 3, 3, 3, 3, # SL
	2, 2, 2, 2, 2, 2, 2, 2, # SP
	2, 2, 2, 2, 2, 2, 2, 2, # SE
	0, 7, 2, 0, 1, 3, 3, 7, # Ethnic
	0, 6, 6, 6, 6, 6, 6, 6, # Percussive
	6, 6, 6, 6, 6, 6, 6, 6, # Percussive
]
org_instrument_mapping = [
	0, 0, 1, 3, 2, 2, 2, 3, 3, 7,
	0, 1, 3, 3, 2, 2, 3, 7, 7, 7,
	3, 3, 3, 3, 3, 3, 3, 7, 7, 7,
	1, 1, 1, 7, 2, 2, 2, 3, 7, 7,
	7, 3, 3, 3, 3, 3, 7, 2, 3, 2,
	3, 7, 7, 7, 7, 7, 7, 2, 2, 3,
	3, 3, 3, 3, 3, 7, 7, 7, 2, 7,
	0, 1, 2, 3, 2, 3, 7, 5, 4, 5,
	7, 3, 2, 1, 2, 7, 7, 7, 7, 7,
	4, 2, 7, 7, 7, 7, 7, 3, 3, 3,
]
percussion_mats = {int((data := line.split("#", 1)[0].strip().split("\t"))[0]): (data[1], int(data[2])) for line in """
0	PLACEHOLDER	0
31	heavy_core	16	# Sticks
32	blue_stained_glass	24	# Square Click
33	blue_stained_glass	8	# Metronome Click
34	gold_block	18	# Metronome Bell
35	netherrack	4	# Acoustic Bass Drum
36	netherrack	8	# Bass Drum 1
37	blue_stained_glass	4	# Side Stick
38	heavy_core	4	# Acoustic Snare
39	heavy_core	12	# Hand Clap
40	heavy_core	8	# Electric Snare
41	obsidian	0	# Low Floor Tom
42	blue_stained_glass	18	# Closed Hi-Hat
43	obsidian	6	# High Floor Tom
44	blue_stained_glass	21	# Pedal Hi-Hat
45	obsidian	12	# Low Tom
46	heavy_core	18	# Open Hi-Hat
47	obsidian	16	# Low-Mid Tom
48	obsidian	20	# Hi-Mid Tom
49	creeper_head	0	# Crash Cymbal 1
50	obsidian	24	# High Tom
51	heavy_core	23	# Ride Cymbal 1
52	heavy_core	20	# Chinese Cymbal
53	heavy_core	19	# Ride Bell
54	blue_stained_glass	19	# Tambourine
55	heavy_core	17	# Splash Cymbal
56	soul_sand	15	# Cowbell
57	creeper_head	0	# Crash Cymbal 2
58	skeleton_skull	0	# Vibraslap
59	heavy_core	24	# Ride Cymbal 2
60	netherrack	23	# Hi Bongo
61	netherrack	21	# Low Bongo
62	netherrack	10	# Mute Hi Conga
63	soul_sand	7	# Open Hi Conga
64	soul_sand	2	# Low Conga
65	obsidian	22	# High Timbale
66	obsidian	6	# Low Timbale
67	bone_block	11	# High Agogo
68	soul_sand	18	# Low Agogo
69	blue_stained_glass	20	# Cabasa
70	blue_stained_glass	22	# Maracas
71	gold_block	20	# Short Whistle
72	packed_ice	17	# Long Whistle
73	blue_stained_glass	12	# Short Guiro
74	skeleton_skull	0	# Long Guiro
75	bone_block	19	# Claves
76	bone_block	14	# Hi Wood Block
77	bone_block	7	# Low Wood Block
78	piglin_head	0	# Mute Cuica
79	zombie_head	0	# Open Cuica
80	bone_block	24	# Mute Triangle
81	packed_ice	24	# Open Triangle
82	blue_stained_glass	16	# Closed Hi-Hat
83	skeleton_skull	0	# Jingle Bell
84	packed_ice	20	# Bell Tree
""".strip().splitlines()}

# Thank you deepseek-r1 for optimisation
def approximate_gcd(arr, min_value=8):
	if not arr:
		return 0, 0

	# Check if any element is >= min_value
	has_element_above_min = any(x >= min_value for x in arr)
	if not has_element_above_min:
		return gcd(arr), len(arr)

	# Collect non-zero elements
	non_zero = [x for x in arr if x != 0]
	if not non_zero:
		return 0, 0  # All elements are zero

	# Generate all possible divisors >= min_value from non-zero elements
	divisors = set()
	for x in non_zero:
		x_abs = abs(x)
		# Find all divisors of x_abs
		for i in range(1, int(isqrt(x_abs)) + 1):
			if x_abs % i == 0:
				if i >= min_value:
					divisors.add(i)
				counterpart = x_abs // i
				if counterpart >= min_value:
					divisors.add(counterpart)

	# If there are no divisors >= min_value, return the GCD of all elements
	if not divisors:
		return gcd(arr), len(arr)

	# Sort divisors in descending order
	sorted_divisors = sorted(divisors, reverse=True)

	max_count = 0
	candidates = []

	# Find the divisor(s) with the maximum count of divisible elements
	for d in sorted_divisors:
		count = 0
		for x in arr:
			if x % d == 0:
				count += 1
		if count > max_count:
			max_count = count
			candidates = [d]
		elif count == max_count:
			candidates.append(d)

	# Now find the maximum GCD among the candidates
	max_gcd = 0
	for d in candidates:
		elements = [x for x in arr if x % d == 0]
		current_gcd = gcd(*elements)
		if current_gcd > max_gcd:
			max_gcd = current_gcd

	return (max_gcd, len(arr) - max_count) if max_gcd >= min_value else (gcd(arr), len(arr))

def binary_map():
	yield 0
	denominator = 2
	while True:
		for numerator in range(1, denominator, 2):
			yield numerator / denominator
		denominator *= 2
it = binary_map()
binary_fracs = [next(it) for i in range(64)]

@functools.lru_cache(maxsize=256)
def get_note_mat(note, transpose=0):
	material = material_map[note[0]]
	pitch = note[1]
	if not material:
		try:
			return percussion_mats[pitch]
		except KeyError:
			print("WARNING: Note", pitch, "not yet supported for drums, discarding...")
			return "PLACEHOLDER", 0
	pitch += transpose
	c4 = 60
	fs4 = c4 + 6
	fs1 = fs4 - 36
	normalised = pitch - fs1
	if normalised < 0:
		normalised += 12
	if normalised < 0:
		normalised = 0
	elif normalised > 72:
		normalised = 72
	assert 0 <= normalised <= 72, normalised
	ins, mod = divmod(normalised, 12)
	if ins > 5:
		ins = 5
		mod += 12
	mat = material[ins]
	if note[3] == 2:
		replace = dict(
			hay_block="snow_block",
			emerald_block="snow_block",
			snow_block="snow_block",
			iron_block="snow_block",
			soul_sand="snow_block+",
			glowstone="snow_block",
			clay="snow_block+"
		)
		replace.update({
			"hay_block+": "snow_block+",
			"emerald_block+": "snow_block+",
			"snow_block+": "snow_block+",
			"black_wool+": "snow_block",
			"iron_block+": "snow_block+",
			"glowstone+": "snow_block+",
		})
		try:
			mat = replace[mat]
		except KeyError:
			return "PLACEHOLDER", 0
	elif note[3]:
		replace = dict(
			bamboo_planks="pumpkin",
			black_wool="pumpkin+",
			bone_block="gold_block",
			iron_block="snow_block",
			soul_sand="glowstone+",
			glowstone="snow_block",
		)
		replace.update({
			"bamboo_planks+": "pumpkin+",
			"black_wool+": "snow_block",
			"gold_block+": "packed_ice+",
			"bone_block+": "gold_block+",
			"iron_block+": "snow_block+",
			"glowstone+": "snow_block+",
		})
		try:
			mat = replace[mat]
		except KeyError:
			pass
	if mat.endswith("+"):
		mat = mat[:-1]
		mod += 12
	return mat, mod

def get_note_block(note, positioning=[0, 0, 0], replace=None, transpose=0):
	base, pitch = get_note_mat(note, transpose=transpose)
	x, y, z = positioning
	coords = [(x, y, z), (x, y + 1, z), (x, y + 2, z)]
	if replace and base in replace:
		base = replace[base]
	if base == "PLACEHOLDER":
		return (
			(coords[0], "mangrove_roots"),
		)
	if base.endswith("_head"):
		return (
			(coords[0], "mangrove_roots"),
			(coords[1], "note_block", dict(note=pitch, instrument=instrument_names[base])),
			(coords[2], base),
		)
	return (
		(coords[0], base),
		(coords[1], "note_block", dict(note=pitch, instrument=instrument_names[base])),
	)

def get_step_speed(midi_events, tps=20):
	orig_tempo = 0
	clocks_per_crotchet = 0
	milliseconds_per_clock = 0
	timestamps = {}
	started = False
	for event in midi_events:
		mode = event[2].strip().casefold()
		timestamp = int(event[1])
		if mode == "note_on_c":
			if int(event[5]) == 1:
				continue
			started = True
		try:
			timestamps[timestamp] += 1
		except KeyError:
			timestamps[timestamp] = 1
		if started:
			continue
		if mode == "header":
			# Header tempo directly specifies clock pulses per quarter note
			clocks_per_crotchet = int(event[5])
		elif mode == "tempo":
			# Tempo event specifies microseconds per quarter note
			orig_tempo = int(event[3])
			milliseconds_per_clock = orig_tempo / 1000 / clocks_per_crotchet
	if not clocks_per_crotchet:
		clocks_per_crotchet = 4 # Default to 4/4 time signature
	if not milliseconds_per_clock:
		orig_tempo = orig_tempo or 60 * 120
		milliseconds_per_clock = orig_tempo / 1000 / clocks_per_crotchet # Default to 120 BPM
	step_ms = 1000 // tps
	rev = deque(sorted(timestamps, key=lambda k: timestamps[k]))
	mode = timestamps[rev[-1]]
	while len(timestamps) > 4096 or timestamps[rev[0]] < mode / 64:
		timestamps.pop(rev.popleft())
	min_value = step_ms / milliseconds_per_clock
	print("Estimating true resolution...", len(timestamps), milliseconds_per_clock, step_ms, min_value)
	speed, exclusions = approximate_gcd(timestamps, min_value=min_value - 1)
	use_exact = False
	if speed > min_value * sqrt(2) or exclusions > len(timestamps) / 8:
		print("Rejecting first estimate:", speed, min_value, exclusions)
		if exclusions >= len(timestamps) * 3 / 4:
			print("Discarding tempo...", exclusions, len(timestamps))
			speed = 1
			use_exact = True
		else:
			print("Finding closest speed...", exclusions, len(timestamps))
			div = max(1, round(speed / min_value - 0.25) / 2)
			if div == 1:
				speed *= 0.75
			else:
				speed /= div
	sm = globals()["SPEED_MULTIPLIER"]
	if sm != 1:
		print("Speed manually scaled up by", sm)
		speed *= sm
		use_exact = True
	if use_exact:
		real_ms_per_clock = milliseconds_per_clock
	else:
		real_ms_per_clock = round(milliseconds_per_clock * min_value / step_ms) * step_ms
	print("Speed scale:", speed, real_ms_per_clock)
	return real_ms_per_clock, speed, step_ms, orig_tempo

def convert_midi(midi_events, tps=20):
	title = None
	is_org = False
	for event in midi_events[:20]:
		mode = event[2].strip().casefold()
		if mode == "title_t":
			title = event[3].strip(" \t\r\n\"'")
			break
	print("Title:", repr(title))
	if title and title.startswith("Organya Symphony No. 1"):
		print("Using Org mapping...")
		is_org = True
	played_notes = []
	pitchbend_ranges = {}
	instrument_map = {}
	last_timestamp = 0
	for event in midi_events:
		mode = event[2].strip().casefold()
		if mode == "program_c":
			channel = int(event[3])
			if channel == 9 and not globals()["NO_DRUMS"]:
				continue
			value = int(event[4])
			instrument_map[channel] = org_instrument_mapping[value] if is_org else instrument_mapping[value]
		elif mode == "note_on_c":
			channel = int(event[3])
			if channel not in instrument_map:
				instrument_map[channel] = -1 if channel == 9 and not globals()["NO_DRUMS"] else 0
			last_timestamp = max(last_timestamp, int(event[1]))
		elif mode == "note_off_c":
			last_timestamp = max(last_timestamp, int(event[1]))
		elif mode == "control_c":
			control = int(event[4])
			if control == 6:
				channel = int(event[3])
				pitchbend_ranges[channel] = int(event[5])
	active_notes = {i: [] for i in range(len(material_map))}
	active_notes[-1] = []
	print("Instrument mapping:", instrument_map)
	channel_stats = {}
	midi_events.sort(key=lambda x: (int(x[1]), x[2].strip().casefold() not in ("tempo", "header"))) # Sort events by timestamp
	real_ms_per_clock, scale, orig_step_ms, orig_tempo = get_step_speed(midi_events, tps=tps)
	step_ms = orig_step_ms
	midi_events = deque(midi_events)
	timestamp = 0
	loud = 0
	note_candidates = 0
	print("Processing notes...")
	with tqdm.tqdm(total=ceil(last_timestamp * real_ms_per_clock / scale / 1000), bar_format="{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as bar:
		while midi_events:
			event = midi_events[0]
			event_timestamp = int(event[1])
			if event_timestamp > last_timestamp:
				break
			curr_step = step_ms
			time = event_timestamp * real_ms_per_clock / scale
			if time + curr_step / 2 < timestamp:
				# Process all events at the current timestamp
				midi_events.popleft()
				mode = event[2].strip().casefold()
				if mode == "note_on_c":
					channel = int(event[3])
					instrument = instrument_map[channel]
					pitch = int(event[4])
					velocity = int(event[5]) # Note velocity currently not fully supported, used only for note off
					if velocity == 0:
						notec = len(active_notes[instrument])
						for i in range(notec):
							note = active_notes[instrument][notec - i - 1]
							if note.pitch == pitch:
								note.sustain = False
					elif velocity < loud * 0.0625 and sum(map(len, active_notes.items())) >= 16:
						note_candidates += 1
					else:
						note_candidates += 1
						sustain = sustain_map[instrument]
						end = inf
						if sustain:
							off = None
							for i, e in enumerate(midi_events):
								m = e[2].strip().casefold()
								if m == "note_off_c" or m == "note_on_c" and int(e[5]) <= 1:
									c = int(e[3])
									p = int(e[4])
									if c == channel and p == pitch:
										off = i
										end = int(e[1]) * real_ms_per_clock / scale
										break
							if off is not None:
								del midi_events[off]
						if end == inf:
							end = timestamp + step_ms
							sustain = False
						note = SimpleNamespace(pitch=pitch, velocity=velocity, start=timestamp, timestamp=timestamp, end=end, channel=channel, sustain=sustain, updated=True, volume=0)
						active_notes[instrument].append(note)
						loud = max(loud, velocity)
				elif mode == "pitch_bend_c":
					note_candidates += 1
					channel = int(event[3])
					value = int(event[4])
					offset = round((value - 8192) / 8192 * pitchbend_ranges.get(channel, 2))
					if offset != channel_stats.get(channel, {}).get("bend", 0):
						instrument = instrument_map[channel]
						channel_stats.setdefault(channel, {})["bend"] = offset
						candidate = None
						for note in active_notes[instrument]:
							if note.channel == channel:
								note.updated = True
								if not candidate or note.start > candidate.start:
									candidate = note
						if candidate:
							candidate.end += 50
				elif mode == "control_c" and event[4] == "7":
					channel = int(event[3])
					value = int(event[5])
					volume = value / 100
					orig_volume = channel_stats.setdefault(channel, {}).get("volume", 1)
					if volume >= orig_volume * 1.1:
						note_candidates += 1
						instrument = instrument_map[channel]
						for note in active_notes[instrument]:
							if note.channel == channel:
								note.updated = True
					elif volume <= orig_volume * 0.5:
						for note in active_notes[instrument]:
							if note.channel == channel:
								note.sustain = False
					channel_stats[channel]["volume"] = volume
				elif mode == "tempo":
					tempo = int(event[3])
					step_ms = orig_tempo / tempo * orig_step_ms
					if not played_notes:
						timestamp = time
			else:
				beat = []
				started = {}
				max_volume = 0
				for notes in active_notes.values():
					for note in notes:
						note.volume = channel_stats.get(note.channel, {}).get("volume", 1) * note.velocity
						if note.volume > max_volume:
							max_volume = note.volume
				for instrument, notes in active_notes.items():
					notes.reverse()
					for i in range(len(notes) - 1, -1, -1):
						note = notes[i]
						volume = note.volume
						length = note.end - note.start
						sa = globals()["STRUM_AFFINITY"]
						long = note.sustain and length > 150 / sa and volume >= min(80, max_volume) / sa
						needs_sustain = note.sustain and length > 200 / sa and (length > 400 / sa or long and (volume >= 120 / sa or note.sustain == 1 and length >= 300 / sa))
						recur = inf
						if needs_sustain:
							if volume >= 110 / sa and volume == max_volume:
								recur = 50
							elif volume >= 80 / sa:
								recur = 100
							elif volume >= 40 / sa:
								recur = 200
							elif volume >= 20 / sa:
								recur = 400
							if note.sustain > 1 and recur < 200:
								recur *= 2
							if note.updated and recur > 50 and recur < 800:
								h = (note.channel, recur)
								offset = round(recur * binary_fracs[started.setdefault(h, 0)] / 50) * 50
								started[h] += 1
								recur += offset
						if note.updated or (timestamp >= note.timestamp and timestamp + recur < note.end):
							pitch = channel_stats.get(note.channel, {}).get("bend", 0) + note.pitch
							block = (instrument, pitch, note.updated, long, volume)
							if block not in beat:
								beat.append(block)
								note.timestamp = timestamp + recur
						if not note.sustain or timestamp + step_ms * 2 >= note.end:
							notes.pop(i)
						else:
							note.updated = False
					notes.reverse()
				if beat or played_notes:
					played_notes.append(beat)
				timestamp += curr_step
				bar.update(curr_step / 1000)
				loud *= 0.5
	while not played_notes[-1]:
		played_notes.pop(-1)
	return played_notes, note_candidates, is_org

MAIN = 4
SIDE = 2
DIV = 4
BAR = 32
def render_minecraft(notes, transpose=0):
	def extract_notes(notes, offset, direction=1, invert=False, elevations=()):
		for i in range(16):
			beat = []
			for j in range(4):
				try:
					block = notes.pop(0)
				except IndexError:
					block = []
				beat.append(block)
			if not beat:
				break
			for pulse in (0, 1, 2, 3):
				if pulse >= len(beat):
					break
				curr = beat[pulse]
				if not curr:
					continue
				cap = MAIN * 2 + SIDE * 2 + 2 if pulse == 0 else SIDE * 4 if pulse == 2 else SIDE * 2
				padding = (-1, 0, inf, 0, 0)
				transparent = ("glowstone", "heavy_core", "blue_stained_glass", "red_stained_glass")
				lowest = min((note[1], note) for note in curr)[1]
				ordered = sorted(curr, key=lambda note: (note[2], round(note[4] * 8), note[0] == -1, note[1]), reverse=True)[:cap]
				if lowest != padding and lowest not in ordered:
					ordered[-1] = lowest
				if pulse == 0:
					x = (-3 - i if direction == 1 else -18 + i) * (-1 if invert else 1)
					y = 0
					z = offset
					ordered, remainder = ordered[:MAIN * 2], ordered[MAIN * 2:]
					required_padding = sum(get_note_mat(note, transpose=transpose)[0] in transparent for note in ordered) * 2 - len(ordered)
					for p in range(required_padding):
						ordered.append(padding)
					ordered.sort(key=lambda note: get_note_mat(note, transpose=transpose)[0] in transparent)
					while ordered and ordered[-1] == padding:
						ordered.pop(-1)
					if ordered:
						yield ((x, y, z + 1), "observer", dict(facing="north"))
						if len(ordered) > MAIN / 2:
							yield ((x, y + 1, z), "observer", dict(facing="down"))
							if len(ordered) > MAIN:
								yield ((-x, y, z + 1), "observer", dict(facing="north"))
								if len(ordered) > MAIN * 3 / 2:
									yield ((-x, y + 1, z), "observer", dict(facing="down"))
					if len(ordered) & 1:
						note = ordered[-1]
						ordered = list(itertools.chain.from_iterable(zip(ordered[:len(ordered) // 2], ordered[len(ordered) // 2:-1])))
						if get_note_mat(note, transpose=transpose)[0] in transparent:
							ordered.append(padding)
						ordered.append(note)
					else:
						ordered = list(itertools.chain.from_iterable(zip(ordered[:len(ordered) // 2], ordered[len(ordered) // 2:])))
					for j, note in enumerate(ordered[:MAIN * 2]):
						replace = {}
						if j == 0:
							positioning = [x, y, z + 2]
							replace["glowstone"] = "snow_block"
						elif j == 1:
							positioning = [x, y - 1, z + 3]
						elif j == 2:
							positioning = [x, y + 2, z]
							replace["glowstone"] = "snow_block"
							replace["heavy_core"] = "black_concrete_powder"
						elif j == 3:
							positioning = [x, y + 1, z + 1]
							replace["heavy_core"] = "black_concrete_powder"
						elif j == 4:
							positioning = [-x, y, z + 2]
							replace["glowstone"] = "snow_block"
						elif j == 5:
							positioning = [-x, y - 1, z + 3]
						elif j == 6:
							positioning = [-x, y + 2, z]
							replace["glowstone"] = "snow_block"
							replace["heavy_core"] = "black_concrete_powder"
						elif j == 7:
							positioning = [-x, y + 1, z + 1]
							replace["heavy_core"] = "black_concrete_powder"
						else:
							raise ValueError(j)
						yield from get_note_block(
							note,
							positioning,
							replace,
							transpose=transpose,
						)
					ordered = ordered[MAIN * 2:] + remainder
					while True:
						try:
							ordered.remove(padding)
						except ValueError:
							break
					taken = False
					for note in ordered[SIDE * 2:]:
						note = list(note)
						note[3] = 2
						note = tuple(note)
						mat, pitch = get_note_mat(note, transpose=transpose)
						if mat != "PLACEHOLDER":
							yield ((-x if taken else x, y + 2, z - 1), "note_block", dict(note=pitch, instrument="harp"))
							if taken:
								break
							else:
								taken = True
					ordered = ordered[:SIDE * 2]
				for y in elevations[pulse]:
					cap = SIDE
					left, right, ordered = ordered[:cap], ordered[cap:cap * 2], ordered[cap * 2:]
					v = 0
					if i >= 8:
						x = (1 - (17 - i) * 2 if direction == 1 else -22 + (17 - i) * 2) * (-1 if invert else 1)
						v = 0 if y < 0 else -2
						z = offset + 2
					else:
						x = (-4 - i * 2 if direction == 1 else -17 + i * 2) * (-1 if invert else 1)
						v = -1
						z = offset
					for w, note in enumerate(left):
						yield from get_note_block(
							note,
							[x, y + v, z - w],
							transpose=transpose,
						)
					if i >= 8:
						x = (1 - (17 - i) * 2 if direction == 1 else -22 + (17 - i) * 2) * (1 if invert else -1)
						v = 0 if y < 0 else -2
						z = offset + 2
					else:
						x = (-4 - i * 2 if direction == 1 else -17 + i * 2) * (1 if invert else -1)
						v = -1
						z = offset
					for w, note in enumerate(right):
						yield from get_note_block(
							note,
							[x, y + v, z - w],
							transpose=transpose,
						)

	def generate_layer(direction="right", offset=0, elevation=0):
		right = direction == "right"
		x = -1 if right else 1
		y = elevation
		z = offset
		mirrored = offset % 8 >= 4
		if y == 0:
			yield ((x * 19, y - 1, z), "black_stained_glass")
			if mirrored:
				yield from (
					((x * 19, y, z), "observer", dict(facing="north")),
					((x * 19, y, z + 1), "redstone_lamp"),
					((x * 2, y - 1, z + 3), "black_stained_glass"),
					((x * 2, y, z + 2), "black_stained_glass"),
					((x * 2, y, z + 1), "target"),
					((x * 2, y + 1, z + 2), "activator_rail", dict(shape="north_south")),
					((x * 2, y + 1, z + 1), "activator_rail", dict(shape="north_south")),
					((x * 2, y, z + 3), "activator_rail", dict(shape="ascending_north")),
					((x * 2, y - 1, z), "sculk"),
					((x, y - 2, z), "sculk"),
					((0, y - 3, z), "polished_blackstone_slab", dict(type="top")),
					((-x, y - 2, z), "sculk"),
					((-x * 2, y - 1, z), "sculk"),
					((x * 2, y, z), "redstone_wire", dict(east="side", west="side", north="none", south="side")),
					((x, y - 1, z), "redstone_wire", dict(east="up" if not right else "side", west="up" if right else "side", north="none", south="none")),
					((0, y - 2, z), "redstone_wire", dict(east="up", west="up", north="none", south="none")),
					((-x, y - 1, z), "redstone_wire", dict(east="up" if right else "side", west="up" if not right else "side", north="none", south="none")),
					((-x * 2, y, z), "redstone_wire", dict(east="side", west="side", north="none", south="side")),
					((-x * 2, y - 1, z + 3), "black_stained_glass"),
					((-x * 2, y, z + 2), "black_stained_glass"),
					((-x * 2, y, z + 1), "target"),
					((-x * 2, y + 1, z + 2), "activator_rail", dict(shape="north_south")),
					((-x * 2, y + 1, z + 1), "activator_rail", dict(shape="north_south")),
					((-x * 2, y, z + 3), "activator_rail", dict(shape="ascending_north")),
				)
			else:
				yield from (
					((x * 2, y - 1, z), "black_stained_glass"),
					((x * 2, y, z), "observer", dict(facing="north")),
					((x * 2, y, z + 1), "redstone_lamp"),
					((x, y - 1, z + 1), "glass"),
					((x, y, z + 1), "repeater", dict(facing="west" if right else "east", delay=4)),
					((x, y + 1, z + 1), "glass"),
					((0, y, z), "glass"),
					((0, y, z + 1), "powered_rail", dict(shape="ascending_north")),
					((0, y, z - 1), "activator_rail", dict(shape="ascending_south")),
					((0, y + 1, z), "rail", dict(shape="north_south")),
					((x * 19, y - 1, z + 3), "black_stained_glass"),
					((x * 19, y, z), "activator_rail", dict(shape="ascending_south")),
					((x * 19, y, z + 3), "activator_rail", dict(shape="ascending_north")),
					((x * 19, y, z + 1), "black_stained_glass"),
					((x * 19, y, z + 2), "black_stained_glass"),
					((x * 19, y + 1, z + 1), "activator_rail", dict(shape="north_south")),
					((x * 19, y + 1, z + 2), "activator_rail", dict(shape="north_south")),
				)
		if y == 0:
			yield ((x * 3, y - 1, z), "beacon")
			for i in range(4, 18):
				yield ((x * i, y - 1, z), "crimson_trapdoor" if i & 1 == right else "acacia_trapdoor", dict(facing="south", half="top"))
			yield ((x * 18, y - 1, z), "beacon")
			yield ((x * (18 if mirrored else 3), y, z), "observer", dict(facing="east" if right ^ mirrored else "west"))
			for i in range(3, 18):
				yield ((x * (i if mirrored else i + 1), y, z), "repeater", dict(facing="east" if right ^ mirrored else "west", delay=2))
		elif y != 0:
			lower = y < 0
			mid = y == -3
			block = "crying_obsidian" if mid else "slime_block" if lower else "pearlescent_froglight"
			slab = "oxidized_cut_copper_slab" if mid else "bamboo_mosaic_slab" if lower else "prismarine_slab"
			edge = "purpur_slab" if mid else "resin_brick_slab" if lower else "dark_prismarine_slab"
			o1 = 3 + mirrored
			o2 = 4 - mirrored
			for i in range(8):
				yield from (
					((x * (i * 2 + o2), y, z), block),
					((x * (i * 2 + o2 - 1), y - 1, z), slab, dict(type="top")),
					((x * (i * 2 + o2 - 1), y, z), "repeater", dict(facing="east" if right ^ mirrored else "west", delay=2)),
					((x * (i * 2 + o1), y + (1 if lower else -1), z + 2), block),
					((x * (i * 2 + o1 + 1), y + (0 if lower else -2), z + 2), slab, dict(type="top")),
					((x * (i * 2 + o1 + 1), y + (1 if lower else -1), z + 2), "repeater", dict(facing="west" if right ^ mirrored else "east", delay=2)),
				)
			x2 = x * 2 if mirrored else x * 19 
			yield from (
				((x * (18 if mirrored else 3), y, z), "observer", dict(facing="east" if right ^ mirrored else "west")),
				((x * (3 if mirrored else 18), y + (1 if lower else -1), z + 2), "observer", dict(facing="west" if right ^ mirrored else "east")),
				((x2, y + (1 if lower else -1), z + 2), "observer", dict(facing="north")),
				((x2, y - 1, z), edge, dict(type="top")),
				((x2, y + (0 if lower else -2), z + 1), edge, dict(type="top")),
				((x2, y + (0 if lower else -2), z + 2), edge, dict(type="top")),
				((x2, y + (1 if lower else -1), z + 1), "powered_rail", dict(shape="north_south" if lower else "ascending_north")),
				((x2, y, z), "powered_rail", dict(shape="ascending_south" if lower else "north_south")),
			)

	def ensure_layer(direction="right", offset=0):
		right = direction == "right"
		x = -1 if right else 1
		z = offset
		mirrored = offset % 8 >= 4
		x2 = x * 19 if mirrored else x * 2
		yield from (
			((x2, -1, z + 1), "oxidized_copper_trapdoor", dict(facing="south", half="top")),
			((x2, -1, z), "prismarine_wall"),
			((x2, -2, z), "prismarine_wall"),
			((x2, -3, z), "observer", dict(facing="up")),
			((x2, -4, z), "redstone_lamp"),
			((x2, -5, z), "warped_fence_gate", dict(facing="north")),
			((x2, -6, z), "observer", dict(facing="up")),
			((x2, -7, z), "gilded_blackstone"),
			((x2, -9, z), "note_block"),
			((x2, -8, z), "redstone_wire", dict(east="none", north="none", south="none", west="none")),
			((x * (20 if mirrored else 1), -1, z), "glass"),
			((x * (20 if mirrored else 1), -2, z), "glass"),
			((x * (18 if mirrored else 3), -2, z), "glass"),
		)

	def ensure_top(direction="right", offset=0):
		right = direction == "right"
		x = -1 if right else 1
		z = offset
		mirrored = offset % 8 >= 4
		if mirrored:
			x1, x2 = x * 20, x * 19
		else:
			x1, x2 = x, x * 2
		z2 = z + 1
		yield from (
			((x1, 1, z2), "glass"),
			((x1, 2, z2), "scaffolding", dict(distance=0)),
			((x2, 1, z2), "bamboo_trapdoor", dict(facing="south", half="top")),
			((x2, 2, z2), "scaffolding", dict(distance=0)),
			((x2, 3, z2), "observer", dict(facing="down")),
			((x2, 4, z2), "redstone_lamp"),
			((x2, 5, z), "white_stained_glass"),
			((x2, 6, z2), "white_stained_glass"),
			((x2, 7, z), "white_stained_glass"),
			((x2, 8, z2), "white_stained_glass"),
			((x2, 9, z), "observer", dict(facing="south")),
			((x2, 5, z2), "redstone_wire", dict(north="up")),
			((x2, 6, z), "redstone_wire", dict(south="up")),
			((x2, 7, z2), "redstone_wire", dict(north="up")),
			((x2, 8, z), "redstone_wire", dict(south="up")),
			((x2, 9, z2), "redstone_wire", dict(north="side")),
		)

	def profile_notes(notes):
		return (max(map(len, notes[i:64:4]), default=0) for i in range(4))

	print("Preparing output...")
	bars = ceil(len(notes) / BAR / DIV)
	elevations = ((-3,), (6,), (-6, -9), (9,))
	for b in tqdm.trange(bars):
		inverted = not b & 1
		left, right = ("left", "right") if not inverted else ("right", "left")
		offset = b * 8 + 1
		for i in range(offset, offset + 8):
			yield from (
				((0, -1, i), "tinted_glass"),
				((-1, 0, i), "glass"),
				((1, 0, i), "glass"),
				((0, 0, i), "rail", dict(shape="north_south")),
			)

		def iter_half(main=True):
			strong, weak1, mid, weak2 = profile_notes(notes)
			if not main and not strong and not weak1 and not mid and not weak2:
				return
			yield from generate_layer(right, offset, 0)
			if strong > MAIN or mid > SIDE or weak1 > SIDE or weak2 > SIDE:
				yield from generate_layer(left, offset, 0)
			if weak1 or weak2:
				yield from ensure_top(right, offset)
				if weak1:
					yield from generate_layer(right, offset, 6)
				if weak2:
					yield from generate_layer(right, offset, 9)
				if weak1 > SIDE or weak2 > SIDE:
					yield from ensure_top(left, offset)
					if weak1 > SIDE:
						yield from generate_layer(left, offset, 6)
					if weak2 > SIDE:
						yield from generate_layer(left, offset, 9)
			if strong > MAIN * 2 or mid:
				yield from ensure_layer(right, offset)
				if strong > MAIN * 2 + SIDE or mid > SIDE:
					yield from ensure_layer(left, offset)
				if strong > MAIN * 2:
					yield from generate_layer(right, offset, -3)
					if strong > MAIN * 2 + SIDE:
						yield from generate_layer(left, offset, -3)
				if mid:
					yield from generate_layer(right, offset, -6)
					if mid > SIDE:
						yield from generate_layer(left, offset, -6)
						if mid > SIDE * 2:
							yield from generate_layer(right, offset, -9)
							if mid > SIDE * 3:
								yield from generate_layer(left, offset, -9)

		yield from iter_half()
		yield from extract_notes(notes, offset, 1, inverted, elevations)

		offset += 4
		yield from iter_half()
		yield from extract_notes(notes, offset, -1, inverted, elevations)

	offset = b * 8 + 8
	for i in range(-2, 3):
		yield ((i, -1, 0), "dark_prismarine")
		yield ((i, 0, 0), "activator_rail", dict(shape="east_west", powered="false"))
	yield from (
		((0, 0, offset), "powered_rail", dict(shape="north_south")),
		((0, 0, 0), "crying_obsidian"),
		((0, 1, 0), "lever", dict(facing="north", face="floor", powered="false")),
	)

def convert_file(args):
	if not args.output:
		path, name = args.input.replace("\\", "/").rsplit("/", 1)
		args.output = path + "/" + name.rsplit(".", 1)[0] + ".litematic"
	if args.transpose is not None:
		globals()["TRANSPOSE"] = args.transpose
	if args.speed is not None:
		globals()["SPEED_MULTIPLIER"] = args.speed
	if args.strum_affinity is not None:
		globals()["STRUM_AFFINITY"] = args.strum_affinity
	if args.drums is False:
		globals()["NO_DRUMS"] = 1
	print("Converting midi...")
	csv_list = py_midicsv.midi_to_csv(args.input)
	midi_events = list(csv.reader(csv_list))
	notes, note_candidates, is_org = convert_midi(midi_events)
	print("Note candidates:", note_candidates)
	print("Note count:", sum(map(len, notes)))
	print("Max detected polyphony (will be reduced to <=14):", max(map(len, notes)))
	print("Lowest note:", min(min(n[1] for n in b) for b in notes if b))
	print("Highest note:", max(max(n[1] for n in b) for b in notes if b))
	lazy = render_minecraft(notes, transpose=globals()["TRANSPOSE"] + 12 if is_org else globals()["TRANSPOSE"])
	bars = ceil(len(notes) / BAR / DIV)
	depth = bars * 8 + 8
	nc = 0
	if args.output.endswith(".mcfunction"):
		with open(args.output, "w") as f:
			for (x, y, z), block, *kwargs in lazy:
				nc += block == "note_block"
				if kwargs:
					extra = "[" + ",".join(f"{k}={v}" for k, v in kwargs[0].items()) + "]"
				else:
					extra = ""
				f.write(f"setblock ~{x} ~{y} ~{z} {block}{extra}\n")
	else:
		import litemapy
		mx, my = 20, 11
		reg = litemapy.Region(-mx, -my, 0, mx * 2 + 1, my * 2 + 1, depth)
		schem = reg.as_schematic(
			name=args.output.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0],
			author="Hyperchoron",
			description="Exported MIDI",
		)
		for (x, y, z), block, *kwargs in lazy:
			nc += block == "note_block"
			if kwargs:
				block = litemapy.BlockState("minecraft:" + block, **{k: str(v) for k, v in kwargs[0].items()})
			else:
				block = litemapy.BlockState("minecraft:" + block)
			reg[x + mx, y + my, z] = block
		schem.save(args.output)
	print("Final note count:", nc)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		prog="",
		description="MIDI to Minecraft Note Block Converter",
	)
	parser.add_argument("-i", "--input", required=True, help="Input file (.mid)")
	parser.add_argument("-o", "--output", nargs="?", help="Output file (.mcfunction | .litematic)")
	parser.add_argument("-t", "--transpose", nargs="?", type=int, help="Transposes song up/down a certain amount of semitones; higher = higher pitched")
	parser.add_argument("-s", "--speed", nargs="?", type=float, help="Scales song speed up/down as a multiplier; higher = faster")
	parser.add_argument("-sa", "--strum-affinity", nargs="?", type=float, help="Increases or decreases threshold for sustained notes to be cut into discrete segments; higher = more notes")
	parser.add_argument("-d", "--drums", action=argparse.BooleanOptionalAction, default=True, help="Disables percussion channel")
	args = parser.parse_args()
	convert_file(args)