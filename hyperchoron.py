# Coco eats a gold block on the 18/2/2025. Nom nom nom nom. Output sound weird? Sorgy accident it was the block I eated. Rarararrarrrr 😋

from collections import deque
import csv
import fractions
import functools
import itertools
from math import ceil, inf, isqrt, sqrt, log2, gcd
import os
from types import SimpleNamespace
if os.name == "nt" and os.path.exists("Midicsv.exe"):
	py_midicsv = None
else:
	import py_midicsv
try:
	import tqdm
except ImportError:
	import contextlib
	tqdm = None
else:
	import warnings
	warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)


# Predefined list attempting to match instruments across pitch ranges
material_map = [
	["bamboo_planks", "black_wool", "black_wool+", "amethyst_block+", "gold_block", "gold_block+"],
	["bamboo_planks", "bamboo_planks+", "glowstone", "glowstone+", "gold_block", "gold_block+"],
	["pumpkin", "pumpkin+", "amethyst_block", "clay", "clay+", "packed_ice+"],
	["pumpkin", "pumpkin+", "emerald_block", "emerald_block+", "gold_block", "gold_block+"],
	["bamboo_planks", "bamboo_planks+", "iron_block", "iron_block+", "gold_block", "gold_block+"],
	["bamboo_planks", "black_wool", "amethyst_block", "amethyst_block+", "packed_ice", "packed_ice+"],
	["cobblestone", "cobblestone+", "red_stained_glass", "red_stained_glass+", "heavy_core", "heavy_core+"],
	["bamboo_planks", "black_wool", "hay_block", "hay_block+", "soul_sand+", "bone_block+"],
	None
]
default_instruments = dict(
	harp="Plucked",
	pling="Keyboard",
	flute="Wind",
	bit="Synth",
	iron_xylophone="Pitched Percussion",
	chime="Bell",
	basedrum="Unpitched Percussion",
	banjo="String",
	creeper="Drumset",
)
instrument_codelist = list(default_instruments.values())
default_instruments.update(dict(
	snare="Unpitched Percussion",
	hat="Unpitched Percussion",
	bell="Plucked",
	cow_bell="Pitched Percussion",
	didgeridoo="Wind",
	guitar="Plucked",
	bass="Plucked",
))
instrument_names = dict(
	amethyst_block="harp",
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
nbs_names = {k: i for i, k in enumerate([
	"harp",
	"bass",
	"basedrum",
	"snare",
	"hat",
	"guitar",
	"flute",
	"bell",
	"chime",
	"xylophone",
	"iron_xylophone",
	"cow_bell",
	"didgeridoo",
	"bit",
	"banjo",
	"pling",
])}
nbs_values = {v: k for k, v in nbs_names.items()}
for unsupported in ("skeleton", "wither_skeleton", "zombie", "creeper", "piglin"):
	nbs_names[unsupported] = nbs_names["snare"]
pitches = dict(
	harp=24,
	bass=0,
	basedrum=0,
	snare=48,
	hat=24,
	guitar=12,
	flute=36,
	bell=48,
	chime=48,
	xylophone=48,
	iron_xylophone=24,
	cow_bell=36,
	didgeridoo=0,
	bit=24,
	banjo=24,
	pling=24,
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
	1, 2, 3, 3, 2, 2, 3, 7, 7, 7,
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
		return gcd(*arr), len(arr)

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
		return gcd(*arr), len(arr)

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

	return (max_gcd, len(arr) - max_count) if max_gcd >= min_value else (gcd(*arr), len(arr))

# Remapping of midi note range to note block note range
c4 = 60
fs4 = c4 + 6
fs1 = fs4 - 36

@functools.lru_cache(maxsize=256)
def get_note_mat(note, transpose, odd=False):
	material = material_map[note[0]]
	pitch = note[1]
	if not material:
		try:
			return percussion_mats[pitch]
		except KeyError:
			print("WARNING: Note", pitch, "not yet supported for drums, discarding...")
			return "PLACEHOLDER", 0
	pitch += transpose
	normalised = pitch - fs1
	if normalised < 0:
		normalised += 12
	elif normalised > 72:
		normalised -= 12
	assert 0 <= normalised <= 72, normalised
	ins, mod = divmod(normalised, 12)
	if mod == 0 and (ins > 5 or ins > 0 and odd):
		mod += 12
		ins -= 1
	elif odd:
		leeway = 1
		if ins > 0 and mod <= leeway and not material[ins - 1].endswith("+"):
			mod += 12
			ins -= 1
		elif ins < 5 and mod >= 24 - leeway and material[ins + 1].endswith("+"):
			mod -= 12
			ins += 1
		elif ins < 5 and mod >= 12 - leeway and material[ins].endswith("+") and material[ins + 1].endswith("+"):
			mod -= 12
			ins += 1
	mat = material[ins]
	if note[3] == 2:
		replace = dict(
			hay_block="amethyst_block",
			emerald_block="amethyst_block",
			amethyst_block="amethyst_block",
			iron_block="amethyst_block",
			glowstone="amethyst_block",
		)
		replace.update({
			"hay_block+": "amethyst_block+",
			"emerald_block+": "amethyst_block+",
			"amethyst_block+": "amethyst_block+",
			"black_wool+": "amethyst_block",
			"iron_block+": "amethyst_block+",
			"glowstone+": "amethyst_block+",
		})
		try:
			mat = replace[mat]
		except KeyError:
			return "PLACEHOLDER", 0
	elif note[3] and (not odd or mod not in (0, 1, 11, 12, 23, 24)):
		replace = dict(
			bamboo_planks="pumpkin",
			bone_block="gold_block",
			iron_block="amethyst_block",
			glowstone="amethyst_block",
		)
		replace.update({
			"bamboo_planks+": "pumpkin+",
			"gold_block+": "packed_ice+",
			"bone_block+": "gold_block+",
			"iron_block+": "amethyst_block+",
			"glowstone+": "amethyst_block+",
		})
		try:
			mat = replace[mat]
		except KeyError:
			match mat:
				case "black_wool":
					if mod <= 12:
						mat = "pumpkin"
						mod += 12
					else:
						mat = "amethyst_block"
						mod -= 12
				case "black_wool+":
					if mod >= 0:
						mat = "amethyst_block"
				case "soul_sand":
					if mod <= 12:
						mat = "amethyst_block"
						mod += 12
					else:
						mat = "gold_block"
						mod -= 12
				case "soul_sand+":
					if mod >= 0:
						mat = "gold_block"
	if mat.endswith("+"):
		mat = mat[:-1]
		mod += 12
	return mat, mod

def get_note_block(note, positioning=[0, 0, 0], replace=None, odd=False, ctx=None):
	base, pitch = get_note_mat(note, transpose=ctx.transpose if ctx else 0, odd=odd)
	x, y, z = positioning
	coords = [(x, y, z), (x, y + 1, z), (x, y + 2, z)]
	if replace and base in replace:
		base = replace[base]
	if base == "PLACEHOLDER":
		return (
			(coords[0], "mangrove_roots"),
		)
	if base.endswith("_head") or base.endswith("_skull"):
		return (
			(coords[0], "magma_block"),
			(coords[1], "note_block", dict(note=pitch, instrument=instrument_names[base])),
			(coords[2], base),
		)
	return (
		(coords[0], base),
		(coords[1], "note_block", dict(note=pitch, instrument=instrument_names[base])),
	)

def get_step_speed(midi_events, tps=20, ctx=None):
	orig_tempo = 0
	clocks_per_crotchet = 0
	milliseconds_per_clock = 0
	timestamps = {}
	time_diffs = {}
	tempos = {}
	since_tempo = 0
	for event in midi_events:
		mode = event[2].strip().casefold()
		timestamp = int(event[1])
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
			if orig_tempo:
				tempos[orig_tempo] = tempos.get(orig_tempo, 0) + timestamp - since_tempo
			since_tempo = timestamp
			orig_tempo = int(event[3])
			milliseconds_per_clock = orig_tempo / 1000 / clocks_per_crotchet
	if tempos:
		tempos[orig_tempo] = tempos.get(orig_tempo, 0) + int(midi_events[-1][1]) - since_tempo
		tempo_list = list(tempos.items())
		tempo_list.sort(key=lambda t: t[1], reverse=True)
		orig_tempo = tempo_list[0][0]
		milliseconds_per_clock = orig_tempo / 1000 / clocks_per_crotchet
		print("Multiple tempos detected! Auto-selecting most common from:", min(tempo_list), max(tempo_list))
	if not clocks_per_crotchet:
		clocks_per_crotchet = 4 # Default to 4/4 time signature
	if not milliseconds_per_clock:
		orig_tempo = orig_tempo or 60 * 120
		milliseconds_per_clock = orig_tempo / 1000 / clocks_per_crotchet # Default to 120 BPM
	step_ms = round(1000 / tps)
	rev = deque(sorted(timestamps, key=lambda k: timestamps[k]))
	mode = timestamps[rev[-1]]
	while len(timestamps) > 1024 or timestamps[rev[0]] < mode / 64:
		timestamps.pop(rev.popleft())
	timestamp_collection = list(itertools.chain.from_iterable([k] * ceil(max(1, log2(v / 2))) for k, v in timestamps.items()))
	min_value = step_ms / milliseconds_per_clock
	print("Estimating true resolution...", len(timestamp_collection), clocks_per_crotchet, milliseconds_per_clock, step_ms, min_value)
	speed, exclusions = approximate_gcd(timestamp_collection, min_value=min_value - 1)
	use_exact = False
	req = 1 / 8
	print("Confidence:", 1 - exclusions / len(timestamp_collection), req)
	if speed > min_value * 1.25 or exclusions > len(timestamp_collection) * req:
		if exclusions >= len(timestamp_collection) * 0.75:
			speed2 = milliseconds_per_clock / step_ms * clocks_per_crotchet
			while speed2 > sqrt(2):
				speed2 /= 2
			print("Rejecting first estimate:", speed, speed2, min_value, exclusions)
			step = 1 / speed2
			inclusions = sum((res := x % step) < step / 12 or res > step - step / 12 or (res := x % (step * 4)) < (step * 4) / 12 or res > (step * 4) - (step * 4) / 12 for x in timestamp_collection)
			req = (max(speed2, 1 / speed2) - 1) * sqrt(2)
			print("Confidence:", inclusions / len(timestamp_collection), req)
			if inclusions < len(timestamp_collection) * req:
				print("Discarding tempo...")
				speed = 1
			else:
				speed = speed2
			use_exact = True
		elif speed > min_value * 1.25:
			print("Finding closest speed...", exclusions, len(timestamps))
			div = round(speed / min_value - 0.25)
			if div <= 1:
				if speed % 3 == 0:
					print("Speed too close for rounding, autoscaling by 2/3...")
					speed *= 2
					speed //= 3
				else:
					print("Speed too close for rounding, autoscaling by 75%...")
					speed *= 0.75
			else:
				speed /= div
	if use_exact:
		real_ms_per_clock = milliseconds_per_clock
	else:
		real_ms_per_clock = round(milliseconds_per_clock * min_value / step_ms) * step_ms
	print("Final speed scale:", speed, real_ms_per_clock)
	return real_ms_per_clock, speed, step_ms, orig_tempo

def convert_midi(midi_events, speed_info, ctx=None):
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
	channel_stats = {}
	last_timestamp = 0
	max_vel = 0
	for event in midi_events:
		mode = event[2].strip().casefold()
		match mode:
			case "program_c":
				channel = int(event[3])
				if channel == 9 and ctx.drums:
					continue
				value = int(event[4])
				instrument_map[channel] = org_instrument_mapping[value] if is_org else instrument_mapping[value]
			case "note_on_c":
				channel = int(event[3])
				velocity = int(event[5])
				if channel not in instrument_map:
					instrument_map[channel] = -1 if channel == 9 and ctx.drums else 0
				last_timestamp = max(last_timestamp, int(event[1]))
				volume = velocity * channel_stats.setdefault(channel, {}).get("volume", 1)
				max_vel = max(max_vel, volume)
			case "note_off_c":
				last_timestamp = max(last_timestamp, int(event[1]))
			case "control_c":
				control = int(event[4])
				if control == 7:
					channel = int(event[3])
					value = int(event[5])
					volume = value / 127
					channel_stats.setdefault(channel, {})["volume"] = volume
	print("Instrument mapping:", instrument_map)
	active_notes = {i: [] for i in range(len(material_map))}
	active_notes[-1] = []
	midi_events.sort(key=lambda x: (int(x[1]), x[2].strip().casefold() not in ("tempo", "header", "control_c"))) # Sort events by timestamp, keep headers first
	real_ms_per_clock, scale, orig_step_ms, orig_tempo = speed_info
	step_ms = orig_step_ms
	midi_events = deque(midi_events)
	timestamp = 0
	loud = 0
	note_candidates = 0
	print("Max volume:", max_vel)
	print("Processing notes...")
	progress = tqdm.tqdm(total=ceil(last_timestamp * real_ms_per_clock / scale / 1000), bar_format="{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") if tqdm else contextlib.nullcontext()
	with progress as bar:
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
				match mode:
					case "note_on_c":
						channel = int(event[3])
						instrument = instrument_map[channel]
						pitch = int(event[4])
						velocity = int(event[5])
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
							sustain = sustain_map[instrument] or (2 if is_org else 0)
							end = inf
							if sustain or len(active_notes[instrument]) <= 4 and pitchbend_ranges.get(channel):
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
					case "pitch_bend_c":
						channel = int(event[3])
						value = int(event[4])
						offset = round((value - 8192) / 8192 * pitchbend_ranges.get(channel, 2))
						if offset != channel_stats.get(channel, {}).get("bend", 0):
							note_candidates += 1
							pitchbend_ranges.setdefault(channel, 2)
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
					case "control_c" if event[4].strip() == "6":
						channel = int(event[3])
						pitchbend_ranges[channel] = int(event[5])
					case "control_c" if event[4].strip() == "7":
						channel = int(event[3])
						value = int(event[5])
						volume = value / 127
						orig_volume = channel_stats.setdefault(channel, {}).get("volume", 1)
						if volume >= orig_volume * 1.1:
							if note_candidates:
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
					case "tempo":
						tempo = int(event[3])
						ratio = tempo / orig_tempo
						if max(ratio, 1 / ratio) - 1 < 1 / 16:
							# print(f"Detected small tempo change of ratio {ratio}, ignoring...")
							ratio = 1
						elif ratio > 1:
							r2 = fractions.Fraction(ratio).limit_denominator(8)
							if abs(r2 - ratio) < 1 / 64:
								# print(f"Detected close tempo change of ratio {ratio}, auto-syncing to {r2}...")
								ratio = r2
						elif ratio < 1:
							r2 = fractions.Fraction(1 / ratio).limit_denominator(8)
							if abs(r2 - ratio) < 1 / 64:
								# print(f"Detected close tempo change of ratio {ratio}, auto-syncing to {r2}...")
								ratio = 1 / r2
						new_step = orig_step_ms / ratio
						step_ms = int(new_step) if new_step.is_integer() else new_step
						if not note_candidates:
							timestamp = round(time)
			else:
				started = {}
				ticked = {}
				max_volume = 0
				for notes in active_notes.values():
					for note in notes:
						note.volume = channel_stats.get(note.channel, {}).get("volume", 1) * note.velocity / max_vel * 127
						if note.volume > max_volume:
							max_volume = note.volume
				for instrument, notes in active_notes.items():
					notes.reverse()
					for i in range(len(notes) - 1, -1, -1):
						note = notes[i]
						volume = note.volume
						length = note.end - note.start
						sa = ctx.strum_affinity
						long = note.sustain and length > 150 / sa and volume >= min(80, max_volume) / sa
						needs_sustain = note.sustain and length > 200 / sa and (length > 400 / sa or long and (volume >= 120 / sa or note.sustain == 1 and length >= 300 / sa))
						recur = inf
						if needs_sustain:
							if volume > 100 / sa and volume >= max_volume * 7 / 8:
								recur = 50
							elif volume >= 60 / sa and volume >= max_volume / 2:
								recur = 100
							elif volume >= 20 / sa and volume >= max_volume / 4:
								recur = 200
							else:
								recur = 400
							if recur < inf and length >= 200:
								h = recur
								n = started.setdefault(h, 0)
								if note.updated:
									if recur == 200:
										offset = bool(n & 1) * 100 + bool(n & 2) * 50
									elif recur == 100:
										offset = bool(n & 1) * 50
									else:
										offset = 0
									recur += offset
								started[h] += 1
						if note.updated or (timestamp >= note.timestamp and (timestamp + recur < note.end or timestamp + recur <= note.end and volume == max_volume)):
							pitch = channel_stats.get(note.channel, {}).get("bend", 0) + note.pitch
							normalised = pitch + ctx.transpose - fs1
							if normalised > 84:
								pitch = 84 - ctx.transpose + fs1
							elif normalised < -12:
								pitch = -12 - ctx.transpose + fs1
							bucket = (instrument, pitch)
							try:
								temp = ticked[bucket]
							except KeyError:
								temp = ticked[bucket] = [note.updated, long, volume]
							else:
								temp[0] |= note.updated
								temp[1] |= long
								temp[2] = isqrt(ceil(volume ** 2 + temp[2] ** 2))
							note.timestamp = timestamp + recur
						if timestamp + step_ms * 2 >= note.end or len(notes) >= 64 and not needs_sustain:
							notes.pop(i)
						else:
							note.updated = False
					notes.reverse()
				beat = []
				for k, v in ticked.items():
					instrument, pitch = k
					updated, long, volume = v
					count = max(1, volume // 127)
					vel = max(1, min(127, round(volume / count)))
					block = (instrument, pitch, updated, long, vel)
					for w in range(count):
						beat.append(block)
				played_notes.append(beat)
				timestamp += curr_step
				if bar:
					bar.update(curr_step / 1000)
				loud *= 0.5
	while not played_notes[-1]:
		played_notes.pop(-1)
	return played_notes, note_candidates, is_org, speed_info

MAIN = 4
SIDE = 2
DIV = 4
BAR = 32
def render_minecraft(notes, ctx):
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
				cap = MAIN * 2 + SIDE * 2 + 2 if pulse == 0 else SIDE * 4 if pulse == 2 else SIDE * 3
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
					if len(ordered) > MAIN * 2 + SIDE * 2:
						found = []
						taken = False
						for k, note in enumerate(reversed(ordered)):
							note = list(note)
							note[3] = 2
							note = tuple(note)
							mat, pitch = get_note_mat(note, transpose=ctx.transpose, odd=pulse & 1)
							if mat != "PLACEHOLDER":
								found.append(k)
								yield ((-x if taken else x, y + 2, z - 1), "note_block", dict(note=pitch, instrument="harp"))
								if taken:
									break
								else:
									taken = True
						if found:
							ordered = [note for k, note in enumerate(reversed(ordered)) if k not in found][::-1]
					ordered, remainder = ordered[:MAIN * 2], ordered[MAIN * 2:]
					required_padding = sum(get_note_mat(note, transpose=ctx.transpose, odd=pulse & 1)[0] in transparent for note in ordered) * 2 - len(ordered)
					for p in range(required_padding):
						ordered.append(padding)
					ordered.sort(key=lambda note: get_note_mat(note, transpose=ctx.transpose, odd=pulse & 1)[0] in transparent)
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
						if get_note_mat(note, transpose=ctx.transpose, odd=pulse & 1)[0] in transparent:
							ordered.append(padding)
						ordered.append(note)
					else:
						ordered = list(itertools.chain.from_iterable(zip(ordered[:len(ordered) // 2], ordered[len(ordered) // 2:])))
					for j, note in enumerate(ordered[:MAIN * 2]):
						replace = {}
						match j:
							case 0:
								positioning = [x, y, z + 2]
								replace["glowstone"] = "amethyst_block"
							case 1:
								positioning = [x, y - 1, z + 3]
							case 2:
								positioning = [x, y + 2, z]
								replace["glowstone"] = "amethyst_block"
								replace["heavy_core"] = "black_concrete_powder"
							case 3:
								positioning = [x, y + 1, z + 1]
								replace["heavy_core"] = "black_concrete_powder"
							case 4:
								positioning = [-x, y, z + 2]
								replace["glowstone"] = "amethyst_block"
							case 5:
								positioning = [-x, y - 1, z + 3]
							case 6:
								positioning = [-x, y + 2, z]
								replace["glowstone"] = "amethyst_block"
								replace["heavy_core"] = "black_concrete_powder"
							case 7:
								positioning = [-x, y + 1, z + 1]
								replace["heavy_core"] = "black_concrete_powder"
							case _:
								raise ValueError(j)
						yield from get_note_block(
							note,
							positioning,
							replace,
							odd=pulse & 1,
							ctx=ctx,
						)
					ordered = ordered[MAIN * 2:] + remainder
					while True:
						try:
							ordered.remove(padding)
						except ValueError:
							break
					ordered = ordered[:SIDE * 2]
				cap = SIDE
				flipped = False # direction == 1
				for y, r in elevations[pulse]:
					if not ordered:
						break
					taken, ordered = ordered[:cap], ordered[cap:]
					if i >= 8:
						x = (1 - (17 - i) * 2 if flipped else -22 + (17 - i) * 2) * (-1 if invert ^ r else 1)
						v = 0 if y < 0 else -2
						z = offset + 2
					else:
						x = (-4 - i * 2 if flipped else -17 + i * 2) * (-1 if invert ^ r else 1)
						v = -1
						z = offset
					for w, note in enumerate(taken):
						yield from get_note_block(
							note,
							[x, y + v, z - w],
							odd=pulse & 1,
							ctx=ctx,
						)

	def generate_layer(direction="right", offset=0, elevation=0):
		right = direction == "right"
		x = -1 if right else 1
		y = elevation
		z = offset
		mirrored = offset % 8 < 4
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
			reverse = True
			lower = y < 0
			mid = y == -3
			block = "ochre_froglight" if mid else "verdant_froglight" if lower else "pearlescent_froglight"
			slab = "waxed_cut_copper_slab" if mid else "bamboo_mosaic_slab" if lower else "prismarine_slab"
			edge = "purpur_slab" if mid else "resin_brick_slab" if lower else "dark_prismarine_slab"
			o1 = 3 + reverse
			o2 = 4 - reverse
			for i in range(8):
				yield from (
					((x * (i * 2 + o2), y, z), block),
					((x * (i * 2 + o2 - 1), y - 1, z), slab, dict(type="top")),
					((x * (i * 2 + o2 - 1), y, z), "repeater", dict(facing="east" if right ^ reverse else "west", delay=2)),
					((x * (i * 2 + o1), y + (1 if lower else -1), z + 2), block),
				)
				if i < 7:
					yield ((x * (i * 2 + o1 + 1), y + (0 if lower else -2), z + 2), slab, dict(type="top"))
					yield ((x * (i * 2 + o1 + 1), y + (1 if lower else -1), z + 2), "repeater", dict(facing="west" if right ^ reverse else "east", delay=2))
			x2 = x * 2 if reverse else x * 19 
			yield from (
				((x * (18 if reverse else 3), y, z), "observer", dict(facing="east" if right ^ reverse else "west")),
				((x * (3 if reverse else 18), y + (1 if lower else -1), z + 2), "observer", dict(facing="west" if right ^ reverse else "east")),
				((x2, y + (1 if lower else -1), z + 2), "observer", dict(facing="north")),
				((x2, y - 1, z), edge, dict(type="top")),
				((x2, y + (0 if lower else -2), z + 1), edge, dict(type="top")),
				((x2, y + (1 if lower else -1), z + 1), "powered_rail", dict(shape="north_south" if lower else "ascending_north")),
				((x2, y, z), "powered_rail", dict(shape="ascending_south" if lower else "north_south")),
			)
			if not mirrored:
				x3 = x * 19
				x4 = x * 18
				yield ((x3, y, z), "observer", dict(facing="north"))
				yield ((x3, y - 1, z - 1), edge, dict(type="top"))
				if lower:
					yield from (
						((x4, y + (0 if lower else -2), z - 1), edge, dict(type="top")),
						((x4, y + (1 if lower else -1), z - 1), "powered_rail", dict(shape="east_west")),
						((x3, y, z - 1), "powered_rail", dict(shape="ascending_east" if right else "ascending_west")),
					)
				else:
					yield from (
						((x4, y + (0 if lower else -2), z - 1), edge, dict(type="top")),
						((x4, y + (1 if lower else -1), z - 1), "powered_rail", dict(shape="ascending_west" if right else "ascending_east")),
						((x3, y, z - 1), "powered_rail", dict(shape="east_west")),
					)

	def ensure_layer(direction="right", offset=0):
		right = direction == "right"
		x = -1 if right else 1
		z = offset
		mirrored = offset % 8 < 4
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
		mirrored = offset % 8 < 4
		if mirrored:
			x1, x2 = x * 20, x * 19
		else:
			x1, x2 = x, x * 2
		z2 = z + 1
		opposing = offset % 16 >= 8
		yield from (
			((x1, 1, z2), "glass"),
			((x1, 2, z2), "scaffolding", dict(distance=0)),
			((x2, 1, z2), "bamboo_trapdoor", dict(facing="south", half="top")),
			((x2, 2, z2), "scaffolding", dict(distance=0)),
			((x2, 3, z2), "observer", dict(facing="down")),
			((x2, 4, z2), "redstone_lamp"),
		)
		if right ^ opposing:
			yield from (
				((x2, 5, z2), "redstone_wire", dict(north="up", south="side")),
				((x2, 5, z), "white_stained_glass"),
				((x2, 6, z), "redstone_wire", dict(south="up", north="side")),
				((x2, 6, z2), "white_stained_glass"),
				((x2, 7, z2), "redstone_wire", dict(north="up", south="side")),
				((x2, 7, z), "white_stained_glass"),
				((x2, 8, z), "redstone_wire", dict(south="up", north="side")),
				((x2, 8, z2), "white_stained_glass"),
				((x2, 9, z2), "redstone_wire", dict(north="up", south="side")),
				((x2, 9, z), "oxidized_copper_bulb", dict(lit="false")),
				((x2, 10, z), "redstone_wire", dict(south="up", north="side")),
				((x2, 10, z2), "white_stained_glass"),
				((x2, 11, z2), "redstone_wire", dict(north="up", south="side")),
				((x2, 11, z), "white_stained_glass"),
				((x2, 12, z), "redstone_wire", dict(south="side", north="side")),
			)
		else:
			yield from (
				((x2, 5, z2), "warped_fence_gate", dict(facing="west")),
				((x2, 6, z2), "observer", dict(facing="down")),
				((x2, 6, z), "oxidized_copper_bulb", dict(lit="false")),
				((x2, 7, z2), "mangrove_roots"),
				((x2, 7, z), "redstone_wire", dict(south="up", north="side")),
				((x2, 8, z), "white_stained_glass"),
				((x2, 8, z2), "redstone_wire", dict(north="up", south="side")),
				((x2, 9, z2), "white_stained_glass"),
				((x2, 9, z), "redstone_wire", dict(south="up", north="side")),
				((x2, 10, z), "white_stained_glass"),
				((x2, 10, z2), "redstone_wire", dict(north="up", south="side")),
				((x2, 11, z2), "white_stained_glass"),
				((x2, 11, z), "redstone_wire", dict(south="up", north="side")),
				((x2, 12, z), "oxidized_copper_bulb", dict(lit="false")),
				((x2, 12, z2), "redstone_wire", dict(north="side", south="side")),
			)

	def profile_notes(notes, early=False):
		return (max(map(len, notes[i:(128 if early else 64):4]), default=0) for i in range(4))

	print("Preparing output...")
	bars = ceil(len(notes) / BAR / DIV)
	elevations = (
		((-3, 0), (-3, 1)),
		((6, 1), (9, 1), (12, 1)),
		((-6, 0), (-6, 1), (-9, 0), (-9, 1)),
		((6, 0), (9, 0), (12, 0)),
	)
	for b in (tqdm.trange(bars) if tqdm else range(bars)):
		inverted = not b & 1
		offset = b * 8 + 1
		for i in range(offset, offset + 8):
			yield from (
				((0, -1, i), "tinted_glass"),
				((-1, 0, i), "glass"),
				((1, 0, i), "glass"),
				((0, 0, i), "rail", dict(shape="north_south")),
			)

		def iter_half(inverted=False, ensure=False):
			left, right = ("left", "right") if not inverted else ("right", "left")
			strong, weak1, mid, weak2 = profile_notes(notes, early=ensure)
			yield from generate_layer(right, offset, 0)
			yield from generate_layer(left, offset, 0)
			if weak1:
				if ensure:
					yield from ensure_top(left, offset)
				yield from generate_layer(left, offset, 6)
				if weak1 > SIDE:
					yield from generate_layer(left, offset, 9)
					if weak1 > SIDE * 2:
						yield from generate_layer(left, offset, 12)
			if weak2:
				if ensure:
					yield from ensure_top(right, offset)
				yield from generate_layer(right, offset, 6)
				if weak2 > SIDE:
					yield from generate_layer(right, offset, 9)
					if weak2 > SIDE * 2:
						yield from generate_layer(right, offset, 12)
			if strong > MAIN * 2 or mid or weak2:
				if ensure:
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

		yield from iter_half(inverted, ensure=True)
		yield from extract_notes(notes, offset, -1, inverted, elevations)

		offset += 4
		yield from iter_half(inverted)
		yield from extract_notes(notes, offset, 1, inverted, elevations)

	offset = b * 8 + 8
	yield from (
		((0, -1, 1), "hopper"),
		((0, 0, 1), "powered_rail", dict(shape="north_south")),
		((0, 0, 0), "netherite_block"),
		((0, -1, 0), "honey_block"),
		((0, 1, 0), "calibrated_sculk_sensor", dict(face="floor", facing="south")),
		((1, 0, 0), "oxidized_copper_bulb", dict(lit="false")),
		((-1, 0, 0), "oxidized_copper_bulb", dict(lit="false")),
		((1, -1, 0), "obsidian"),
		((-1, -1, 0), "obsidian"),
		((1, 1, 0), "redstone_wire", dict(north="side", west="side")),
		((-1, 1, 0), "redstone_wire", dict(east="side", west="side")),
		# ((0, 2, 0), "purple_wool"),
		((1, 0, -1), "obsidian"),
		((-1, 0, -1), "obsidian"),
		((0, 1, -1), "obsidian"),
		((2, 1, -1), "composter", dict(level=6)),
		((1, 1, -1), "comparator", dict(facing="east", powered="true")),
		((0, -1, -2), "sticky_piston", dict(facing="south", extended="true")),
		((0, -1, -1), "piston_head", dict(facing="south", short="false", type="sticky")),
		((0, -3, -2), "cobbled_deepslate"),
		((0, -2, -2), "redstone_torch"),
		((1, -4, -2), "cobbled_deepslate"),
		((1, -3, -2), "redstone_wire", dict(east="side", north="side", south="side", west="side")),
		((0, -4, -1), "cobbled_deepslate"),
		((1, -5, -1), "cobbled_deepslate"),
		((1, -4, -1), "repeater", dict(delay=4, facing="south")),
		((1, -4, 0), "observer", dict(facing="south")),
		((1, -4, 1), "observer", dict(facing="west")),
		((1, -6, -2), "cobbled_deepslate"),
		((1, -5, -2), "redstone_wire"),
		((1, -7, -2), "note_block"),
		((0, -7, -2), "observer", dict(facing="east")),
		((0, -7, -1), "observer", dict(facing="north")),
		((0, -7, 0), "sticky_piston", dict(facing="up")),
		((0, -6, 0), "slime_block"),
		((1, -5, 0), "crying_obsidian"),
		((-1, -5, 0), "crying_obsidian"),
		((0, -5, 1), "crying_obsidian"),
		((0, -5, -1), "crying_obsidian"),
		((0, -4, 1), "detector_rail", dict(shape="north_south")),
	)
	for i in range(2, offset):
		if i <= 4 or i >= offset - 3 or not i & 15:
			yield ((0, -4, i), "powered_rail", dict(shape="north_south", powered="true"))
		else:
			yield ((0, -4, i), "rail", dict(shape="north_east"))
		yield ((0, -5, i), "redstone_block")
	yield from (
		((0, -5, offset), "redstone_block"),
		((0, -4, offset), "powered_rail", dict(shape="ascending_south", powered="true")),
		((0, -4, offset + 1), "red_wool"),
		((0, -3, offset + 1), "powered_rail", dict(shape="ascending_south", powered="true")),
		((0, -3, offset + 2), "red_wool"),
		((0, -2, offset + 2), "powered_rail", dict(shape="ascending_south", powered="true")),
		((0, -2, offset + 3), "red_wool"),
		((0, -1, offset + 3), "powered_rail", dict(shape="north_south", powered="true")),
		((0, -1, offset + 4), "red_wool"),
	)
	for x in (-1, 1):
		for n in range(2, 20):
			yield ((x * n, -1, 0), "tinted_glass")
		yield from (
			((x * 2, 0, 0), "yellow_wool"),
			((x * 3, 0, 0), "comparator", dict(facing="east" if x == -1 else "west")),
			((x * 4, 0, 0), "magma_block"),
			((x * 5, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 6, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 7, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 8, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 9, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 10, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 11, 0, 0), "observer", dict(facing="east" if x == -1 else "west")),
			((x * 12, 0, 0), "magma_block"),
			((x * 13, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 14, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 15, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 16, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 17, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 18, 0, 0), "activator_rail", dict(shape="east_west")),
			((x * 19, 0, 0), "activator_rail", dict(shape="east_west")),
		)

def export(transport, ctx=None):
	print("Saving...")
	bars = ceil(len(transport) / BAR / DIV)
	depth = bars * 8 + 8
	nc = 0
	block_replacements = {}
	if ctx.cheap:
		block_replacements.update(dict(
			netherite_block="cobblestone",
			obsidian="cobblestone",
			crying_obsidian="obsidian",
			pearlescent_froglight="cobblestone",
			verdant_froglight="cobblestone",
			ochre_froglight="cobblestone",
			gilded_blackstone="cobblestone",
			sculk="cobblestone",
			polished_blackstone_slab="cobblestone_slab",
			beacon="cobblestone_slab",
			acacia_trapdoor="cobblestone_slab",
			crimson_trapdoor="cobblestone_slab",
			bamboo_mosaic_slab="cobblestone_slab",
			resin_brick_slab="cobblestone_slab",
			prismarine_slab="cobblestone_slab",
			waxed_cut_copper_slab="cobblestone_slab",
			purpur_slab="cobblestone_slab",
			dark_prismarine_slab="cobblestone_slab",
			dark_prismarine="cobblestone",
			prismarine_wall="cobblestone_wall",
			oxidized_copper_trapdoor="bamboo_trapdoor",
			white_stained_glass="glass",
			blue_stained_glass="glass",
			red_stained_glass="glass",
			black_stained_glass="glass",
			tinted_glass="glass",
			mangrove_roots="cobblestone",
			cobbled_deepslate="cobblestone",
			black_wool="white_wool",
			yellow_wool="cobblestone",
			black_concrete_powder="sand",
			heavy_core="sand",
			magma_block="sand",
			amethyst_block="dirt",
			wither_skeleton_skull="air",
			skeleton_skull="air",
			creeper_head="air",
			piglin_head="air",
			zombie_head="air",
		))
	else:
		block_replacements.update(dict(
			hopper="beacon",
		))
	blocks = None
	for output in ctx.output:
		if output.endswith(".nbs"):
			import pynbs
			nbs = pynbs.new_file(
				song_name=output.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0],
				tempo=20,
			)
			nbs.header.song_origin = ctx.input[0].replace("\\", "/").rsplit("/", 1)[-1]
			nbs.header.song_author="Hyperchoron"
			nbs.header.description="Exported MIDI"
			layer_poly = {}
			for i, beat in enumerate(transport):
				current_poly = {}
				beat.sort(key=lambda note_value: (note_value[2], note_value[1]), reverse=True)
				for note in beat:
					ins = note[0]
					base, pitch = get_note_mat(note, transpose=ctx.transpose, odd=i & 1)
					if base == "PLACEHOLDER":
						continue
					instrument = instrument_names[base]
					nbi = nbs_names[instrument]
					try:
						current_poly[ins] += 1
					except KeyError:
						current_poly[ins] = 1
					rendered = pynbs.Note(
						tick=i,
						layer=ins,
						key=pitch + 33,
						instrument=nbi,
						velocity=round(note[4] / 127 * 100),
						panning=0 if note[2] else 1 if i & 1 else -1,
					)
					nbs.notes.append(rendered)
				for k, v in current_poly.items():
					layer_poly[k] = max(v, layer_poly.get(k, 0))
			layer_map = sorted(layer_poly.items(), key=lambda tup: (tup[0] not in (-1, 8), tup[0] != 6, tup[-1]), reverse=True)
			layer_index = 0
			layer_starts = {}
			for ins, poly in layer_map:
				layer_starts[ins] = layer_index
				for i in range(poly):
					idx = layer_index + i
					name = instrument_codelist[ins]
					if i:
						name += f"_{i}"
					layer = pynbs.Layer(
						id=idx,
						name=name,
					)
					try:
						nbs.layers[idx] = layer
					except IndexError:
						nbs.layers.append(layer)
				layer_index += poly
			current_tick = 0
			used_layers = {}
			for note in nbs.notes:
				t = note.tick
				if t > current_tick:
					current_tick = t
					used_layers.clear()
				ins = note.layer
				layer = layer_starts[ins] + used_layers.setdefault(ins, 0)
				used_layers[ins] += 1
				note.layer = layer
			nbs.notes.sort(key=lambda note: (note.tick, note.layer))
			nc = len(nbs.notes)
			nbs.save(output)
		elif output.endswith(".mcfunction"):
			if blocks is None:
				blocks = list(render_minecraft(list(transport), ctx=ctx))
			with open(output, "w") as f:
				for (x, y, z), block, *kwargs in blocks:
					if block in block_replacements:
						block = block_replacements[block]
						if block == "cobblestone_slab":
							kwargs = [dict(type="top")]
					if block == "sand":
						f.write(f"setblock ~{x} ~{y - 1} ~{z} dirt keep\n")
					nc += block == "note_block"
					if kwargs:
						extra = "[" + ",".join(f"{k}={v}" for k, v in kwargs[0].items()) + "]"
					else:
						extra = ""
					f.write(f"setblock ~{x} ~{y} ~{z} {block}{extra}\n")
		else:
			if blocks is None:
				blocks = list(render_minecraft(list(transport), ctx=ctx))
			import litemapy
			air = litemapy.BlockState("minecraft:air")
			mx, my, mz = 20, 13, 2
			reg = litemapy.Region(-mx, -my, -mz, mx * 2 + 1, my * 2 + 1, depth + mz)
			schem = reg.as_schematic(
				name=output.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0],
				author="Hyperchoron",
				description="Exported MIDI",
			)
			for (x, y, z), block, *kwargs in blocks:
				if block in block_replacements:
					block = block_replacements[block]
					if block == "cobblestone_slab":
						kwargs = [dict(type="top")]
				if block == "sand" and reg[x + mx, y + my - 1, z + mz] == air:
					reg[x + mx, y + my - 1, z + mz] = litemapy.BlockState("minecraft:dirt")
				nc += block == "note_block"
				if kwargs:
					block = litemapy.BlockState("minecraft:" + block, **{k: str(v) for k, v in kwargs[0].items()})
				else:
					block = litemapy.BlockState("minecraft:" + block)
				reg[x + mx, y + my, z + mz] = block
			schem.save(output)
	print("Final note block count:", nc)

def convert_file(args):
	ctx = args
	inputs = list(ctx.input)
	if not ctx.output:
		path, name = inputs[0].replace("\\", "/").rsplit("/", 1)
		ctx.output = [path + "/" + name.rsplit(".", 1)[0] + ".litematic"]
	if inputs[0].endswith(".zip"):
		import zipfile
		z = zipfile.ZipFile(inputs.pop(0))
		inputs.extend(z.open(f) for f in z.filelist)
	event_list = []
	transport = []
	note_candidates = 0
	for file in inputs:
		if isinstance(file, str) and file.endswith(".nbs"):
			print("Converting NBS...")
			import pynbs
			nbs = pynbs.read(file)
			for tick, chord in nbs:
				while tick > len(transport):
					transport.append([])
				mapped_chord = []
				for note in chord:
					instrument_name = nbs.layers[note.layer].name.rsplit("_", 1)[0]
					if instrument_name == "Drumset":
						ins = 6
						default = 6
					else:
						default = instrument_codelist.index(default_instruments[nbs_values[note.instrument]])
						try:
							ins = instrument_codelist.index(instrument_name)
						except ValueError:
							ins = default
					block = (
						ins,
						note.key - 33 + fs1 + pitches[nbs_values[note.instrument]],
						note.panning == 0,
						ins != default,
						round(note.velocity * 127 / 100),
					)
					mapped_chord.append(block)
				transport.append(mapped_chord)
			continue
		print("Converting MIDI...")
		if py_midicsv:
			csv_list = py_midicsv.midi_to_csv(file)
		else:
			import subprocess
			if isinstance(file, str):
				csv_list = subprocess.check_output(["Midicsv.exe", file, "-"]).decode("utf-8", "replace").splitlines()
			else:
				p = subprocess.Popen(["Midicsv.exe", "-", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
				b = file.read()
				csv_list = p.communicate(b)[0].decode("utf-8", "replace").splitlines()
		midi_events = list(csv.reader(csv_list))
		if not isinstance(file, str):
			file.close()
		event_list.append(midi_events)
	if event_list:
		all_events = list(itertools.chain.from_iterable(event_list))
		all_events.sort(key=lambda e: int(e[1]))
		speed_info = get_step_speed(all_events, tps=20 / ctx.speed, ctx=ctx)
		for midi_events in event_list:
			notes, nc, is_org, speed_info = convert_midi(midi_events, speed_info, ctx=ctx)
			note_candidates += nc
			if len(inputs) == 1:
				transport = notes
				break
			for i, beat in enumerate(notes):
				if len(transport) <= i:
					transport.append(beat)
				else:
					transport[i].extend(beat)
		if is_org:
			ctx.transpose += 12
	else:
		speed_info = (50, 1, 50, 60 * 120)
	if transport and not transport[0]:
		transport = deque(transport)
		while transport and not transport[0]:
			transport.popleft()
		transport = list(transport)
	maxima = [(sum(map(len, transport[i::4])), i) for i in range(4)]
	strongest_beat = max(maxima)[1]
	if strongest_beat != 0:
		buffer = [[]] * (4 - strongest_beat)
		transport = buffer + transport
	print("Note candidates:", note_candidates)
	print("Note count:", sum(map(len, transport)))
	print("Max detected polyphony (will be reduced to <=14 in schematics):", max(map(len, transport), default=0))
	print("Lowest note:", min(min(n[1] for n in b) for b in transport if b))
	print("Highest note:", max(max(n[1] for n in b) for b in transport if b))
	export(transport, ctx=ctx)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		prog="",
		description="MIDI to Minecraft Note Block Converter",
	)
	parser.add_argument("-i", "--input", nargs="+", help="Input file (.mid | .zip | .nbs)")
	parser.add_argument("-o", "--output", nargs="*", help="Output file (.mcfunction | .litematic | .nbs)")
	parser.add_argument("-t", "--transpose", nargs="?", type=int, default=0, help="Transposes song up/down a certain amount of semitones, applied before instrument material mapping; higher = higher pitched")
	parser.add_argument("-s", "--speed", nargs="?", type=float, default=1, help="Scales song speed up/down as a multiplier, applied before tempo sync; higher = faster")
	parser.add_argument("-sa", "--strum-affinity", nargs="?", default=1, type=float, help="Increases or decreases threshold for sustained notes to be cut into discrete segments; higher = more notes")
	parser.add_argument("-d", "--drums", action=argparse.BooleanOptionalAction, default=True, help="Allows percussion channel. If disabled, the default MIDI percussion channel will be treated as a regular instrument channel. Defaults to TRUE")
	parser.add_argument("-c", "--cheap", action=argparse.BooleanOptionalAction, default=False, help="Restricts the list of non-instrument blocks to a more survival-friendly set. Also enables compatibility with previous versions of minecraft. May cause spacing issues with the sand/snare drum instruments. Defaults to FALSE")
	args = parser.parse_args()
	convert_file(args)
