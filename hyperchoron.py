# Coco eats a gold block on the 18/2/2025. Nom nom nom nom. Output sound weird? Sorgy accident it was the block I eated. Rarararrarrrr 😋

from collections import deque
import csv
import fractions
import itertools
from math import trunc, ceil, inf, isqrt, sqrt, log2, gcd
from types import SimpleNamespace
try:
	import tqdm
except ImportError:
	import contextlib
	tqdm = None
else:
	import warnings
	warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
import rendering
from mappings import (
	material_map, default_instruments, instrument_codelist, instrument_names,
	nbs_names, nbs_values, sustain_map, pitches,
	instrument_mapping, org_instrument_mapping,
	DIV, BAR, fs1,
)


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
			new_tempo = int(event[3])
			if orig_tempo and orig_tempo != new_tempo:
				tempos[orig_tempo] = tempos.get(orig_tempo, 0) + timestamp - since_tempo
			since_tempo = timestamp
			orig_tempo = new_tempo
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
		print("No time signature found! Defaulting to 4/4...")
	if not milliseconds_per_clock:
		orig_tempo = orig_tempo or 500 * 1000
		milliseconds_per_clock = orig_tempo / 1000 / clocks_per_crotchet # Default to 120 BPM
		print("No BPM found! Defaulting to 120...")
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
				speed = 0.4 if ctx.exclusive else 1
			else:
				speed = speed2 / 2 if ctx.exclusive else speed2
			use_exact = True
		elif speed > min_value * 1.25:
			print("Finding closest speed...", exclusions, len(timestamps))
			div = round(speed / min_value - 0.25)
			if div <= 1:
				if not ctx.exclusive:
					if speed % 3 == 0:
						print("Speed too close for rounding, autoscaling by 2/3...")
						speed *= 2
						speed //= 3
					else:
						print("Speed too close for rounding, autoscaling by 75%...")
						speed *= 0.75
				else:
					speed //= 2
			else:
				speed /= div
	if use_exact:
		real_ms_per_clock = milliseconds_per_clock
	else:
		real_ms_per_clock = round(milliseconds_per_clock * min_value / step_ms) * step_ms
	print("Final speed scale:", speed, real_ms_per_clock)
	return milliseconds_per_clock, real_ms_per_clock, speed, step_ms, orig_tempo

def preprocess(midi_events, ctx):
	title = None
	copyright = None
	is_org = False
	midi_events = [(int(e[0]), int(e[1]), e[2].strip().casefold(), *e[3:]) for e in midi_events]
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
	instrument_map = {}
	channel_stats = {}
	note_lengths = {}
	temp_active = {}
	discard = set()
	midi_events.sort(key=lambda x: (x[1], x[2] not in ("tempo", "header", "control_c"))) # Sort events by timestamp, keep headers first
	max_vel = 0
	last_timestamp = 0
	for i, e in enumerate(midi_events):
		m = e[2]
		match m:
			case "program_c":
				c = int(e[3])
				if c == 9 and ctx.drums:
					continue
				value = int(e[4])
				instrument_map[c] = org_instrument_mapping[value] if is_org else instrument_mapping[value]
			case "note_on_c":
				ti = e[0]
				t = e[1]
				c = int(e[3])
				p = int(e[4])
				h = (ti, c, p)
				velocity = int(e[5])
				if c not in instrument_map:
					instrument_map[c] = -1 if c == 9 and ctx.drums else 0
				volume = velocity * channel_stats.setdefault(c, {}).get("volume", 1)
				max_vel = max(max_vel, volume)
				if velocity <= 1:
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
					channel_stats.setdefault(c, {})["volume"] = volume
					if c not in instrument_map:
						instrument_map[c] = -1 if c == 9 and ctx.drums else 0
	for k, v in temp_active.items():
		for t in v:
			h2 = (t, *k)
			note_lengths.setdefault(h2, 1)
	print("Instrument mapping:", instrument_map)
	print("Max volume:", max_vel)
	midi_events = [e for i, e in enumerate(midi_events) if i not in discard]
	return midi_events, instrument_map, channel_stats, note_lengths, max_vel, last_timestamp, is_org

def convert_midi(midi_events, speed_info, ctx=None):
	played_notes = []
	pitchbend_ranges = {}
	max_pitch = 101 if ctx.exclusive else 84
	active_notes = {i: [] for i in range(len(material_map))}
	active_notes[-1] = []
	instrument_activities = {}
	timestamp = 0
	loud = 0
	note_candidates = 0
	_orig_ms_per_clock, real_ms_per_clock, scale, orig_step_ms, orig_tempo = speed_info
	step_ms = orig_step_ms
	midi_events, instrument_map, channel_stats, note_lengths, max_vel, last_timestamp, is_org = preprocess(midi_events, ctx=ctx)
	print("Processing notes...")
	progress = tqdm.tqdm(total=ceil(last_timestamp * real_ms_per_clock / scale / 1000), bar_format="{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") if tqdm else contextlib.nullcontext()
	global_index = 0
	curr_frac = round(step_ms)
	with progress as bar:
		while global_index < len(midi_events):
			event = midi_events[global_index]
			event_timestamp = event[1]
			if event_timestamp > last_timestamp:
				break
			curr_step = step_ms
			time = event_timestamp * real_ms_per_clock / scale
			if time - curr_step / 2 < timestamp:
				# Process all events at the current timestamp
				mode = event[2]
				match mode:
					case "note_on_c":
						channel = int(event[3])
						instrument = instrument_map[channel]
						pitch = int(event[4])
						velocity = int(event[5])
						if velocity == 0:
							pass
						elif velocity < loud * 0.0625 and sum(map(len, active_notes.items())) >= 16:
							note_candidates += 1
						else:
							note_candidates += 1
							if instrument in (-1, 6, 8):
								sustain = 0
							else:
								sustain = sustain_map[instrument] or (1 if is_org else 2 if ctx.exclusive else 0)
							length = 0
							if sustain or pitchbend_ranges.get(channel):
								track = int(event[0])
								h = (event_timestamp, track, channel, pitch)
								try:
									length = (note_lengths[h] + 0.25) * real_ms_per_clock / scale
								except KeyError:
									length = 0
							min_sustain = curr_frac * 2.25 if sustain == 2 else curr_frac * 1.25
							if length < min_sustain:
								length = min_sustain
								if sustain == 1:
									sustain = 0
							note = SimpleNamespace(
								pitch=pitch,
								velocity=velocity,
								start=timestamp,
								timestamp=timestamp,
								length=length,
								channel=channel,
								sustain=sustain,
								updated=2,
								volume=0,
							)
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
									note.updated = max(note.updated, 1)
									if not candidate or note.start > candidate.start:
										candidate = note
							if candidate:
								candidate.length = max(candidate.length, float(timestamp + curr_frac - candidate.start))
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
									note.updated = max(note.updated, 1)
						elif volume <= orig_volume * 0.5:
							instrument = instrument_map[channel]
							for note in active_notes[instrument]:
								if note.channel == channel:
									note.sustain = False
						channel_stats[channel]["volume"] = volume
					case "control_c" if event[4].strip() == "10":
						channel = int(event[3])
						value = int(event[5])
						pan = max(-1, (value - 64) / 63)
						channel_stats[channel]["pan"] = pan
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
						curr_frac = step_ms if step_ms.is_integer() else float(step_ms)
						if not note_candidates:
							timestamp = round(time)
				global_index += 1
			else:
				started = {}
				ticked = {}
				max_volume = 0
				poly = 0
				for notes in active_notes.values():
					for note in notes:
						note.volume = channel_stats.get(note.channel, {}).get("volume", 1) * note.velocity / max_vel * 127
						if note.volume > max_volume:
							max_volume = note.volume
						poly += note.sustain
				for instrument, notes in active_notes.items():
					notes.reverse()
					for i in range(len(notes) - 1, -1, -1):
						note = notes[i]
						volume = note.volume
						length = note.length
						end = note.start + note.length
						sa = ctx.strum_affinity
						sms = curr_frac
						long = note.sustain and length > sms * 3 / sa and volume >= min(80, max_volume) / sa
						needs_sustain = note.sustain and length > sms * 4 / sa and (length > sms * 8 / sa or long and (volume >= 120 / sa or note.sustain == 1 and length >= sms * 6 / sa))
						recur = inf
						if ctx.exclusive and needs_sustain:
							recur = sms
						elif needs_sustain:
							if volume >= 100 / sa and (poly <= 4 and volume >= max_volume * 7 / 8 / sa or volume >= max_volume * 127 / 128 / sa):
								recur = sms
							elif volume >= 60 / sa and volume >= max_volume / 2 / sa:
								recur = sms * 2
							elif volume >= 20 / sa and volume >= max_volume / 4 / sa:
								recur = sms * 4
							else:
								recur = sms * 8
							if recur < inf and length >= sms * 4:
								h = recur
								n = started.setdefault(h, 0)
								if note.updated:
									if recur == sms * 4:
										offset = bool(n & 1) * sms * 2 + bool(n & 2) * sms
									elif recur == sms * 2:
										offset = bool(n & 1) * sms
									else:
										offset = 0
									recur += offset
								started[h] += 1
						if note.updated or (timestamp >= note.timestamp and timestamp + recur <= end):
							pitch = channel_stats.get(note.channel, {}).get("bend", 0) + note.pitch
							normalised = pitch + ctx.transpose - fs1
							if normalised > max_pitch:
								pitch = max_pitch - ctx.transpose + fs1
							elif normalised < -12:
								pitch = -12 - ctx.transpose + fs1
							bucket = (instrument, pitch)
							try:
								temp = ticked[bucket]
							except KeyError:
								temp = ticked[bucket] = [note.updated, long, volume ** 2, channel_stats.get(note.channel, {}).get("pan", 0)]
							else:
								temp[0] = max(temp[0], note.updated)
								temp[1] |= long
								temp[3] = temp[3] if temp[2] > volume ** 2 else channel_stats.get(note.channel, {}).get("pan", 0)
								temp[2] = temp[2] + volume ** 2
							note.timestamp = timestamp + recur - sms / 4
						if timestamp >= end or len(notes) >= 64 and not needs_sustain:
							notes.pop(i)
						else:
							note.updated = 0
							if note.sustain == 2 and (length < sms * 3 or (timestamp - note.start) % (sms * 2) >= sms):
								note.velocity = max(2, note.velocity - sms / (length + sms) * note.velocity * 2)
					notes.reverse()
				beat = []
				poly = {}
				for k, v in ticked.items():
					instrument, pitch = k
					updated, long, volume, pan = v
					volume = sqrt(volume)
					count = max(1, int(volume // 127))
					vel = max(1, min(127, round(volume / count)))
					block = (instrument, pitch, updated, long, vel, pan)
					for w in range(count):
						beat.append(block)
					try:
						poly[instrument] += count
					except KeyError:
						poly[instrument] = count
					try:
						instrument_activities[instrument][0] += volume
						instrument_activities[instrument][1] = max(instrument_activities[instrument][1], poly[instrument])
					except KeyError:
						instrument_activities[instrument] = [volume, poly[instrument]]
				played_notes.append(beat)
				timestamp += curr_step
				if timestamp.is_integer():
					timestamp = int(timestamp)
				if bar:
					bar.update(curr_step / 1000)
				loud *= 0.5
	while not played_notes[-1]:
		played_notes.pop(-1)
	return played_notes, note_candidates, is_org, instrument_activities, speed_info

def export(transport, instrument_activities, speed_info, ctx=None):
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
		out_name = output.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
		if output.endswith(".nbs"):
			import pynbs
			nbs = pynbs.new_file(
				song_name=out_name,
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
					base, pitch = rendering.get_note_mat(note, transpose=ctx.transpose, odd=i & 1)
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
						panning=trunc(note[5] * 49) * 2 + (0 if note[2] else 1 if i & 1 else -1),
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
		elif output.endswith(".csv") or output.endswith(".mid") or output.endswith(".midi"):
			instruments, wait = list(rendering.render_midi(list(transport), instrument_activities, speed_info, ctx=ctx))
			import io
			b = io.StringIO() if output.endswith(".mid") else open(output, "w", newline="", encoding="utf-8")
			with b:
				writer = csv.writer(b)
				writer.writerows([
					[0, 0, "header", 1, len(instruments) + 1, 8],
					[1, 0, "start_track"],
					[1, 0, "title_t", out_name],
					[1, 0, "copyright_t", "Hyperchoron"],
					[1, 0, "text_t", "Exported MIDI"],
					[1, 0, "time_signature", 4, 4, 8, 8],
					[1, 0, "tempo", wait],
					[1, 0, "end_track"],
				])
				for i, ins in enumerate(instruments, 2):
					notes = deque(ins.notes)
					nc += len(notes)
					start = 0
					writer.writerows([
						[i, start, "start_track"],
						[i, start, "title_t", ins.name],
						[i, start, "program_c", ins.channel, ins.id if ins.id >= 0 else 0],
						[i, start, "control_c", ins.channel, 10, 64],
						[i, start, "control_c", ins.channel, 7, 100],
					])
					pan = 64
					vol = 100
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
								case _:
									raise NotImplementedError(mode)
							instrument.append([i, tick, "control_c", ins.channel, mode_i, value])
						end = max(end, note.tick + note.length)
					instrument.sort(key=lambda t: (t[1], t[2] == "note_on_c"))
					instrument.append([i, end, "end_track"])
					writer.writerows(instrument)
				writer.writerows([[0, 0, "end_of_file"]])
				if output.endswith(".mid"):
					b.seek(0)
					rendering.csv2midi(b, output)
		elif output.endswith(".org"):
			import struct
			instruments, wait = list(rendering.render_org(list(transport), instrument_activities, speed_info, ctx=ctx))
			with open(output, "wb") as org:
				org.write(b"\x4f\x72\x67\x2d\x30\x32")
				org.write(struct.pack("<H", wait))
				org.write(b"\x04\x08")
				org.write(struct.pack("<L", 0))
				org.write(struct.pack("<L", ceil(len(transport) / 4) * 4))
				for i, ins in enumerate(instruments):
					org.write(struct.pack("<H", 1000 + i * (50 if i & 1 else -50)))
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
		elif output.endswith(".mcfunction"):
			if blocks is None:
				blocks = list(rendering.render_minecraft(list(transport), ctx=ctx))
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
				blocks = list(rendering.render_minecraft(list(transport), ctx=ctx))
			import litemapy
			air = litemapy.BlockState("minecraft:air")
			mx, my, mz = 20, 13, 2
			reg = litemapy.Region(-mx, -my, -mz, mx * 2 + 1, my * 2 + 1, depth + mz)
			schem = reg.as_schematic(
				name=out_name,
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
	print("Final note count:", nc)

def convert_file(args):
	ctx = args
	if ctx.output and (ctx.output[0].rsplit(".", 1)[-1] in ("org", "csv", "mid", "midi")):
		ctx.strum_affinity = inf
		if ctx.exclusive is None:
			print("Auto-switching to Exclusive mode...")
			ctx.exclusive = True
	inputs = list(ctx.input)
	if not ctx.output or not any("." in fn for fn in ctx.output):
		*path, name = inputs[0].replace("\\", "/").rsplit("/", 1)
		ext = ctx.output[0] if ctx.output else "litematic"
		ctx.output = [("".join(path) + "/" if path else "") + name.rsplit(".", 1)[0] + "." + ext]
	print(ctx.output)
	if inputs[0].endswith(".zip"):
		import zipfile
		z = zipfile.ZipFile(inputs.pop(0))
		inputs.extend(z.open(f) for f in z.filelist)
	event_list = []
	transport = []
	instrument_activities = {}
	speed_info = None
	note_candidates = 0
	for file in inputs:
		if isinstance(file, str):
			if file.endswith(".nbs"):
				print("Converting NBS...")
				import pynbs
				nbs = pynbs.read(file)
				speed_info = (50, 50, 20 / nbs.header.tempo, 50, 500)
				for tick, chord in nbs:
					while tick > len(transport):
						transport.append([])
					mapped_chord = []
					poly = {}
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
							not note.panning & 1,
							ins != default,
							round(note.velocity * 127 / 100),
							note.panning / 50,
						)
						try:
							poly[ins] += 1
						except KeyError:
							poly[ins] = 1
						try:
							instrument_activities[ins][0] += note.velocity
							instrument_activities[ins][1] = max(instrument_activities[ins][1], poly[ins])
						except KeyError:
							instrument_activities[ins] = [note.velocity, poly[ins]]
						mapped_chord.append(block)
					transport.append(mapped_chord)
				continue
			if file.endswith(".csv"):
				with open(file, "r", encoding="utf-8") as f:
					csv_list = f.read().splitlines()
				midi_events = list(csv.reader(csv_list))
				event_list.append(midi_events)
				continue
		print("Converting MIDI...")
		csv_list = rendering.midi2csv(file)
		midi_events = list(csv.reader(csv_list))
		if not isinstance(file, str):
			file.close()
		event_list.append(midi_events)
	if event_list:
		if len(event_list) > 1:
			all_events = list(itertools.chain.from_iterable(event_list))
		else:
			all_events = event_list[0]
		all_events.sort(key=lambda e: int(e[1]))
		speed_info = get_step_speed(all_events, tps=20 / ctx.speed, ctx=ctx)
		for midi_events in event_list:
			notes, nc, is_org, instrument_activities2, speed_info = convert_midi(midi_events, speed_info, ctx=ctx)
			for k, v in instrument_activities2.items():
				if k in instrument_activities:
					instrument_activities[k][0] += v[0]
					instrument_activities[k][1] += v[1]
				else:
					instrument_activities[k] = v
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
	if not speed_info:
		speed_info = (50, 50, 1, 50, 500)
	if transport and not transport[0]:
		transport = deque(transport)
		while transport and not transport[0]:
			transport.popleft()
		transport = list(transport)
	maxima = [(sum(sum(note[2] + 1 for note in beat) for beat in transport[i::4]), i) for i in range(4)]
	strongest_beat = max(maxima)[1]
	if strongest_beat != 0:
		buffer = [[]] * (4 - strongest_beat)
		transport = buffer + transport
	print("Note candidates:", note_candidates)
	print("Note count:", sum(map(len, transport)))
	print("Max detected polyphony (will be reduced to <=14 in schematics):", max(map(len, transport), default=0))
	print("Lowest note:", min(min(n[1] for n in b) for b in transport if b))
	print("Highest note:", max(max(n[1] for n in b) for b in transport if b))
	export(transport, instrument_activities, speed_info, ctx=ctx)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		prog="",
		description="MIDI converter and Minecraft Note Block exporter",
	)
	parser.add_argument("-i", "--input", nargs="+", help="Input file (.mid | .zip | .nbs | .csv)")
	parser.add_argument("-o", "--output", nargs="*", help="Output file (.mcfunction | .litematic | .nbs | .org | .csv | .mid)")
	parser.add_argument("-t", "--transpose", nargs="?", type=int, default=0, help="Transposes song up/down a certain amount of semitones, applied before instrument material mapping; higher = higher pitched")
	parser.add_argument("-s", "--speed", nargs="?", type=float, default=1, help="Scales song speed up/down as a multiplier, applied before tempo sync; higher = faster")
	parser.add_argument("-sa", "--strum-affinity", nargs="?", default=1, type=float, help="Increases or decreases threshold for sustained notes to be cut into discrete segments; higher = more notes")
	parser.add_argument("-d", "--drums", action=argparse.BooleanOptionalAction, default=True, help="Allows percussion channel. If disabled, the default MIDI percussion channel will be treated as a regular instrument channel. Defaults to TRUE")
	parser.add_argument("-c", "--cheap", action=argparse.BooleanOptionalAction, default=False, help="Restricts the list of non-instrument blocks to a more survival-friendly set. Also enables compatibility with previous versions of minecraft. May cause spacing issues with the sand/snare drum instruments. Defaults to FALSE")
	parser.add_argument("-x", "--exclusive", action=argparse.BooleanOptionalAction, default=None, help="Disables speed re-matching and strum quantisation, increases pitch bucket limit. Defaults to FALSE if outputting to any Minecraft-related format, and included for compatibility with other export formats.")
	args = parser.parse_args()
	convert_file(args)
