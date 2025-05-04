from collections import deque
import itertools
from math import ceil, isqrt, sqrt, log2, gcd
from .mappings import note_names

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
	for x in set(map(abs, non_zero)):
		# Find all divisors of x
		for i in range(1, int(isqrt(x)) + 1):
			if x % i == 0:
				if i >= min_value:
					divisors.add(i)
				counterpart = x // i
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
		count = sum(x % d == 0 for x in arr)
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

def sync_tempo(timestamps, milliseconds_per_clock, clocks_per_crotchet, tps, orig_tempo, fine=False, ctx=None):
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
			req = 0 if not ctx.mc_legal else (max(speed2, 1 / speed2) - 1) * sqrt(2)
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
				if ctx.mc_legal:
					if speed % 3 == 0:
						print("Speed too close for rounding, autoscaling by 2/3...")
						speed *= 2
						speed //= 3
					else:
						print("Speed too close for rounding, autoscaling by 75%...")
						speed *= 0.75
				elif speed > 1:
					speed //= 2
			elif fine or ctx.mc_legal:
				speed /= div
	while speed > 50:
		speed /= 2
	if use_exact:
		real_ms_per_clock = milliseconds_per_clock
	else:
		real_ms_per_clock = round(milliseconds_per_clock * min_value / step_ms) * step_ms
	print("Final speed scale:", milliseconds_per_clock, real_ms_per_clock, speed, step_ms, orig_tempo)
	return milliseconds_per_clock, real_ms_per_clock, speed, step_ms, orig_tempo

def create_reader(file):
	def read(offset, length, decode=True):
		file.seek(offset)
		b = file.read(length)
		if not decode:
			return b
		return int.from_bytes(b, "little")
	return read

def transport_note_priority(n, sustained=False, multiplier=8):
	return n[2] + round(n[4] * multiplier / 127) + sustained

def merge_activities(a1, a2):
	if not a1:
		a1.update(a2)
	else:
		for k, v in a2.items():
			if k in a1:
				a1[k][0] += v[0]
				a1[k][1] += v[1]
			else:
				a1[k] = v
	return a1

def merge_imports(inputs, ctx):
	event_list = []
	transport = []
	instrument_activities = {}
	speed_info = None
	note_candidates = 0
	for data in inputs:
		if isinstance(data, list):
			event_list.append(data)
		else:
			for i, x in enumerate(data.transport):
				if i >= len(transport):
					transport.append(x)
				else:
					transport[i].extend(x)
			merge_activities(instrument_activities, data.instrument_activities)
			speed_info = speed_info or data.speed_info
	if event_list:
		from . import midi
		if len(event_list) > 1:
			all_events = list(itertools.chain.from_iterable(event_list))
		else:
			all_events = event_list[0]
		all_events.sort(key=lambda e: int(e[1]))
		speed_info = midi.get_step_speed(all_events, ctx=ctx)
		for midi_events in event_list:
			notes, nc, is_org, instrument_activities2, speed_info = midi.deconstruct(midi_events, speed_info, ctx=ctx)
			merge_activities(instrument_activities, instrument_activities2)
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
	return transport, instrument_activities, speed_info, note_candidates

def transpose(transport, ctx):
	mapping = {}
	if ctx.invert_key:
		notes = [0] * 12
		for beat in transport:
			for note in beat:
				p = note[1] % 12
				notes[p] += 1
		major_scores = [0] * 12
		minor_scores = [0] * 12
		for key in range(12):
			score = 0
			for scale in (0, 0, 2, 4, 4, 5, 7, 7, 9, 11):
				score += notes[(key + scale) % 12]
			for scale in (1, 3, 8):
				score -= notes[(key + scale) % 12]
			major_scores[key] = score
			score = 0
			for scale in (0, 0, 2, 3, 3, 5, 7, 7, 8, 11):
				score += notes[(key + scale) % 12]
			for scale in (4, 4, 6):
				score -= notes[(key + scale) % 12]
			minor_scores[key] = score
		candidates = list(enumerate(major_scores + minor_scores))
		candidates.sort(key=lambda t: t[1], reverse=True)
		key = candidates[0][0]
		key_repr = f"{note_names[key]} Major" if key < 12 else f"{note_names[key - 12]} Minor"
		print("Detected key signature:", key_repr)
		inv_key = key + 12 if key < 12 else key - 12
		inv_repr = f"{note_names[inv_key]} Major" if inv_key < 12 else f"{note_names[inv_key - 12]} Minor"
		print("Inverse key:", inv_repr)
		if key < 12:
			mapping[(10 + key) % 12] = 1
			mapping[(4 + key) % 12] = -1
			mapping[(9 + key) % 12] = -1
			mapping[(11 + key) % 12] = -1
		else:
			mapping[(3 + key) % 12] = 1
			mapping[(8 + key) % 12] = 1
			mapping[(10 + key) % 12] = 1
	if ctx.invert_key or ctx.transpose:
		for beat in transport:
			for i, note in enumerate(beat):
				pitch = note[1]
				p = pitch % 12
				adj = ctx.transpose + mapping.get(p, 0)
				if adj:
					pitch += adj
					note = (note[0], pitch, *note[2:])
					beat[i] = note
	maxima = [(sum(sum(map(transport_note_priority, beat)) for beat in transport[i::4]), i) for i in range(4)]
	strongest_beat = max(maxima)[1]
	if strongest_beat != 0:
		buffer = [[]] * (4 - strongest_beat)
		transport = buffer + transport