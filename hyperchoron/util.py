from collections import deque, namedtuple
import csv
import datetime
import fractions
import itertools
from math import ceil, isqrt, sqrt, log2, log10, gcd, lcm
import os
from .mappings import note_names

sample_rate = 44100
DEFAULT_NAME = "Hyperchoron"
DEFAULT_DESCRIPTION = f"Exported by Hyperchoron on {datetime.datetime.now().date()}"
base_path = __file__.replace("\\", "/").rsplit("/", 1)[0] + "/"
temp_dir = os.path.abspath(base_path.rsplit("/", 2)[0]).replace("\\", "/").rstrip("/") + "/temp/"
if not os.path.exists(temp_dir):
	os.mkdir(temp_dir)
csv_reader = type(csv.reader([]))

def round_min(x):
	try:
		y = int(x)
	except (ValueError, OverflowError, TypeError):
		return x
	if x == y:
		return y
	return x

def log2lin(x, min_db=-40):
	if x <= 0:
		return 0
	return 10 ** ((x - 1) * -min_db / 20)

def lin2log(y, min_db=-40):
	if y <= 0:
		return 0
	return 1 + 20 * log10(y) / (-min_db)

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
	# print("Estimating true resolution...", len(timestamp_collection), clocks_per_crotchet, milliseconds_per_clock, step_ms, min_value)
	speed, exclusions = approximate_gcd(timestamp_collection, min_value=min_value * 3 / 4)
	use_exact = False
	req = 1 / 8
	# print("Confidence:", 1 - exclusions / len(timestamp_collection), req, speed)
	if speed > min_value * 1.25 or exclusions > len(timestamp_collection) * req:
		if exclusions >= len(timestamp_collection) * 0.75:
			speed2 = milliseconds_per_clock / step_ms * clocks_per_crotchet
			while speed2 > sqrt(2):
				speed2 /= 2
			# print("Rejecting first estimate:", speed, speed2, min_value, exclusions)
			step = 1 / speed2
			inclusions = sum((res := x % step) < step / 12 or res > step - step / 12 or (res := x % (step * 4)) < (step * 4) / 12 or res > (step * 4) - (step * 4) / 12 for x in timestamp_collection)
			req = 0 if not ctx.mc_legal else (max(speed2, 1 / speed2) - 1) * sqrt(2)
			# print("Confidence:", inclusions / len(timestamp_collection), req)
			if inclusions < len(timestamp_collection) * req:
				print("Discarding tempo...")
				speed = 1
				use_exact = True
			elif not ctx.mc_legal:
				speed = speed2
				use_exact = True
		elif speed > min_value * 1.25:
			# print("Finding closest speed...", exclusions, len(timestamps))
			div = round(speed / min_value - 0.25)
			if div <= 1:
				if ctx.mc_legal:
					if speed % 3 == 0:
						# print("Speed too close for rounding, autoscaling by 2/3...")
						speed *= 2
						speed //= 3
					else:
						# print("Speed too close for rounding, autoscaling by 75%...")
						speed *= 0.75
				elif speed > 1:
					speed //= 2
			elif fine or ctx.mc_legal:
				speed /= div
			elif div > 2:
				speed /= div / 2
	if not use_exact and ctx.mc_legal and (speed < min_value * 0.85 or speed > min_value * 1.15):
		frac = fractions.Fraction(min_value / speed).limit_denominator(5)
		print("For MC compliance: rescaling speed by:", frac)
		speed = float(speed * frac)
	if use_exact:
		real_ms_per_clock = milliseconds_per_clock
	else:
		real_ms_per_clock = round(milliseconds_per_clock * min_value / step_ms) * step_ms
	print("Detected speed scale:", milliseconds_per_clock, real_ms_per_clock, speed, step_ms, orig_tempo)
	return milliseconds_per_clock, real_ms_per_clock, speed, step_ms, orig_tempo

def estimate_filesize(file):
	if isinstance(file, (list, tuple, deque)):
		return len(file)
	if isinstance(file, str):
		return os.path.getsize(file)
	orig = file.tell()
	file.seek(0, 2)
	try:
		return file.tell()
	finally:
		file.seek(orig)

def create_reader(file):
	def read(offset, length, decode=True):
		file.seek(offset)
		b = file.read(length)
		if not decode:
			return b
		return int.from_bytes(b, "little")
	return read

def transport_note_priority(n, sustained=False, multiplier=8):
	return n[2] + round((n[1] / 127) + n[4] * multiplier / 127) + sustained

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

HEADER = 0xff
TEMPO = 0x51
TITLE_T = 0x04
COPYRIGHT_T = 0x02

NOTE_OFF_C = 0x8
NOTE_ON_C = 0x9
POLY_AFTERTOUCH_C = 0xA
CONTROL_C = 0xB
PROGRAM_C = 0xC
CHANNEL_AFTERTOUCH_C = 0xD
PITCH_BEND_C = 0xE

event_dict = dict(
	header=HEADER,
	tempo=TEMPO,
	title_t=TITLE_T,
	copyright_t=COPYRIGHT_T,

	note_on_c=NOTE_ON_C,
	note_off_c=NOTE_OFF_C,
	poly_aftertouch_c=POLY_AFTERTOUCH_C,
	control_c=CONTROL_C,
	program_c=PROGRAM_C,
	channel_aftertouch_c=CHANNEL_AFTERTOUCH_C,
	pitch_bend_c=PITCH_BEND_C,
)
MIDIEvents = namedtuple("MIDIEvents", tuple(t.upper() for t in event_dict))
event_types = MIDIEvents(*event_dict.values())

# The commented code below was a work-in-progress MIDI parser that was ultimately scrapped due to having worse performance than Midicsv.exe; eventually a rewrite in Rust or similar language may be due.

# from libmidi.types.messages import meta  # noqa: E402
# _orig = meta.BaseMessageMetaText.__dict__['from_bytes'].__func__

# def _mv_friendly_from_bytes(cls, data):
# 	if not isinstance(data, (bytes, bytearray)):
# 		data = bytes(data)
# 	msg, rem = _orig(cls, data)
# 	return msg, rem

# meta.BaseMessageMetaText.from_bytes = classmethod(_mv_friendly_from_bytes)

# try:
# 	from libmidi.types.messages.channel import CHANNEL_MESSAGE_TYPES
# 	from libmidi.types.messages.system import SYSTEM_MESSAGE_TYPES, system_message_from_bytes
# except ImportError:
# 	from libmidi.types.messages.channel import ALL_CHANNEL_MESSAGE_TYPES as CHANNEL_MESSAGE_TYPES
# 	from libmidi.types.messages.system import ALL_SYSTEM_MESSAGE_TYPES as SYSTEM_MESSAGE_TYPES, system_message_from_bytes
# from libmidi.types.messages.meta import META_MESSAGE_VALUE, meta_message_from_bytes  # noqa: E402

# channel_event_attrs = [
# 	['note', 'velocity'],
# 	['note', 'velocity'],
# 	['note', 'value'],
# 	['control', 'value'],
# 	['program'],
# 	['value'],
# 	['value_lsb', 'value_msb'],
# ]
# class Track:

# 	def __init__(self, file, n):
# 		from libmidi.types.chunk import Chunk
# 		from libmidi.utils.variable_length import VariableInt
# 		chunk = Chunk.from_stream(file).data
# 		data = memoryview(bytearray(chunk))

# 		def decode(data):
# 			initial = data
# 			events = deque()
# 			last_status_byte = 0
# 			while data:
# 				t, data = VariableInt.from_bytes(data)
# 				message_status = data[0]
# 				if message_status < 0x80:
# 					message_status = last_status_byte
# 					data = initial[len(initial) - len(data) - 1:]
# 					data[0] = last_status_byte
# 				last_status_byte = message_status
# 				message_type = message_status >> 4
# 				if message_type in CHANNEL_MESSAGE_TYPES:
# 					m = message_type - 8
# 					attributes = channel_event_attrs[m]
# 					a = len(attributes)
# 					data = data[1:]
# 					message_data, data = data[:a], data[a:]
# 					c = message_status & 0xf
# 					match message_type:
# 						case event_types.NOTE_OFF_C | event_types.NOTE_ON_C | event_types.POLY_AFTERTOUCH_C | event_types.CONTROL_C:
# 							e = [n, t, message_type, c, *message_data]
# 						case event_types.PROGRAM_C | event_types.CHANNEL_AFTERTOUCH_C:
# 							e = [n, t, message_type, c, *message_data]
# 						case event_types.PITCH_BEND_C:
# 							e = [n, t, message_type, c, (message_data[1] << 7) | message_data[0]]
# 						case _:
# 							raise NotImplementedError(message_type)
# 					events.append(e)
# 				elif message_status in SYSTEM_MESSAGE_TYPES:
# 					message, data = system_message_from_bytes(data)
# 					# print(t, message, len(data))
# 				elif message_status == META_MESSAGE_VALUE:
# 					message, data = meta_message_from_bytes(data)
# 					# print(t, message, len(data))
# 					match message.meta_message_type:
# 						case 2:
# 							e = [n, t, COPYRIGHT_T, message.text]
# 						case 4:
# 							e = [n, t, TITLE_T, message.text]
# 						case 81:
# 							e = [n, t, TEMPO, message.tempo]
# 						case _:
# 							continue
# 					events.append(e)
# 			return events
# 		self.events = decode(data)


# def parse_midi(file):
# 	"Minimalist MIDI parser that only reads required events, using lightweight data types"
# 	if isinstance(file, str):
# 		file = open(file, "rb")
# 	from libmidi.types.header import Header
# 	header = Header.from_stream(file)
# 	events = [[0, 0, HEADER, header.format, header.ntrks, header.division]]
# 	for n in range(header.ntrks):
# 		track = Track(file, n)
# 		events.extend(track.events)
# 		print(n, len(track.events))
# 	return events


def merge_imports(inputs, ctx):
	event_list = []
	transport = []
	instrument_activities = {}
	speed_info = None
	note_candidates = 0
	for data in inputs:
		if isinstance(data, (list, deque, csv_reader)):
			if not isinstance(data, (list, deque)) or data and isinstance(data[0][2], str):
				midi_events = [(int(e[0]), int(e[1]), getattr(event_types, e[2].strip().upper(), 0), *e[3:]) for e in data]
			else:
				midi_events = data
			event_list.append(midi_events)
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
		# if len(event_list) > 1:
		# 	all_events = list(itertools.chain.from_iterable(event_list))
		# else:
		# 	all_events = event_list[0]
		# all_events.sort(key=itemgetter(1))
		# speed_info = midi.get_step_speed(all_events, ctx=ctx)
		main_speed_info = None
		main_tempo = 0
		for midi_events in event_list:
			speed_info = midi.get_step_speed(midi_events, ctx=ctx)
			notes, nc, is_org, instrument_activities2, speed_info = midi.deconstruct(midi_events, speed_info, ctx=ctx)
			merge_activities(instrument_activities, instrument_activities2)
			note_candidates += nc
			if len(inputs) == 1:
				transport = notes
				break
			orig_ms_per_clock, real_ms_per_clock, scale, orig_step_ms, _orig_tempo = speed_info
			speed_ratio = real_ms_per_clock / scale / orig_ms_per_clock
			wait = round(orig_step_ms / speed_ratio / 50) * 50 if ctx.mc_legal else orig_step_ms / speed_ratio
			tempo = 1000 / wait
			if main_speed_info is None:
				main_speed_info = speed_info
				main_tempo = tempo
				ratio = 1
			else:
				ratio = fractions.Fraction(main_tempo / tempo).limit_denominator(12)
				if ratio.is_integer():
					ratio = int(ratio)
			if type(ratio) is not int:
				mult = lcm(ratio.numerator, ratio.denominator)
				r = mult // ratio.numerator
				last = []
				temp = []
				for i in range(r * len(transport)):
					if i % r == 0:
						last = transport[i // r]
						temp.append(last)
						last = last.copy()
						for j, note in enumerate(last):
							if note[0] == -1:
								continue
							last[j] = (*note[:2], 0, *note[3:])
					else:
						temp.append(last)
				transport = temp
				ratio = mult // ratio.denominator
			speed_info = (orig_ms_per_clock / ratio, real_ms_per_clock, scale, orig_step_ms, _orig_tempo)
			if ratio < 1:
				ratio = 1 / ratio
				notes, transport = transport, notes
			print(ratio)
			ratio = round(ratio)
			for i, beat in enumerate(notes):
				i2 = i * ratio
				while len(transport) <= i2:
					transport.append([])
				transport[i2].extend(beat)
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
	divisions = 4
	scores = [0] * divisions
	for i in range(0, len(transport), divisions):
		priorities = [sum(map(transport_note_priority, beat)) for beat in transport[i:i + divisions]]
		for j in range(divisions):
			if j >= len(priorities):
				break
			scores[j] += priorities[j]
	strongest_beat = scores.index(max(scores))
	if strongest_beat != 0:
		buffer = [[]] * (4 - strongest_beat)
		transport = buffer + transport

def get_parser():
	import argparse
	parser = argparse.ArgumentParser(
		prog="hyperchoron",
		# drop the eager version interpolation here:
		description="MIDI-Tracker-DAW converter and Minecraft Note Block exporter",
	)
	# install our lazy action instead of action="version"; _version.py takes an additional 0.5~1s overhead which we do not want to introduce to the whole program
	from ._version import get_versions
	class LazyVersion(argparse.Action):
		def __init__(self, option_strings, dest=argparse.SUPPRESS,
					default=argparse.SUPPRESS,
					help="show program's version and exit"):
			super().__init__(
				option_strings=option_strings,
				dest=dest,
				default=default,
				nargs=0,
				help=help
			)
		def __call__(self, parser, namespace, values, option_string=None):
			ver = get_versions()['version']
			print(f"{parser.prog} v{ver}")
			parser.exit()
	parser.add_argument(
		"-V", "--version",
		action=LazyVersion
	)
	parser.add_argument("-i", "--input", nargs="+", help="Input file (.zip | .mid | .csv | .nbs | .org | *)", required=True)
	parser.add_argument("-o", "--output", nargs="*", help="Output file (.mid | .csv | .nbs | .nbt | .mcfunction | .litematic | .org | *)", required=True)
	parser.add_argument("-r", "--resolution", nargs="?", type=float, default=None, help="Target time resolution of data, in hertz (per-second). Defaults to 20 for Minecraft outputs, 40 otherwise")
	parser.add_argument("-s", "--speed", nargs="?", type=float, default=1, help="Scales song speed up/down as a multiplier, applied before tempo sync; higher = faster. Defaults to 1")
	parser.add_argument("-v", "--volume", nargs="?", type=float, default=1, help="Scales volume of all notes up/down as a multiplier, applied before note quantisation. Defaults to 1")
	parser.add_argument("-t", "--transpose", nargs="?", type=int, default=0, help="Transposes song up/down a certain amount of semitones, applied before instrument material mapping; higher = higher pitched. Defaults to 0")
	parser.add_argument("-ik", "--invert-key", action=argparse.BooleanOptionalAction, default=False, help="Experimental: During transpose step, autodetects song key signature, then inverts it (e.g. C Major <=> C Minor). Defaults to FALSE")
	parser.add_argument("-sa", "--strum-affinity", nargs="?", default=1, type=float, help="Increases or decreases threshold for sustained notes to be cut into discrete segments; higher = more notes. Defaults to 1")
	parser.add_argument("-d", "--drums", action=argparse.BooleanOptionalAction, default=True, help="Allows percussion channel. If disabled, percussion channels will be treated as regular instrument channels. Defaults to TRUE")
	parser.add_argument("-ml", "--mc-legal", action=argparse.BooleanOptionalAction, default=None, help="Forces song to be vanilla Minecraft compliant. Defaults to TRUE for .litematic, .mcfunction and .nbt outputs, FALSE otherwise")
	parser.add_argument("-md", "--max-distance", nargs="?", type=int, default=42, help="For Minecraft outputs only: Restricts the maximum block distance the notes may be placed from the centre line of the structure, in increments of 3 (one module). Decreasing this value makes the output more compact, at the cost of note volume accuracy. Defaults to 42")
	parser.add_argument("-cb", "--command-blocks", action=argparse.BooleanOptionalAction, default=None, help="For Minecraft outputs only: Enables the use of command blocks in place of notes that require finetune/pitchbend to non-integer semitones. Not survival-legal. Defaults to FALSE if --mc-legal is set, TRUE otherwise")
	parser.add_argument("-mi", "--minecart-improvements", action=argparse.BooleanOptionalAction, default=False, help="For Minecraft outputs only: Assumes the server is running the [Minecart Improvements](https://minecraft.wiki/w/Minecart_Improvements) version(s). Less powered rails will be applied on the main track, to account for the increased deceleration. Currently only semi-functional; the rail section connecting the midway point may be too slow.")
	return parser