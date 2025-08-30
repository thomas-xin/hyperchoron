import collections
from collections import deque, namedtuple
import csv
from dataclasses import dataclass
import datetime
import fractions
import functools
import lzma
from math import isqrt, sqrt, log10
import os
import random
import shutil
import struct
import time
import numpy as np
from .mappings import note_names, note_names_ex, c1


in_sample_rate = 44100
out_sample_rate = 48000
DEFAULT_NAME = "Hyperchoron"
DEFAULT_DESCRIPTION = f"Exported by Hyperchoron on {datetime.datetime.now().date()}"
base_path = __file__.replace("\\", "/").rsplit("/", 1)[0] + "/"
temp_dir = os.path.abspath(base_path.rsplit("/", 2)[0]).replace("\\", "/").rstrip("/") + "/temp/"
if not os.path.exists(temp_dir):
	os.mkdir(temp_dir)
binary_dir = os.path.abspath(base_path.rsplit("/", 2)[0]).replace("\\", "/").rstrip("/") + "/binaries/"
if not os.path.exists(binary_dir):
	os.mkdir(binary_dir)
csv_reader = type(csv.reader([]))

if not hasattr(time, "time_ns"):
	time.time_ns = lambda: int(time.time() * 1e9)

UNIQUE_TS = 0
def ts_us():
	global UNIQUE_TS
	ts = max(UNIQUE_TS + 1, time.time_ns())
	UNIQUE_TS = ts
	return ts

fluidsynth = os.path.abspath(base_path + "/fluidsynth/fluidsynth")
orgexport = os.path.abspath(base_path + "/fluidsynth/orgexport202")

def get_sf2():
	sf2 = binary_dir + "soundfont.sf2"
	if not os.path.exists(sf2) or not os.path.getsize(sf2):
		s7z = os.path.abspath(temp_dir + "soundfont.7z")
		if not os.path.exists(s7z):
			import urllib.request
			req = urllib.request.Request("https://mizabot.xyz/u/iO-ouosmGJ_wB-xHIHx3H2wypUmn/soundfont.7z", headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"})
			with open(s7z, "wb") as f:
				f.write(urllib.request.urlopen(req).read())
		import py7zr
		with py7zr.SevenZipFile(s7z, mode='r') as z:
			z.extractall(temp_dir)
		import subprocess
		sf2convert = os.path.abspath(base_path + "/fluidsynth/sf2convert")
		sf3 = os.path.abspath(temp_dir + "soundfont.sf3")
		subprocess.run([sf2convert, "-x", sf3, sf2])
	return sf2

def get_ext(path) -> str:
	return path.split("?", 1)[0].replace("\\", "/").rsplit(".", 1)[-1].rstrip("/")

archive_formats = ("7z", "zip", "tar", "gz", "bz", "xz")

def extract_archive(archive_path, format=None):
	path = temp_dir + str(ts_us())
	os.mkdir(path)
	if format == "7z" or archive_path.endswith(".7z"):
		import py7zr
		with py7zr.SevenZipFile(archive_path, mode="r") as z:
			z.extractall(path=path)
	else:
		shutil.unpack_archive(archive_path, extract_dir=path, format=format or get_ext(archive_path))
	return [f"{path}/{fn}" for fn in os.listdir(path)]

def create_archive(root, archive_path, format=None):
	if format == "7z" or archive_path.endswith(".7z"):
		import py7zr
		with py7zr.SevenZipFile(archive_path, "w") as z:
			return z.writeall(root)
	return shutil.make_archive(archive_path.rsplit(".", 1)[0], format=format or get_ext(archive_path), root_dir=root)

@functools.lru_cache(maxsize=4096)
def get_children(path) -> list:
	assert isinstance(path, str), "Only filename strings are currently supported."
	if path.startswith("https://") or path.startswith("http://"):
		import urllib.request
		req = urllib.request.Request(path, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"})
		name = path.split("?", 1)[0].rsplit("/", 1)[-1].replace("__", "_")
		path = temp_dir + name
		with open(path, "wb") as f:
			f.write(urllib.request.urlopen(req).read())
	assert os.path.exists(path), f"File {path} does not exist or is not accessible."
	if os.path.isdir(path):
		return [f"{path}/{fn}" for fn in os.listdir(path)]
	if path.rsplit(".", 1)[-1].casefold() in archive_formats:
		return extract_archive(path)
	return [path]

def to_numpy(events, sort=True):
	if not isinstance(events, np.ndarray):
		max_len = max(map(len, events))
		padded = [row + [0] * (max_len - len(row)) for row in events]
		data = np.array(padded, dtype=object)
		assert len(data.shape) == 2, (data, data.shape)
		data[:, 0] = np.int32(data[:, 0])
		data[:, 1] = arround_min(data[:, 1])
		if len(data) and isinstance(data[0][2], str):
			data[:, 2] = [getattr(event_types, e.strip().upper(), 0) for e in data[:, 2]]
		data[:, 3] = np.int32(data[:, 3])
		data[:, 4] = np.float32(data[:, 4])
		events = data
	if sort:
		timestamps = events[:, 1]
		highest = np.max(timestamps)
		if highest < 65535:
			timestamps = np.uint16(timestamps)
		events = events[np.argsort(timestamps, kind="stable")]
	return events

def round_min(x):
	try:
		y = int(x)
	except (ValueError, OverflowError, TypeError):
		return x
	if x == y:
		return y
	return x

def arround_min(x, atol=1 / 4096):
	if x.dtype == object:
		x = np.float64(x)
	elif x.dtype not in (np.float16, np.float32, np.float64):
		return x
	try:
		if np.allclose(x, np.round(x), atol=atol):
			if x.dtype == np.float64:
				return x.astype(np.int64)
			return x.astype(np.int32)
		return x
	except TypeError:
		print(x.dtype)
		raise

def round_random(x) -> int:
	try:
		y = int(x)
	except (ValueError, TypeError):
		return x
	if y == x:
		return y
	x -= y
	if random.random() <= x:
		y += 1
	return y

def as_int(x):
	try:
		return x.item()
	except AttributeError:
		return int(x)

def as_float(x):
	try:
		return x.item()
	except AttributeError:
		return float(x)

def resolve(path, scope=None):
	scope = scope or globals()
	root, *rest = path.split(".")
	try:
		obj = scope[root]
	except KeyError:
		obj = scope[root] = getattr(__import__(__name__.split(".", 1)[0] + "." + root), root)
	if rest:
		from operator import attrgetter
		return attrgetter(".".join(rest))(obj)
	return obj

def log2lin(x, min_db=-40):
	if x <= 0:
		return 0
	return 10 ** ((x - 1) * -min_db / 20)

def lin2log(y, min_db=-40):
	if y <= 0:
		return 0
	return 1 + 20 * log10(y) / (-min_db)

def leb128(n):
	"Encodes an integer using LEB128."
	data = bytearray()
	while n:
		data.append(n & 127)
		n >>= 7
		if n:
			data[-1] |= 128
	return data or bytearray(b"\x00")
def decode_leb128(data, mode="cut"): # mode: cut | index
	"Decodes an integer from LEB128 encoded data; optionally returns the remaining data."
	i = n = 0
	shift = 0
	for i, byte in enumerate(data):
		n |= (byte & 0x7F) << shift
		if byte & 0x80 == 0:
			break
		else:
			shift += 7
	if mode == "cut":
		return n, data[i + 1:]
	return n, i + 1

def count_leaves(obj, _type=list):
	stack = [obj]
	count = 0
	while stack:
		current = stack.pop()
		if isinstance(current, _type):
			stack.extend(current)
		else:
			count += 1
	return count

def approximate_gcd(arr, min_value=8):
	if not len(arr):
		return 0, 0

	# Check if any element is >= min_value
	has_element_above_min = np.any(arr >= min_value)
	if not has_element_above_min:
		return np.gcd.reduce(arr), len(arr)

	# Collect non-zero elements
	non_zero = np.nonzero(arr)[0]
	if not len(non_zero):
		return 0, 0  # All elements are zero

	# Generate all possible divisors >= min_value from non-zero elements
	divisors = set()
	for x in np.unique(np.abs(non_zero)):
		# Find all divisors of x
		for i in range(1, int(isqrt(x)) + 1):
			if x in divisors:
				continue
			if x % i == 0:
				if i >= min_value:
					divisors.add(i)
				counterpart = x // i
				if counterpart >= min_value:
					divisors.add(counterpart)

	# If there are no divisors >= min_value, return the GCD of all elements
	if not divisors:
		return np.gcd.reduce(arr), len(arr)

	# Sort divisors in descending order
	sorted_divisors = sorted(divisors, reverse=True)

	max_count = 0
	candidates = []

	# Find the divisor(s) with the maximum count of divisible elements
	for d in sorted_divisors:
		count = np.sum(arr % d == 0)
		if count > max_count:
			max_count = count
			candidates = [d]
		elif count == max_count:
			candidates.append(d)

	# Now find the maximum GCD among the candidates
	max_gcd = 0
	for d in candidates:
		elements = arr[arr % d == 0]
		current_gcd = np.gcd.reduce(elements)
		if current_gcd > max_gcd:
			max_gcd = current_gcd

	return (max_gcd, len(arr) - max_count) if max_gcd >= min_value else (np.gcd.reduce(arr), len(arr))

def sync_tempo(timestamps, milliseconds_per_clock, clocks_per_crotchet, tps, orig_tempo, fine=False, ctx=None):
	step_ms = round(1000 / tps)
	timestamp_collection = timestamps.astype(np.int64)
	min_value = step_ms / milliseconds_per_clock
	# print("Estimating true resolution...", len(timestamp_collection), clocks_per_crotchet, milliseconds_per_clock, step_ms, min_value)
	speed, exclusions = approximate_gcd(timestamp_collection, min_value=min_value * 3 / 4)
	if speed <= 0:
		speed = 1
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
			req = 0 if not ctx.strict_tempo else (max(speed2, 1 / speed2) - 1) * sqrt(2)
			# print("Confidence:", inclusions / len(timestamp_collection), req)
			if inclusions < len(timestamp_collection) * req:
				print("Discarding tempo...")
				speed = 1
				use_exact = True
			elif not ctx.strict_tempo:
				speed = speed2
				use_exact = True
		elif speed > min_value * 1.25:
			# print("Finding closest speed...", exclusions, len(timestamps))
			div = round(speed / min_value - 0.25)
			if div <= 1:
				if ctx.strict_tempo:
					if speed % 3 == 0:
						# print("Speed too close for rounding, autoscaling by 2/3...")
						speed *= 2
						speed //= 3
					else:
						# print("Speed too close for rounding, autoscaling by 75%...")
						speed *= 0.75
				elif speed > 1:
					speed //= 2
			elif fine or ctx.strict_tempo:
				speed /= div
			elif div > 2:
				speed /= div / 2
	if not use_exact and ctx.strict_tempo and (speed < min_value * 0.85 or speed > min_value * 1.15):
		frac = fractions.Fraction(min_value / speed).limit_denominator(5)
		if frac > 0:
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
	return n.priority + round(n.velocity * 8) + round(n.pitch / 127) + sustained * multiplier / 8

def priority_ordering(beat, n_largest=100, active=(), lenient=False):
	ins = list(beat)
	if len(ins) == 1:
		return ins

	def weak_priority(n):
		return n.priority + round(n.velocity * 8) + round(n.pitch / 127) + ((n.instrument_class, n.pitch - c1) in active)

	if lenient and n_largest >= len(ins):
		ins.sort(key=weak_priority)
		return ins
	out = []
	seen = set()
	keys = np.array([True] * 12, dtype=np.float32)
	octaves = np.array([True] * 16, dtype=np.float32) * 2.5
	pitches = [round(n.pitch) for n in ins]
	noticeable = np.float32(pitches) / 64 + [n.velocity for n in ins]
	i = np.argmax(noticeable)
	pitches.pop(i)
	out.append(ins.pop(i))
	if ins:
		keylist = [p % 12 for p in pitches]
		octavelist = [p % 12 for p in pitches]
		precomputes = [np.float32(weak_priority(n)) for n in ins]
		while ins and len(out) < n_largest:
			ordering = np.float32(precomputes) + np.float32(keys[keylist]) + np.float32(octaves[octavelist]) - [(p in seen) * 2 for p in pitches]
			selected = np.argmax(ordering)
			precomputes.pop(selected)
			keylist.pop(selected)
			octavelist.pop(selected)
			note = ins.pop(selected)
			keys[round(note.pitch) % 12] = False
			octaves[round(note.pitch) // 12] = False
			seen.add(pitches.pop(selected))
			out.append(note)
	return out

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

HEADER = 0xFF
TEMPO = 0x51
TITLE_T = 0x04
COPYRIGHT_T = 0x02

NOTE_OFF_C = 0x80
NOTE_ON_C = 0x90
POLY_AFTERTOUCH_C = 0xA0
CONTROL_C = 0xB0
PROGRAM_C = 0xC0
CHANNEL_AFTERTOUCH_C = 0xD0
PITCH_BEND_C = 0xE0

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

# Modalities:
# 0: MIDI
# 1: NBS
# 2: ORG

@dataclass(slots=True)
class NoteSegment:
	priority: int
	modality: int
	instrument_id: int
	instrument_class: int
	pitch: float
	velocity: float
	panning: float
	timing: int

	def __hash__(self):
		return sum(round(getattr(self, k) * 7 ** i) for i, k in enumerate(self.__slots__))

	def __lt__(self, other):
		return (self.pitch, self.velocity, abs(self.panning)) < (other.pitch, other.velocity, abs(other.panning))

	def __gt__(self, other):
		return (self.pitch, self.velocity, abs(self.panning)) > (other.pitch, other.velocity, abs(other.panning))

	def copy(self):
		return self.__class__(*(getattr(self, k) for k in self.__slots__))

	def is_compressible(self, modality=0):
		if modality != self.modality:
			return False
		if type(self.pitch) not in (int, np.integer) and not self.pitch.is_integer():
			return False
		if self.velocity and (self.velocity < 1 / 127 or self.velocity >= 256 / 127):
			return False
		return True

filters = {"id": lzma.FILTER_LZMA2, "preset": 7 | lzma.PRESET_DEFAULT}
compress_threshold = 86400

class TransportSection(collections.abc.MutableSequence):

	codec = "<BBbeebB"
	compact_codec = "<BBbBBb"
	codec_size = struct.calcsize(codec)
	compact_codec_size = struct.calcsize(compact_codec)

	def __init__(self, beats=None, tick_delay=0):
		self.data = beats or []
		self.tick_delay = fractions.Fraction(tick_delay).limit_denominator(48000)

	def serialise(self):
		return struct.pack("<LLL", len(self), self.tick_delay.numerator, self.tick_delay.denominator) + self.compress()

	@classmethod
	def deserialise(cls, data, clone=True):
		data = memoryview(data)
		_len, num, denom = struct.unpack("<LLL", data[:12])
		data = data[12:]
		if clone and type(data) is not bytes:
			data = bytes(data)
		self = cls(beats=data, tick_delay=fractions.Fraction(num, denom))
		self._len = _len
		return self

	def pack_note(self, note):
		meta = ((note.priority & 15) << 4) | note.modality
		return struct.pack(
			self.codec,
			meta,
			note.instrument_id & 255,
			note.instrument_class,
			note.pitch,
			note.velocity,
			round(note.panning * 127),
			note.timing & 255,
		)
	def unpack_note(self, data):
		meta, instrument_id, instrument_class, pitch, velocity, panning, timing = struct.unpack(self.codec, data)
		return NoteSegment(
			p if (p := meta >> 4) != 15 else -1,
			meta & 15,
			instrument_id,
			instrument_class,
			pitch,
			velocity,
			panning / 127,
			timing,
		)

	def pack_note_compact(self, note):
		meta = ((note.priority & 15) << 4) | note.timing & 15
		return struct.pack(
			self.compact_codec,
			meta,
			note.instrument_id & 255,
			note.instrument_class,
			round(note.pitch),
			min(255, round(note.velocity * 255)),
			round(note.panning * 127),
		)
	def unpack_note_compact(self, data, modality=0):
		meta, instrument_id, instrument_class, pitch, velocity, panning = struct.unpack(self.compact_codec, data)
		return NoteSegment(
			p if (p := meta >> 4) != 15 else -1,
			modality,
			instrument_id,
			instrument_class,
			pitch,
			velocity / 255,
			panning / 127,
			meta & 15,
		)

	def compress(self):
		if not isinstance(self.data, (bytes, memoryview)):
			self._len = len(self.data)
			data = deque()
			for beat in self.data:
				modality = beat[0].modality if beat else 0
				if not beat or all(note.is_compressible(modality) for note in beat):
					header = leb128(len(beat))
					header.append(modality & 255)
					block = [header]
					block.extend(map(self.pack_note_compact, beat))
					data.extend(block)
				else:
					header = leb128(len(beat))
					header[-1] |= 0x80
					header.append(0)
					block = [header]
					block.extend(map(self.pack_note, beat))
					data.extend(block)
			data = b"".join(data)
			self.data = lzma.compress(data, format=lzma.FORMAT_RAW, filters=[filters], check=lzma.CHECK_NONE)
		return self.data

	def decompress(self):
		if isinstance(self.data, (bytes, memoryview)):
			data = memoryview(lzma.decompress(self.data, format=lzma.FORMAT_RAW, filters=[filters]))
			blocks = []
			while data:
				beat = []
				n, i = decode_leb128(data, mode="index")
				if data[i - 1] == 0 and data[0] != 0:
					data = data[i:]
					size = n * self.codec_size
					block, data = data[:size], data[size:]
					while block:
						beat.append(self.unpack_note(block[:self.codec_size]))
						block = block[self.codec_size:]
				else:
					modality = data[i]
					data = data[i + 1:]
					size = n * self.compact_codec_size
					block, data = data[:size], data[size:]
					while block:
						beat.append(self.unpack_note_compact(block[:self.compact_codec_size], modality=modality))
						block = block[self.compact_codec_size:]
				blocks.append(beat)
			self.data = blocks
		self._len = len(self.data)
		return self.data

	def __len__(self):
		try:
			return self._len
		except AttributeError:
			return len(self.decompress())

	def __iter__(self):
		return iter(self.decompress())

	def __reversed__(self):
		return reversed(self.decompress())

	def __getitem__(self, k):
		if type(k) is int and k >= len(self):
			raise IndexError
		return self.decompress()[k]

	def __setitem__(self, k, v):
		if type(k) is int and k >= len(self):
			raise IndexError
		self.decompress()[k] = v

	def append(self, v):
		self.decompress().append(v)
		self._len += 1

	def insert(self, k, v):
		self.decompress().insert(k, v)
		self._len += 1

	def extend(self, v):
		self.decompress().extend(v)
		self._len += len(v)

	def __delitem__(self, k, **kwargs):
		if type(k) is int and k >= len(self):
			raise IndexError
		try:
			return self.decompress().pop(k, **kwargs)
		finally:
			self._len = len(self.data)


class Transport(collections.abc.MutableSequence):

	def __init__(self, notes=None, tick_delay=0):
		self._tick_delay = tick_delay
		self.sections = []

	def serialise(self):
		if self.tick_delay:
			for s in self.sections:
				if not s.tick_delay:
					s.tick_delay = self.tick_delay
		data = b"".join([leb128(len(b := s.serialise())) + b for s in self.sections])
		version = 1
		title = "Untitled".encode("utf-8")
		header = b"~HPC" + struct.pack("<L", version)
		header += leb128(len(title)) + title
		return header + data

	@classmethod
	def deserialise(cls, data):
		data = memoryview(data)
		assert data[:4] == b"~HPC"
		version, = struct.unpack("<L", data[4:8])
		assert version == 1, version
		data = data[8:]
		n, data = decode_leb128(data)
		data = data[n:]
		self = cls()
		while data:
			n, data = decode_leb128(data)
			assert len(data) >= n, (n, len(data))
			self.sections.append(TransportSection.deserialise(data[:n]))
			data = data[n:]
		return self

	def get_metadata(self):
		instrument_activities = {}
		for beat in self:
			poly = {}
			for note in beat:
				instrument = note.instrument_class
				volume = note.velocity
				if instrument != -1:
					try:
						poly[instrument] += 1
					except KeyError:
						poly[instrument] = 1
				else:
					poly.setdefault(instrument, 0)
				try:
					instrument_activities[instrument][0] += volume
					instrument_activities[instrument][1] = max(instrument_activities[instrument][1], poly[instrument])
				except KeyError:
					instrument_activities[instrument] = [volume, poly[instrument]]
		speed_info = (self.tick_delay * 1000, self.tick_delay * 1000, 1, self.tick_delay * 1000, self.tick_delay * 1000000)
		return instrument_activities, speed_info

	@property
	def tick_delay(self):
		return self._tick_delay or (self.sections[0].tick_delay if self.sections else 1)

	@tick_delay.setter
	def set_tick_delay(self, tick_delay):
		self._tick_delay = tick_delay

	def append(self, beat):
		if not self.sections or len(self.sections[-1]) >= compress_threshold:
			if self.sections:
				self.sections[-1].compress()
			self.sections.append(TransportSection(tick_delay=self.tick_delay))
		self.sections[-1].append(beat)

	def extend(self, notes):
		while notes:
			if not self.sections or len(self.sections[-1]) >= compress_threshold:
				if self.sections:
					self.sections[-1].compress()
				self.sections.append(TransportSection(tick_delay=self.tick_delay))
			take = max(0, compress_threshold - len(self.sections[-1]))
			self.sections[-1].extend(notes[:take])
			notes = notes[take:]

	def __len__(self):
		return sum(map(len, self.sections))

	def __iter__(self):
		for s in self.sections:
			yield from s.decompress()

	def __reversed__(self):
		for s in reversed(self.sections):
			yield from reversed(s.decompress())

	def __getitem__(self, k):
		if type(k) is slice:
			out = []
			start, stop, step = k.start, k.stop, k.step
			if start is None:
				start = 0
			if stop is None:
				stop = len(self)
			if step is None:
				step = 1
			assert step > 0
			if start < 0:
				start += len(self)
			if stop < 0:
				stop += len(self)
			idx = 0
			for s in self.sections:
				if idx + len(s) <= start:
					continue
				if idx >= stop:
					break
				out.extend(s[start - idx:stop - idx])
				idx += len(s)
			return out[::step] if step != 1 else out
		if k < 0:
			k += len(self)
		ok = k
		for s in self.sections:
			try:
				return s[k]
			except IndexError:
				s.compress()
				k -= len(s)
		raise IndexError(ok)

	def __setitem__(self, k, v):
		if k < 0:
			k += len(self)
		ok = k
		for s in self.sections:
			try:
				s[k] = v
			except IndexError:
				s.compress()
				k -= len(s)
		raise IndexError(ok)

	def __delitem__(self, k, **kwargs):
		if k < 0:
			k += len(self)
		ok = k
		for s in self.sections:
			try:
				return s.pop(k)
			except IndexError:
				s.compress()
				k -= len(s)
			finally:
				if not s:
					self.sections.remove(s)
		raise IndexError(ok)

	def insert(self, k, v):
		if k < 0:
			k += len(self)
		ok = k
		for s in self.sections:
			if k <= len(s):
				return s.insert(k, v)
			else:
				s.compress()
				k -= len(s)
		raise IndexError(ok)


def save_hpc(transport, output, **void):
	print("Exporting HPC...")
	data = transport.serialise()
	# print(transport.tick_delay, [section.tick_delay for section in transport.sections])
	with open(output, "wb") as f:
		f.write(data)
	return len(transport)

def load_hpc(file):
	print("Importing HPC...")
	with open(file, "rb") as f:
		data = f.read()
	return Transport.deserialise(data)

def transpose(transport, ctx):
	key_info = {}
	mapping = {}
	if not ctx.microtones:
		for beat in transport:
			for note in beat:
				note.pitch = round(note.pitch)
	# if ctx.invert_key or not ctx.accidentals:
	notes = np.zeros(12, dtype=np.float32)
	for beat in transport:
		seen = np.zeros(12, dtype=np.float32)
		for note in beat:
			if note.instrument_class == -1 or not note.pitch.is_integer():
				continue
			p = round(note.pitch % 12)
			v = log2lin(note.velocity) * (0.5 if note.priority <= 0 else 1)
			seen[p] = min(1, seen[p] + v)
		notes += np.sqrt(seen)
	major_scores = [0] * 12
	minor_scores = [0] * 12
	for key in range(12):
		score = 0
		for scale in (0, 0, 0, 2, 4, 4, 5, 7, 7, 9, 11):
			score += notes[(key + scale) % 12]
		for scale in (1, 3, 8):
			score -= notes[(key + scale) % 12]
		major_scores[key] = score
		score = 0
		for scale in (0, 0, 0, 2, 3, 3, 5, 7, 7, 8, 11):
			score += notes[(key + scale) % 12]
		for scale in (4, 4, 6):
			score -= notes[(key + scale) % 12]
		minor_scores[key] = score
	candidates = list(enumerate(major_scores + minor_scores))
	candidates.sort(key=lambda t: t[1], reverse=True)
	key = candidates[0][0]
	key_repr = f"{note_names[key]} Major" if key < 12 else f"{note_names[key - 12]} Minor"
	key_info["key"] = note_names[key] if key < 12 else f"{note_names[key - 12]}m"
	key_info["natural_key"] = natural_key = key if key < 12 else (key - 9) % 12
	key_info["natural_name"] = note_names_ex[natural_key]
	print("Detected key signature:", key_repr)
	if ctx.invert_key:
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
	if not ctx.accidentals:
		mapping[(1 + natural_key) % 12] = 1
		mapping[(3 + natural_key) % 12] = 1
		mapping[(6 + natural_key) % 12] = -1
		mapping[(8 + natural_key) % 12] = 1
		mapping[(10 + natural_key) % 12] = 1
	if mapping or ctx.transpose:
		for beat in transport:
			for i, note in enumerate(beat):
				if not note.pitch.is_integer():
					continue
				pitch = note.pitch
				p = pitch % 12
				adj = ctx.transpose + mapping.get(p, 0)
				if adj:
					note.pitch = pitch + adj
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
		for i in range(4 - strongest_beat):
			transport.insert(0, [])
	return key_info

try:
	from importlib.metadata import version
	__version__ = version("hyperchoron")
except Exception:
	__version__ = "0.0.0-unknown"

def get_parser():
	import argparse
	parser = argparse.ArgumentParser(
		prog="hyperchoron",
		description="MIDI-Tracker-DAW converter and Minecraft Note Block exporter",
	)
	parser.add_argument("-V", '--version', action='version', version=f'%(prog)s {__version__}')
	parser.add_argument("-i", "--input", nargs="+", help="Input file (.zip | .mid | .csv | .nbs | .org | *)", required=True)
	parser.add_argument("-o", "--output", nargs="+", help="Output file (.mid | .csv | .nbs | .nbt | .mcfunction | .litematic | .org | .skysheet | .genshinsheet | *)", required=True)
	parser.add_argument("-f", "--format", default=None, help="Output format (mid | csv | nbs | nbt | mcfunction | litematic | org | skysheet | genshinsheet | deltarune | *)")
	parser.add_argument("-x", "--mixing", nargs="?", default="IL", help='Behaviour when importing multiple files. "I" to process individually, "L" to layer/stack, "C" to concatenate. If multiple digits are inputted, this will be interpreted as a hierarchy. For example, for a 3-deep nested zip folder where pairs of midis at the bottom layer should be layered, then groups of those layers should be concatenated, and there are multiple of these groups to process independently, input "ICL". Defaults to "IL"')
	parser.add_argument("-v", "--volume", nargs="?", type=float, default=1, help="Scales volume of all notes up/down as a multiplier, applied before note quantisation. Defaults to 1")
	parser.add_argument("-s", "--speed", nargs="?", type=float, default=1, help="Scales song speed up/down as a multiplier, applied before tempo sync; higher = faster. Defaults to 1")
	parser.add_argument("-r", "--resolution", nargs="?", type=float, default=None, help="Target time resolution of data, in hertz (per-second). Defaults to 12 for .🗿, .skysheet and .genshinsheet outputs, 20 for .nbt, .mcfunction and .litematic outputs, 40 otherwise")
	parser.add_argument("-st", "--strict-tempo", action=argparse.BooleanOptionalAction, default=None, help="Snaps the song's tempo to the target specified by --resolution, being more lenient in allowing misaligned notes to compensate. Defaults to TRUE for .nbt, .mcfunction and .litematic outputs, FALSE otherwise")
	parser.add_argument("-t", "--transpose", nargs="?", type=int, default=0, help="Transposes song up/down a certain amount of semitones, applied before instrument material mapping; higher = higher pitched. Defaults to 0")
	parser.add_argument("-ik", "--invert-key", action=argparse.BooleanOptionalAction, default=False, help="Experimental: During transpose step, autodetects song key signature, then inverts it (e.g. C Major <=> C Minor). Defaults to FALSE")
	parser.add_argument("-mt", "--microtones", action=argparse.BooleanOptionalAction, default=None, help="Allows microtones/pitchbends. If disabled, all notes are clamped to integer semitones. For Minecraft outputs, defers affected notes to command blocks. Has no effect if --accidentals is FALSE. Defaults to FALSE for .nbt, .mcfunction, .litematic, .org, .skysheet and .genshinsheet outputs, TRUE otherwise")
	parser.add_argument("-ac", "--accidentals", action=argparse.BooleanOptionalAction, default=None, help="Allows accidentals. If disabled, all notes are clamped to the closest key signature. Warning: Hyperchoron is currently only implemented to autodetect a single key signature per song. Defaults to FALSE for .skysheet and .genshinsheet outputs, TRUE otherwise")
	parser.add_argument("-d", "--drums", action=argparse.BooleanOptionalAction, default=True, help="Allows percussion channel. If disabled, percussion channels will be discarded. Defaults to TRUE")
	parser.add_argument("-md", "--max-distance", nargs="?", type=int, default=42, help="For Minecraft outputs only: Restricts the maximum block distance the notes may be placed from the centre line of the structure, in increments of 3 (one module). Decreasing this value makes the output more compact, at the cost of note volume accuracy. Defaults to 42")
	parser.add_argument("-mi", "--minecart-improvements", action=argparse.BooleanOptionalAction, default=False, help="For Minecraft outputs only: Assumes the server is running the [Minecart Improvements](https://minecraft.wiki/w/Minecart_Improvements) version(s). Less powered rails will be applied on the main track, to account for the increased deceleration. Currently only semi-functional; the rail section connecting the midway point may be too slow.")
	return parser