import functools
import itertools
from math import ceil, inf
import os
from types import SimpleNamespace
if os.name == "nt" and os.path.exists("Midicsv.exe") and os.path.exists("Csvmidi.exe"):
	py_midicsv = None
else:
	import py_midicsv
try:
	import tqdm
except ImportError:
	tqdm = None
else:
	import warnings
	warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
from mappings import (
	material_map, sustain_map,
	instrument_names, percussion_mats, instrument_codelist,
	midi_instrument_selection, org_instrument_selection,
	c4, c1, c_1, fs1,
	MAIN, SIDE, DIV, BAR,
)


def midi2csv(file):
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
	return csv_list
def csv2midi(file, output):
	if py_midicsv:
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

def render_org(notes, instrument_activities, speed_info, ctx):
	orig_ms_per_clock, real_ms_per_clock, scale, _orig_step_ms, _orig_tempo = speed_info
	speed_ratio = real_ms_per_clock / scale / orig_ms_per_clock
	wait = round(50 / speed_ratio)
	activities = list(map(list, instrument_activities.items()))
	instruments = []
	while len(instruments) < 8:
		activities.sort(key=lambda t: t[1][0], reverse=True)
		curr = activities[0]
		curr[1][0] /= 2
		typeid = curr[0]
		itype = org_instrument_selection[typeid]
		if itype < 0:
			itype = 7
		instruments.append(SimpleNamespace(
			id=itype,
			index=len(instruments),
			type=typeid,
			notes=[],
		))
	for i, drum in enumerate([0, 2, 5, 6, 4, 0, 0, 0]):
		instruments.append(SimpleNamespace(
			id=drum,
			index=len(instruments),
			type=-1,
			notes=[],
		))
	note_count = sum(map(len, notes))
	conservative = False
	if note_count > 4096 * 8 * 4:
		print(f"High note count {note_count} detected, using conservative note sustains...")
		conservative = True
	active = {}
	for i, beat in enumerate(notes):
		taken = []
		next_active = {}
		if beat:
			ordered = sorted(beat, key=lambda note: (note[2], round(note[4] * 8), note[0] == -1, note[1]), reverse=True)
			lowest = min((note[1], note) for note in beat)[1]
			ordered.remove(lowest)
			ordered.insert(0, lowest)
		else:
			ordered = list(beat)
		for note in ordered:
			itype, pitch, updated, _long, vel, pan = note
			volume = min(254, round(vel * 2 / 64) * 64 if conservative else round(vel * 2 / 8) * 8)
			panning = round(pan * 6 + 6)
			ins = org_instrument_selection[itype]
			if ins < 0:
				try:
					mat, rpitch = percussion_mats[pitch]
				except KeyError:
					continue
				if mat == "PLACEHOLDER":
					continue
				pitch = 0
				ins = instrument_names[mat]
				match ins:
					case "basedrum":
						if mat == "netherrack":
							iid = 8
						else:
							iid = 12
							pitch = rpitch - 12
					case "snare":
						iid = 9
					case "hat":
						iid = 10
					case "creeper":
						iid = 11
					case _:
						iid = 12
						pitch = rpitch
				if iid in taken:
					continue
				new_pitch = pitch + c4 - c1
				note = SimpleNamespace(
					tick=i,
					pitch=new_pitch,
					length=1,
					volume=volume,
					panning=panning,
				)
				instruments[iid].notes.append(note)
				taken.append(iid)
				continue
			new_pitch = pitch + ctx.transpose - c1
			if new_pitch < 0:
				new_pitch += 12
			if new_pitch < 0:
				new_pitch = 0
			if new_pitch > 95:
				new_pitch = 95
			h = (itype, new_pitch)
			if (updated < 2 or conservative) and h in active:
				reused = False
				for iid in active[h]:
					if iid not in taken:
						instrument = instruments[iid]
						last_vol, last_pan = instrument.notes[-1].volume, instrument.notes[-1].panning
						idx = -1
						while (last_note := instrument.notes[idx]) and last_note.pitch == 255:
							idx -= 1
						if last_note.length < 128:
							if last_vol != volume or (last_pan != panning and not conservative):
								instrument.notes.append(SimpleNamespace(
									tick=i,
									pitch=255,
									length=1,
									volume=volume,
									panning=last_pan if conservative else panning,
								))
							last_note.length += 1
							taken.append(iid)
							try:
								next_active[h].append(instrument.index)
							except KeyError:
								next_active[h] = [instrument.index]
							reused = True
							break
				if reused:
					continue
			choices = instruments[:8]
			choices = sorted(choices, key=lambda instrument: (instrument.index not in taken, len(instrument.notes) < 3840, instrument.id == ins, -(len(instrument.notes) // 960)), reverse=True)
			instrument = choices[0]
			if instrument.index in taken:
				break
			instrument.notes.append(SimpleNamespace(
				tick=i,
				pitch=new_pitch,
				length=1,
				volume=volume,
				panning=panning,
			))
			taken.append(instrument.index)
			try:
				next_active[h].append(instrument.index)
			except KeyError:
				next_active[h] = [instrument.index]
		active = next_active
	return instruments, wait

def render_midi(notes, instrument_activities, speed_info, ctx):
	orig_ms_per_clock, real_ms_per_clock, scale, _orig_step_ms, _orig_tempo = speed_info
	speed_ratio = real_ms_per_clock / scale / orig_ms_per_clock
	wait = round(50 / speed_ratio * 1000 * 8)
	activities = list(map(list, instrument_activities.items()))
	instruments = [SimpleNamespace(
		id=midi_instrument_selection[curr[0]],
		type=curr[0],
		notes=[],
		name=instrument_codelist[curr[0]],
		channel=curr[0] if curr[0] >= 0 else 9,
	) for curr in activities for c in range(curr[1][1] if curr[0] != -1 else 1)]
	drums = None
	for i, ins in enumerate(instruments):
		ins.index = i
		if ins.type == -1:
			drums = ins
	active = {}
	for i, beat in enumerate(notes):
		taken = []
		next_active = {}
		if beat:
			ordered = sorted(beat, key=lambda note: (note[2], round(note[4] * 8), note[0] == -1, note[1]), reverse=True)
			lowest = min((note[1], note) for note in beat)[1]
			ordered.remove(lowest)
			ordered.insert(0, lowest)
		else:
			ordered = list(beat)
		for note in ordered:
			itype, pitch, updated, _long, vel, pan = note
			volume = min(127, vel)
			panning = round(pan * 63 + 64)
			ins = midi_instrument_selection[itype]
			if ins < 0:
				new_pitch = pitch
				note = SimpleNamespace(
					tick=i,
					pitch=new_pitch,
					length=1,
					volume=volume,
					panning=panning,
					events=[],
				)
				if drums:
					drums.notes.append(note)
				continue
			new_pitch = pitch + ctx.transpose - c_1
			if new_pitch < 0:
				new_pitch += 12
			if new_pitch < 0:
				new_pitch = 0
			if new_pitch > 127:
				new_pitch = 127
			h = (itype, new_pitch)
			if updated < 2 and h in active:
				reused = False
				for iid in active[h]:
					if iid not in taken:
						instrument = instruments[iid]
						last_note = instrument.notes[-1]
						last_vol, last_pan = last_note.volume, last_note.panning
						if last_vol >= volume and not sustain_map[instrument.type]:
							pass
						else:
							if last_vol != volume:
								last_note.events.append((i, "volume", min(127, round(volume / last_vol * 100))))
							if last_pan != panning:
								last_note.events.append((i, "panning", panning))
						last_note.length += 1
						taken.append(iid)
						try:
							next_active[h].append(instrument.index)
						except KeyError:
							next_active[h] = [instrument.index]
						reused = True
						break
				if reused:
					continue
			choices = sorted(instruments, key=lambda instrument: (instrument.index not in taken, instrument.id == ins), reverse=True)
			instrument = choices[0]
			instrument.notes.append(SimpleNamespace(
				tick=i,
				pitch=new_pitch,
				length=1,
				volume=volume,
				panning=panning,
				events=[],
			))
			taken.append(instrument.index)
			try:
				next_active[h].append(instrument.index)
			except KeyError:
				next_active[h] = [instrument.index]
		active = next_active
	return instruments, wait

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
					((x * 19, y, z + 1), "black_stained_glass"),
					((x * 19, y, z + 2), "black_stained_glass"),
					((x * 19, y, z), "activator_rail", dict(shape="ascending_south")),
					((x * 19, y, z + 3), "activator_rail", dict(shape="ascending_north")),
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
		((0, 1, 0), "calibrated_sculk_sensor", dict(facing="south")),
		((1, 0, 0), "oxidized_copper_bulb", dict(lit="false")),
		((-1, 0, 0), "oxidized_copper_bulb", dict(lit="false")),
		((1, -1, 0), "obsidian"),
		((-1, -1, 0), "obsidian"),
		((1, 1, 0), "redstone_wire", dict(north="side", west="side")),
		((-1, 1, 0), "redstone_wire", dict(east="side", west="side")),
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
		((0, -4, offset + 1), "red_wool"),
		((0, -3, offset + 2), "red_wool"),
		((0, -2, offset + 3), "red_wool"),
		((0, -1, offset + 4), "red_wool"),
		((0, -4, offset), "powered_rail", dict(shape="ascending_south", powered="true")),
		((0, -3, offset + 1), "powered_rail", dict(shape="ascending_south", powered="true")),
		((0, -2, offset + 2), "powered_rail", dict(shape="ascending_south", powered="true")),
		((0, -1, offset + 3), "powered_rail", dict(shape="north_south", powered="true")),
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