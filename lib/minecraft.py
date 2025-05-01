import functools
import itertools
from math import ceil, inf, trunc
try:
	import tqdm
except ImportError:
	tqdm = None
else:
	import warnings
	warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
from .mappings import (
	material_map, percussion_mats, pitches,
	instrument_names, nbs_names,
	instrument_codelist, fixed_instruments,
	cheap_materials, expensive_materials,
	fs1,
	MAIN, SIDE, DIV, BAR,
)
from .util import transport_note_priority


def nbt_from_dict(d):
	import nbtlib
	if isinstance(d, int):
		return nbtlib.tag.Int(d)
	if isinstance(d, float):
		return nbtlib.tag.Double(d)
	if isinstance(d, str):
		return nbtlib.tag.String(d)
	if isinstance(d, list):
		return nbtlib.tag.List(map(nbt_from_dict, d))
	if isinstance(d, dict):
		return nbtlib.tag.Compound({str(k): nbt_from_dict(v) for k, v in d.items()})
	raise NotImplementedError(type(d), d)

def nbt_to_str(d):
	import json5
	return json5.dumps(d, separators=(',', ':'))


@functools.lru_cache(maxsize=256)
def get_note_mat(note, odd=False):
	material = material_map[note[0]]
	pitch = note[1]
	if not material:
		try:
			return percussion_mats[pitch]
		except KeyError:
			print("WARNING: Note", pitch, "not yet supported for drums, discarding...")
			return "PLACEHOLDER", 0
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
	base, pitch = get_note_mat(note, odd=odd)
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
				curr = [note for note in beat[pulse] if note[2] >= 0]
				if not curr:
					continue
				cap = MAIN * 2 + SIDE * 2 + 2 if pulse == 0 else SIDE * 4 if pulse == 2 else SIDE * 3
				padding = (-1, 0, inf, 0, 0)
				transparent = ("glowstone", "heavy_core", "blue_stained_glass", "red_stained_glass")
				lowest = min((note[1], note) for note in curr)[1]
				ordered = sorted(curr, key=lambda note: (transport_note_priority(note, note[0] == -1, multiplier=3), note[1]), reverse=True)[:cap]
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
							mat, pitch = get_note_mat(note, odd=pulse & 1)
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
					required_padding = sum(get_note_mat(note, odd=pulse & 1)[0] in transparent for note in ordered) * 2 - len(ordered)
					for p in range(required_padding):
						ordered.append(padding)
					ordered.sort(key=lambda note: get_note_mat(note, odd=pulse & 1)[0] in transparent)
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
						if get_note_mat(note, odd=pulse & 1)[0] in transparent:
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
					((x * 19, y + 1, z), "torch"),
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
					((x * 2, y + 1, z), "torch"),
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
			yield ((x * 3, y - 1, z), "shroomlight")
			for i in range(4, 18):
				yield ((x * i, y - 1, z), "crimson_trapdoor" if i & 1 == right else "acacia_trapdoor", dict(facing="south", half="top"))
			yield ((x * 18, y - 1, z), "shroomlight")
			yield ((x * (18 if mirrored else 3), y, z), "observer", dict(facing="east" if right ^ mirrored else "west"))
			for i in range(3, 18):
				yield ((x * (i if mirrored else i + 1), y, z), "repeater", dict(facing="east" if right ^ mirrored else "west", delay=2))
			for i in range(3, 19):
				yield ((x * i, y + 3, z + 2), "sea_lantern")
			if z <= 1:
				for i in range(3, 19):
					yield ((x * i, y + 3, z - 2), "sea_lantern")
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
					((x * (i * 2 + o2 - 1), y + 1, z), "glass"),
					((x * (i * 2 + o1), y + (1 if lower else -1), z + 2), block),
				)
				if i < 7:
					yield ((x * (i * 2 + o1 + 1), y + (0 if lower else -2), z + 2), slab, dict(type="top"))
					yield ((x * (i * 2 + o1 + 1), y + (1 if lower else -1), z + 2), "repeater", dict(facing="west" if right ^ reverse else "east", delay=2))
					yield ((x * (i * 2 + o1 + 1), y + (1 if lower else -1) + 1, z + 2), "sea_lantern")
			x2 = x * 2 if reverse else x * 19 
			yield from (
				((x * (18 if reverse else 3), y, z), "observer", dict(facing="east" if right ^ reverse else "west")),
				((x * (18 if reverse else 3), y + 1, z), "sea_lantern"),
				((x * (3 if reverse else 18), y + (1 if lower else -1), z + 2), "observer", dict(facing="west" if right ^ reverse else "east")),
				((x * (3 if reverse else 18), y + (1 if lower else -1) + 1, z + 2), "sea_lantern"),
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
				((x2, 13, z), "torch"),
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
	b = 0
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
		((0, -1, 1), "hopper", dict(facing="north"), {'Items': [{'Slot': 0, 'id': 'minecraft:wooden_shovel', 'count': 1}, {'Slot': 1, 'id': 'minecraft:wooden_shovel', 'count': 1}, {'Slot': 2, 'id': 'minecraft:wooden_shovel', 'count': 1}, {'Slot': 3, 'id': 'minecraft:wooden_shovel', 'count': 1}, {'Slot': 4, 'id': 'minecraft:wooden_shovel', 'count': 1}]}),
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
		((0, 1, -1), "shroomlight"),
		((2, 1, -1), "composter", dict(level=6)),
		((1, 1, -1), "comparator", dict(facing="east", powered="true")),
		((0, -1, -2), "sticky_piston", dict(facing="south", extended="true")),
		((0, -1, -1), "piston_head", dict(facing="south", short="false", type="sticky")),
		((0, -3, -2), "target"),
		((0, -2, -2), "redstone_torch"),
		((1, -4, -2), "cobbled_deepslate"),
		((1, -3, -2), "redstone_wire", dict(east="side", west="side")),
		((0, -4, -1), "shroomlight"),
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
		yield ((0, -5, i), "redstone_ore" if i & 1 else "deepslate_redstone_ore")
		yield ((0, -7, i), "glass")
		yield ((0, -6, i), "redstone_torch")
	yield from (
		((0, -5, offset), "deepslate_redstone_ore"),
		((0, -7, offset), "glass"),
		((0, -6, offset), "redstone_torch"),
		((0, -4, offset + 1), "red_wool"),
		((0, -3, offset + 2), "red_wool"),
		((0, -2, offset + 3), "red_wool"),
		((0, -1, offset + 4), "shroomlight"),
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


# def load_nbs(file):
# 	print("Importing NBS...")
# 	import pynbs
# 	nbs = pynbs.read(file)
# 	events = [
# 		[0, 0, "header", 1, len(instruments) + 1, 8],
# 		[1, 0, "tempo", 1000 * 1000 / nbs.header.tempo],
# 	]
# 	for tick, chord in nbs:
# 		while tick > len(transport):
# 			transport.append([])
# 		mapped_chord = []
# 		poly = {}
# 		for note in chord:
# 			instrument_name = nbs.layers[note.layer].name.rsplit("_", 1)[0]
# 			if instrument_name == "Drumset":
# 				ins = 6
# 				default = 6
# 			else:
# 				default = instrument_codelist.index(default_instruments[nbs_values[note.instrument]])
# 				try:
# 					ins = instrument_codelist.index(instrument_name)
# 				except ValueError:
# 					ins = default
# 			block = (
# 				ins,
# 				note.key - 33 + fs1 + pitches[nbs_values[note.instrument]],
# 				not note.panning & 1,
# 				ins != default,
# 				round(note.velocity * 127 / 100),
# 				note.panning / 50,
# 			)
# 			try:
# 				poly[ins] += 1
# 			except KeyError:
# 				poly[ins] = 1
# 			try:
# 				instrument_activities[ins][0] += note.velocity
# 				instrument_activities[ins][1] = max(instrument_activities[ins][1], poly[ins])
# 			except KeyError:
# 				instrument_activities[ins] = [note.velocity, poly[ins]]
# 			mapped_chord.append(block)
# 		transport.append(mapped_chord)
# 	return SimpleNamespace(
# 		transport=transport,
# 		instrument_activities=instrument_activities,
# 		speed_info=(50, 50, 20 / nbs.header.tempo, 50, 500),
# 	)


def save_nbs(transport, output, speed_info, ctx):
	print("Exporting NBS...")
	out_name = output.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
	if ctx.exclusive:
		orig_ms_per_clock, real_ms_per_clock, scale, orig_step_ms, _orig_tempo = speed_info
		speed_ratio = real_ms_per_clock / scale / orig_ms_per_clock
		tempo = round(1000 * speed_ratio / orig_step_ms)
	else:
		tempo = 20
	import pynbs
	nbs = pynbs.new_file(
		song_name=out_name,
		tempo=tempo,
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
			base, pitch = get_note_mat(note, odd=i & 1)
			if base == "PLACEHOLDER":
				continue
			raw = False
			if ins != -1:
				if ctx.exclusive:
					instrument = fixed_instruments[instrument_codelist[ins]]
					pitch = note[1] - pitches[instrument] - fs1
					if pitch < 0:
						pitch += 12
					raw = True
			if not raw:
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
				panning=trunc(note[5] * 49) * 2 + (0 if note[2] > 0 else 1 if i & 1 else -1),
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
	nbs.save(output)
	return len(nbs.notes)

def save_mcfunction(transport, output, ctx):
	print("Exporting MCFunction...")
	block_replacements = cheap_materials if ctx.cheap else expensive_materials
	blocks = render_minecraft(list(transport), ctx=ctx)
	nc = 0
	with open(output, "w") as f:
		for (x, y, z), block, *kwargs in blocks:
			if block in block_replacements:
				block = block_replacements[block]
				if block == "cobblestone_slab":
					kwargs = [dict(type="top")]
			if block == "sand":
				f.write(f"setblock ~{x} ~{y - 1} ~{z} dirt keep\n")
			nc += block == "note_block"
			info = nbt = ""
			if kwargs:
				info = "[" + ",".join(f"{k}={v}" for k, v in kwargs[0].items()) + "]"
				if len(kwargs) > 1:
					nbt = nbt_to_str(kwargs[1])
			f.write(f"setblock ~{x} ~{y} ~{z} {block}{info}{nbt} strict\n")
	return nc

def save_litematic(transport, output, ctx):
	print("Exporting Litematic...")
	out_name = output.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
	block_replacements = cheap_materials if ctx.cheap else expensive_materials
	blocks = render_minecraft(list(transport), ctx=ctx)
	import litemapy
	air = litemapy.BlockState("minecraft:air")
	mx, my, mz = 20, 13, 2
	bars = ceil(len(transport) / BAR / DIV)
	depth = bars * 8 + 8
	reg = litemapy.Region(-mx, -my, -mz, mx * 2 + 1, my * 2 + 1, depth + mz + 8)
	schem = reg.as_schematic(
		name=out_name,
		author="Hyperchoron",
		description="Exported MIDI",
	)
	nc = 0
	for (x, y, z), block, *kwargs in blocks:
		pos = (x + mx, y + my, z + mz)
		if block in block_replacements:
			block = block_replacements[block]
			if block == "cobblestone_slab":
				kwargs = [dict(type="top")]
		if block == "sand" and reg[x + mx, y + my - 1, z + mz] == air:
			reg[x + mx, y + my - 1, z + mz] = litemapy.BlockState("minecraft:dirt")
		nc += block == "note_block"
		if kwargs:
			block = litemapy.BlockState("minecraft:" + block, **{k: str(v) for k, v in kwargs[0].items()})
			if len(kwargs) > 1:
				tile_entity = litemapy.TileEntity(nbt_from_dict(kwargs[1]))
				tile_entity.position = pos
				reg.tile_entities.append(tile_entity)
		else:
			block = litemapy.BlockState("minecraft:" + block)
		reg[pos] = block
	schem.save(output)
	return nc