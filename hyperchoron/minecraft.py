from collections import abc, deque, namedtuple
import copy
import functools
from math import ceil, sqrt, log2
import litemapy
from nbtlib.tag import Int, Double, String, List, Compound, Byte
import numpy as np
try:
	import tqdm
except ImportError:
	tqdm = None
else:
	import warnings
	warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
from .mappings import (
	material_map, percussion_mats, pitches, falling_blocks, non_note_blocks,
	instrument_names, nbs_names, nbs_values, midi_instrument_selection,
	instrument_codelist, fixed_instruments, default_instruments,
	fs1,
)
from .util import round_min, log2lin, lin2log, base_path, temp_dir, transport_note_priority, DEFAULT_NAME, DEFAULT_DESCRIPTION


def nbt_from_dict(d):
	if isinstance(d, int):
		return Int(d)
	if isinstance(d, float):
		return Double(d)
	if isinstance(d, str):
		return String(d)
	if isinstance(d, list):
		return List(map(nbt_from_dict, d))
	if isinstance(d, dict):
		return Compound({str(k): nbt_from_dict(v) for k, v in d.items()})
	raise NotImplementedError(type(d), d)

def nbt_to_str(d):
	import json5
	return json5.dumps(d, separators=(',', ':'))

air = litemapy.BlockState("minecraft:air")
note_block = litemapy.BlockState("minecraft:note_block", instrument="custom_head")
end_rod = litemapy.BlockState("minecraft:end_rod")
black_carpet = litemapy.BlockState("minecraft:black_carpet")

@functools.lru_cache(maxsize=256)
def get_note_mat(note, odd=False):
	material = material_map[note[0]]
	pitch = note[1]
	if not material:
		try:
			return percussion_mats[round(pitch)]
		except KeyError:
			print("WARNING: Note", pitch, "not yet supported for drums, discarding...")
			return "PLACEHOLDER", 0
	normalised = pitch - fs1
	if normalised < 0:
		normalised %= 12
	elif normalised > 72:
		normalised = 60 + normalised % 12
	assert 0 <= normalised <= 72, normalised
	ins, mod = divmod(normalised, 12)
	ins = int(ins)
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
			# glowstone="amethyst_block",
		)
		replace.update({
			"hay_block+": "amethyst_block+",
			"emerald_block+": "amethyst_block+",
			"amethyst_block+": "amethyst_block+",
			"black_wool+": "amethyst_block",
			"iron_block+": "amethyst_block+",
			# "glowstone+": "amethyst_block+",
		})
		try:
			mat = replace[mat]
		except KeyError:
			return "PLACEHOLDER", 0
	elif note[3] and not odd:
		replace = dict(
			bamboo_planks="pumpkin",
			bone_block="gold_block",
			iron_block="amethyst_block",
			# glowstone="amethyst_block",
		)
		replace.update({
			"bamboo_planks+": "pumpkin+",
			"gold_block+": "packed_ice+",
			"bone_block+": "gold_block+",
			"iron_block+": "amethyst_block+",
			# "glowstone+": "amethyst_block+",
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
	coords = [(x, y - 1, z), (x, y, z), (x, y + 1, z)]
	if replace and base in replace:
		base = replace[base]
	if base == "PLACEHOLDER":
		return
	if base in non_note_blocks:
		yield from (
			(coords[0], base),
			(coords[1], "sculk"),
			(coords[2], "air"),
		)
		return
	if base.endswith("_head") or base.endswith("_skull"):
		yield from (
			(coords[1], "note_block", dict(note=round(pitch), instrument=instrument_names[base])),
			(coords[2], base),
		)
		return
	# assert isinstance(pitch, int), "Finetune not supported in vanilla!"
	use_command_block = not isinstance(pitch, int) and not pitch.is_integer() and ctx and (ctx.command_blocks or ctx.command_blocks is None and not ctx.mc_legal)
	if use_command_block:
		p = 2 ** max(-1, min(1, (pitch / 12 - 1)))
		command = f"playsound minecraft:block.note_block.{instrument_names[base]} record @a[distance=..48] ~ ~ ~ 3 {p}"
		compound = Compound({'id': String('minecraft:command_block'), 'Command': String(command)})
		yield from (
			(coords[1], "note_block", dict(note=round(pitch), instrument="custom_head")),
			(coords[2], "command_block", dict(conditional="false", facing="up"), compound),
		)
		return
	if base in falling_blocks:
		yield ((x, y - 2, z), "tripwire")
	yield from (
		(coords[0], base),
		(coords[1], "note_block", dict(note=round(pitch), instrument=instrument_names[base])),
		(coords[2], "air"),
	)

def map_palette(reg):
	try:
		return reg._last_palette
	except AttributeError:
		pass
	reg._last_palette = {k: i for i, k in enumerate(reg._Region__palette)}
	return reg._last_palette

def move_region(reg, dx, dy, dz):
	reg._Region__x = dx
	reg._Region__y = dy
	reg._Region__z = dz
	# for e in reg.tile_entities:
	# 	e.position = (
	# 		e.position[0] + dx,
	# 		e.position[1] + dy,
	# 		e.position[2] + dz,
	# 	)
	return reg

def get_bounding(reg):
	return Region(reg.min_x() + reg.x, reg.min_y() + reg.y, reg.min_z() + reg.z, reg.max_x() + reg.x, reg.max_y() + reg.y, reg.max_z() + reg.z)

def tile_region(reg, axes):
	"""
	Tile a Region's block data by specified repetition factors.

	Parameters
	----------
	reg : litemapy.Region
		The source region whose blocks and palette will be used.
	axes : sequence of int
		The number of times to repeat the block array along each axis
		(x, y, z).

	Returns
	-------
	litemapy.Region
		A new Region instance with its blocks array tiled according to
		the given axes and its palette copied from the original region.
	"""
	blocks = reg._Region__blocks
	new_blocks = np.tile(blocks, axes)
	region = litemapy.Region(reg.x, reg.y, reg.z, *new_blocks.shape)
	region._Region__blocks = new_blocks
	region._Region__palette = reg._Region__palette.copy()
	return region

def paste_region(dst, src, ignore_src_air=False, ignore_dst_non_air=False):
	"""
	Paste a source Litemapy Region into a destination Litemapy Region, optionally
	skipping air blocks or preserving existing non-air blocks.

	Parameters
	----------
	dst : litemapy.Region
		The destination region to paste into. Must be an instance of litemapy.Region.
	src : litemapy.Region
		The source region to paste from. Must be an instance of litemapy.Region.
	ignore_src_air : bool, optional
		If True, any air blocks (block ID 0) in the source will not overwrite
		the destination. Defaults to False.
	ignore_dst_non_air : bool, optional
		If True, any non-air blocks already present in the destination will not
		be overwritten by the source. Defaults to False.

	Returns
	-------
	litemapy.Region
		A region containing the combined blocks of dst and src. If the source
		completely covers the destination and neither ignore flag is set,
		the source region is returned directly. If the destination needs to be
		expanded to fit the source, a new larger region is created and returned.

	Raises
	------
	AssertionError
		If either `dst` or `src` is not an instance of litemapy.Region.

	Notes
	-----
	- Performs a fast in-place copy when the source fits entirely within the
	  destination and palettes match (or destination palette is trivial).
	- Falls back to a slower element-wise copy with a custom palette lookup
	  dictionary when palettes differ or a full rebuild is required.
	- Recursively resizes the destination only once if the source region lies
	  outside its bounds, ensuring no infinite recursion.
	"""
	assert isinstance(dst, litemapy.Region) and isinstance(src, litemapy.Region), "Both regions must be Litematic regions."
	dst_size = get_bounding(dst)
	src_size = get_bounding(src)
	# If the source completely covers the destination, return the source with no changes
	if not ignore_src_air and not ignore_dst_non_air and is_contained(dst_size, src_size):
		return src
	# Check if a reallocation is necessary (dst_size does not fit src_size)
	if is_contained(src_size, dst_size):
		# Check if the palettes match or if the destination palette is empty, allowing a direct copy
		if dst._Region__palette == src._Region__palette or len(dst._Region__palette) in range(2):
			target = dst._Region__blocks[src.x - dst.x:src.x - dst.x + src.width, src.y - dst.y:src.y - dst.y + src.height, src.z - dst.z:src.z - dst.z + src.length]
			mask = None
			if ignore_src_air:
				mask = src._Region__blocks != 0
			if ignore_dst_non_air:
				if mask is not None:
					mask &= target == 0
				else:
					mask = target == 0
			target[mask] = src._Region__blocks[mask]
			dst._Region__palette = src._Region__palette.copy()
			dst._last_palette = map_palette(src)
			for e in src.tile_entities:
				e = copy.deepcopy(e)
				e.position = (
					e.position[0] + src.x - dst.x,
					e.position[1] + src.y - dst.y,
					e.position[2] + src.z - dst.z,
				)
				dst.tile_entities.append(e)
			return dst
		# Otherwise fall through to the slower copy method
	else:
		# Create a new destination region that is large enough to fit the source region
		bounding = Region(
			min(dst_size.mx, src_size.mx),
			min(dst_size.my, src_size.my),
			min(dst_size.mz, src_size.mz),
			max(dst_size.Mx, src_size.Mx),
			max(dst_size.My, src_size.My),
			max(dst_size.Mz, src_size.Mz),
		)
		new_dest = litemapy.Region(
			bounding.mx, bounding.my, bounding.mz,
			bounding.Mx - bounding.mx + 1,
			bounding.My - bounding.my + 1,
			bounding.Mz - bounding.mz + 1,
		)
		# Complete the reallocation of the destination region with a recursive call (this should never run into infinite recursion as the new region is always large enough)
		dst = paste_region(new_dest, dst)
	# We now have a new destination region that is large enough to fit the source region.
	target = dst._Region__blocks[src.x - dst.x:src.x - dst.x + src.width, src.y - dst.y:src.y - dst.y + src.height, src.z - dst.z:src.z - dst.z + src.length]
	palette = map_palette(dst)
	dst_mask = target == 0 if ignore_dst_non_air else None
	for i, block in enumerate(src._Region__palette):
		if i == 0 and ignore_src_air:
			continue
		try:
			j = palette[block]
		except KeyError:
			j = len(dst._Region__palette)
			dst._Region__palette.append(block)
			palette[block] = j
		mask = src._Region__blocks == i
		if ignore_dst_non_air:
			mask &= dst_mask
		target[mask] = j
	for e in src.tile_entities:
		e = copy.deepcopy(e)
		e.position = (
			e.position[0] + src.x - dst.x,
			e.position[1] + src.y - dst.y,
			e.position[2] + src.z - dst.z,
		)
		dst.tile_entities.append(e)
	return dst

def duplicate_region(reg):
	new_region = litemapy.Region(reg.x, reg.y, reg.z, reg.width, reg.height, reg.length)
	new_region._Region__blocks[:] = reg._Region__blocks
	new_region._Region__palette = reg._Region__palette.copy()
	return new_region

def setblock(dst, coords, block, no_replace=(), tile_entity=None):
	"""
	Sets a block in a region at the specified coordinates. Unlike the original method in litemapy, this one accepts out-of-bounds coordinates and will reallocate the region if necessary. Returns the (possibly reallocated) region, alongside a dictionary of the (possible reallocated) palette. Use a custom dictionary-based palette to avoid the O(n^2) overhead in litemapy from calling `.index()` and `__contains__()` on the palette for every single entry.
	Optionally includes a tile entity to assign alongside the block.
	"""
	if tile_entity:
		tile_entity["x"], tile_entity["y"], tile_entity["z"] = map(Int, coords)
		dst.tile_entities.append(litemapy.TileEntity(tile_entity))

	palette = map_palette(dst)
	try:
		if not no_replace:
			try:
				dst._Region__blocks[coords] = palette[block]
				return dst
			except KeyError:
				pass
			dst[coords] = block
			palette[block] = dst._Region__blocks[coords]
			return dst
		temp = dst[coords]
	except IndexError:
		pass
	else:
		if temp.id not in no_replace:
			try:
				dst._Region__blocks[coords] = palette[block]
				return dst
			except KeyError:
				pass
			dst[coords] = block
			palette[block] = dst._Region__blocks[coords]
		return dst

	assert isinstance(dst, litemapy.Region), "Destination must be a Litematic region."
	coords = tuple(coords)
	if len(coords) != 3:
		raise ValueError("Coordinates must be a 3-tuple.")
	if not all(isinstance(coord, int) for coord in coords):
		raise ValueError("All coordinates must be integers.")
	dst_size = get_bounding(dst)
	bounding = Region(
		min(dst_size.mx, coords[0] + dst.x),
		min(dst_size.my, coords[1] + dst.y),
		min(dst_size.mz, coords[2] + dst.z),
		max(dst_size.Mx, coords[0] + dst.x),
		max(dst_size.My, coords[1] + dst.y),
		max(dst_size.Mz, coords[2] + dst.z),
	)
	new_dest = litemapy.Region(
		bounding.mx, bounding.my, bounding.mz,
		bounding.Mx - bounding.mx + 1,
		bounding.My - bounding.my + 1,
		bounding.Mz - bounding.mz + 1,
	)
	dst = paste_region(new_dest, dst)
	try:
		dst._Region__blocks[coords] = palette[block]
		return dst
	except KeyError:
		pass
	dst[coords] = block
	palette[block] = dst._Region__blocks[coords]
	return dst

def setblock_absolute(dst, coords, block, no_replace=()):
	x, y, z = coords
	return setblock(dst, (x - dst.x, y - dst.y, z - dst.z), block, no_replace=no_replace)

def merge_regions(regions):
	"""
	Merges a list of regions into a single region.

	Parameters
	----------
	regions : list of litemapy.Region
		The regions to merge.

	Returns
	-------
	litemapy.Region
		A new region that encompasses all the input regions.
	"""
	if not regions:
		return litemapy.Region(0, 0, 0, 1, 1, 1)
	min_x = min(region.min_x() for region in regions)
	min_y = min(region.min_y() for region in regions)
	min_z = min(region.min_z() for region in regions)
	max_x = max(region.max_x() for region in regions)
	max_y = max(region.max_y() for region in regions)
	max_z = max(region.max_z() for region in regions)
	dst = litemapy.Region(min_x, min_y, min_z, max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1)
	for region in regions:
		dst = paste_region(dst, region, ignore_src_air=True)
	return dst

def get_region(schem):
	reg, *extra = schem.regions.values()
	if extra:
		return merge_regions(schem.regions.values())
	return reg

def extract_bounds(schem):
	"""
	Extracts and registers all empty (void) regions within a schematic.

	This function computes the overall bounding box of the given schematic,
	then subtracts each occupied region to determine the remaining void
	spaces. It subsequently merges adjacent void regions along the X, Y, and
	Z axes to form larger contiguous voids. Finally, each void is converted
	into a `litemapy.Region` and added to `schem.regions` under keys
	"Void 0", "Void 1", etc.

	Parameters:
		schem (litemapy.Schematic):
			The schematic object to analyze. It must expose `width`,
			`height`, `length` attributes and internal minimum coordinates
			`_Schematic__x_min`, `_Schematic__y_min`, `_Schematic__z_min`.
			Existing regions are read from `schem.regions`.

	Returns:
		litemapy.Schematic:
			The same schematic instance with additional entries in
			`schem.regions` for each detected void region.
	"""
	mx, my, mz = schem._Schematic__x_min, schem._Schematic__y_min, schem._Schematic__z_min
	bounding = Region(mx, my, mz, mx + schem.width - 1, my + schem.height - 1, mz + schem.length - 1)
	voids = {bounding}
	for region in schem.regions.values():
		remaining_voids = set()
		mx, my, mz = region.min_x(), region.min_y(), region.min_z()
		H = Region(mx, my, mz, region.max_x() - mx, region.max_y() - my, region.max_z() - mz)
		for B in voids:
			if is_disjoint(B, H):
				remaining_voids.add(B)
				continue
			ix0, iy0, iz0 = max(B.mx, H.mx), max(B.my, H.my), max(B.mz, H.mz)
			ix1, iy1, iz1 = min(B.Mx, H.Mx), min(B.My, H.My), min(B.Mz, H.Mz)
			if B.mx < ix0:
				remaining_voids.add(Region(B.mx, B.my, B.mz, ix0, B.My, B.Mz))
			if ix1 < B.Mx:
				remaining_voids.add(Region(ix1, B.my, B.mz, B.Mx, B.My, B.Mz))
			if B.my < iy0:
				remaining_voids.add(Region(ix0, B.my, B.mz, ix1, iy0, B.Mz))
			if iy1 < B.My:
				remaining_voids.add(Region(ix0, iy1, B.mz, ix1, B.my, B.Mz))
			if B.mz < iz0:
				remaining_voids.add(Region(ix0, iy0, B.mz, ix1, iy1, iz0))
			if iz1 < B.Mz:
				remaining_voids.add(Region(ix0, iy0, iz1, ix1, iy1, B.Mz))
		voids = remaining_voids
	while True:
		all_voids = sorted(voids)
		try:
			for i, A in enumerate(all_voids):
				for B in all_voids[i + 1:]:
					if A.Mx == B.mx and A.my == B.my and A.My == B.My and A.mz == B.mz and A.Mz == B.Mz:
						merged = Region(min(A.mx, B.mx), A.my, A.mz, max(A.Mx, B.Mx), A.My, A.Mz)
						voids.discard(A)
						voids.discard(B)
						voids.add(merged)
						raise StopIteration
					if A.My == B.my and A.mx == B.mx and A.Mx == B.Mx and A.mz == B.mz and A.Mz == B.Mz:
						merged = Region(A.mx, min(A.my, B.my), A.mz, A.Mx, max(A.My, B.My), A.Mz)
						voids.discard(A)
						voids.discard(B)
						voids.add(merged)
						raise StopIteration
					if A.Mz == B.mz and A.mx == B.mx and A.Mx == B.Mx and A.my == B.my and A.My == B.My:
						merged = Region(A.mx, A.my, min(A.mz, B.mz), A.Mx, A.My, max(A.Mz, B.Mz))
						voids.discard(A)
						voids.discard(B)
						voids.add(merged)
						raise StopIteration
		except StopIteration:
			continue
		break
	for i, void in enumerate(voids):
		r = litemapy.Region(void.mx, void.my, void.mz, void.Mx + void.mx + 1, void.My + void.my + 1, void.Mz + void.mz + 1)
		schem.regions[f"Void {i}"] = r
	return schem

def crop_region(region, size):
	"""
	Crop a subregion from a Litemapy Region.

	Args:
		region (litemapy.Region): The source region to crop.
		size (tuple[int, int, int, int, int, int]): A 6-tuple (x, y, z, width, height, length)
			specifying the origin and dimensions of the desired subregion.

	Returns:
		litemapy.Region: A new Region instance containing the cropped blocks and a
		copied palette.

	Raises:
		AssertionError: If `region` is not a litemapy.Region or `size` does not
		have exactly six elements.
	"""
	assert isinstance(region, litemapy.Region), "Region must be a Litematic region."
	assert len(size) == 6, "Size must be a tuple of (x, y, z, width, height, length)."
	x, y, z, width, height, length = size
	new_blocks = region._Region__blocks[x:x + width, y:y + height, z:z + length]
	new_region = litemapy.Region(x, y, z, width, height, length)
	new_region._Region__blocks = new_blocks
	new_region._Region__palette = region._Region__palette.copy()
	new_region._optimize_palette()
	return new_region

def flip_region(region, axes=2):
	assert isinstance(region, litemapy.Region), "Region must be a Litematic region."
	region._Region__blocks = np.flip(region._Region__blocks, axes)
	if not isinstance(axes, abc.Iterable):
		axes = [axes]
	for axis in axes:
		flip_palette(region._Region__palette, axis)
	if axes and region.tile_entities:
		mx, my, mz, Mx, My, Mz = get_bounding(region)
		for e in region.tile_entities:
			x, y, z = e.position
			if 0 in axes:
				x = Mx - x + mx
			if 1 in axes:
				y = My - y + my
			if 2 in axes:
				z = Mz - z + mz
			e.position = (x, y, z)
	return region

flips = [
	{"east": "west", "west": "east"},
	{"up": "down", "down": "up", "top": "bottom", "bottom": "top"},
	{"south": "north", "north": "south"},
]
def flip_palette(palette, axis=2):
	"""Flips all directional blocks within the palette. Used by flip_region to have blocks automatically flip along the correct axis."""
	flip = flips[axis]
	for i, block in enumerate(palette):
		properties = {}
		for k, v in block._BlockState__properties.items():
			properties[k] = v
			for x, y in flip.items():
				if x in v:
					if y in v:
						break
					properties[k] = v.replace(x, y)
					break
		palette[i] = litemapy.BlockState(block.id, **properties)
	return palette

def _optimize_palette(self) -> None:
	"""A more computationally efficient version of litemapy.schematic.Region._optimize_palette that does not re-scan the full list of blocks for every entry."""
	required_blocks = set(np.unique(self._Region__blocks))
	new_palette = []
	for old_index, state in enumerate(self._Region__palette):
		# Skip unused entries, except air that needs to remain at index 0
		if old_index != 0 and old_index not in required_blocks:
			continue
		# Do not copy duplicate entries multiple times
		for i, other_state in enumerate(new_palette):
			if state == other_state:
				new_index = i
				break
		else:
			# Keep that entry
			new_index = len(new_palette)
			new_palette.append(state)
		# Update blocks to reflect the new palette
		self._Region__replace_palette_index(old_index, new_index)
	self._Region__palette = new_palette
	# Additionally fix list of tile entities to only contain one per block (assertion is for tests)
	# for te in self.tile_entities:
	# 	tid, pos = te.data["id"].split(":", 1)[-1], te.position
	# 	assert tid in self[pos].id, (self[pos], te.data["id"], pos)
	tile_entity_map = {te.position: te for te in self.tile_entities}
	if len(self.tile_entities) != len(tile_entity_map):
		self.tile_entities = list(tile_entity_map.values())
litemapy.schematic.Region._optimize_palette = _optimize_palette

Region = namedtuple("Region", ("mx", "my", "mz", "Mx", "My", "Mz"))

def is_disjoint(v1, v2):
	return v1.mx > v2.Mx or v1.my > v2.My or v1.mz > v2.Mz or v1.Mx < v2.mx or v1.My < v2.my or v1.Mz < v2.mz
def is_contained(v1, v2):
	if len(v1) == 3:
		return v1[0] >= v2.mx and v1[1] >= v2.my and v1[2] >= v2.mz and v1[0] <= v2.Mx and v1[1] <= v2.My and v1[2] <= v2.Mz
	return v1.mx >= v2.mx and v1.my >= v2.my and v1.mz >= v2.mz and v1.Mx <= v2.Mx and v1.My <= v2.My and v1.Mz <= v2.Mz


static_blocks = {}
note_locations = (
	(1, 2), (1, 4), (1, 6), (1, 8),
	(3, 7), (3, 5), (3, 3), (3, 1),
	(5, 2), (5, 4), (5, 6), (5, 8),
	(7, 7), (7, 5), (7, 3), (7, 1),
)
def build_minecraft(transport, ctx, name="Hyperchoron"):
	ticks_per_segment = 32
	ticks_per_half = ticks_per_segment // 2
	total_duration = len(transport)
	total_segments = ceil(total_duration / ticks_per_segment)
	half_segments = ceil(total_segments / 2)
	attenuation_distance_limit = max(3, int(ctx.max_distance / 3) * 3)
	tile_width = attenuation_distance_limit * 2 + 3
	primary_width = 0
	secondary_width = 0
	yc = 14
	master = litemapy.Region(0, 0, -3, 1, 1, 1)
	schem = litemapy.Schematic.load(f"{base_path}litematic/Hyperchoron V2 Header.litematic")
	header, = schem.regions.values()
	header = move_region(header, -8, yc - 12, -3)
	timer, = litemapy.Schematic.load(f"{base_path}litematic/Hyperchoron V2 Timer.litematic").regions.values()
	track, = litemapy.Schematic.load(f"{base_path}litematic/Hyperchoron V2 Track.litematic").regions.values()
	if ctx.minecart_improvements:
		track = setblock(track, (1, 4, 0), litemapy.BlockState("minecraft:rail", shape="north_south"))
	skeleton, = litemapy.Schematic.load(f"{base_path}litematic/Hyperchoron V2 skeleton.litematic").regions.values()
	trapdoor = litemapy.BlockState("minecraft:bamboo_trapdoor", half="bottom", facing="north")
	# For faraway segments that do not need to be as quiet, we can allocate trapdoors to conserve survival mode resources
	skeleton2 = copy.deepcopy(skeleton)
	skeleton2.filter(lambda block: air if block.id == "minecraft:tripwire" else trapdoor if block.id in ("minecraft:note_block", "minecraft:waxed_oxidized_copper_bulb") else block)
	looping, = litemapy.Schematic.load(f"{base_path}litematic/Hyperchoron V2 Looping.litematic").regions.values()
	start = header.z + header.length - 2
	notes = deque(transport)
	nc = 0
	buffer = 4
	# The song is split into two halves going in opposite directions, allowing for the player to be returned to the beginning without extra travel time.
	for backwards in range(2):
		dz = half_segments * skeleton.length if backwards else half_segments * skeleton.length + buffer
		main = litemapy.Region(-1, 0, 0, 3, 31, dz)
		timer1 = tile_region(timer, (1, 1, half_segments - backwards))
		timer1 = move_region(timer1, 0, yc - 10, 0)
		main = paste_region(main, timer1)
		timer1 = move_region(timer1, 0, yc + 6, 0)
		main = paste_region(main, timer1)
		if backwards:
			main[1 - main.x, yc - 10, (half_segments - 1) * skeleton.length] = litemapy.BlockState("minecraft:acacia_trapdoor", half="top", facing="south")
			main[1 - main.x, yc - 9, (half_segments - 1) * skeleton.length] = litemapy.BlockState("minecraft:powered_rail", shape="east_west")
			main[1 - main.x, yc + 6, (half_segments - 1) * skeleton.length] = litemapy.BlockState("minecraft:acacia_trapdoor", half="top", facing="south")
			main[1 - main.x, yc + 7, (half_segments - 1) * skeleton.length] = litemapy.BlockState("minecraft:powered_rail", shape="east_west")
		for segment in range(half_segments):
			edge = segment in range(3) or half_segments - segment - 1 in range(3)
			if not backwards and segment == half_segments - 2:
				primary_width = main.min_x() - main.x + 1
			z = segment * skeleton.length
			skeletons = {}
			loops = {}

			def get_skeleton(x, y):
				nonlocal main, skeleton, skeleton2
				try:
					return skeletons[x, y]
				except KeyError:
					pass
				if abs(x) < 15 or edge:
					skeleton = move_region(skeleton, x, y, z)
					main = paste_region(main, skeleton)
				else:
					skeleton2 = move_region(skeleton2, x, y, z)
					main = paste_region(main, skeleton2)
				taken = skeletons[x, y] = set()
				return taken

			def get_looping(x, y):
				nonlocal main, looping
				try:
					return loops[x, y]
				except KeyError:
					pass
				looping = move_region(looping, x, y, z)
				main = paste_region(main, looping, ignore_src_air=True, ignore_dst_non_air=True)
				# main = setblock(main, (x - main.x + 1, y - main.y + 4, z - main.z), litemapy.BlockState("minecraft:observer", facing="up"))
				taken = loops[x, y] = set()
				return taken

			if backwards and segment == 1:
				get_skeleton(0, yc - 8)

			def add_note(note, tick):
				nonlocal main, nc
				# note = (note[0], round(note[1]), *note[2:])
				pitch = note[1]
				note_hash = note[0] ^ round(pitch) // 36
				panning = note[5]
				volume = note[4]
				if note[2] <= 0:
					volume *= 0.9
				if panning == 0:
					panning = 1 if note_hash & 1 else -1
				base, pitch = get_note_mat(note, odd=tick & 1)
				attenuation_multiplier = 16 if base in ("warped_trapdoor", "bamboo_trapdoor", "oak_trapdoor", "bamboo_fence_gate", "dropper") else 48
				x = round(max(0, 1 - log2lin(volume / 100)) * (attenuation_multiplier / 3 - 1)) * (3 if panning > 0 else -3)
				vel = max(0, min(1, 1 - x / attenuation_distance_limit))
				if not backwards and segment >= half_segments - 2 and primary_width >= 12:
					if x <= -primary_width / 2:
						x = -x
					x = round(x / 3 - min(primary_width / 2 / 3, ((segment - half_segments + 2) * 32 + tick) / 2 / 3)) * 3
				elif backwards and segment < 2 and primary_width >= 12:
					if x >= primary_width / 2:
						x = -x
					x = round(x / 3 + min(primary_width / 2 / 3, ((1 - segment) * 32 + (32 - tick)) / 2 / 3)) * 3
				if x > attenuation_distance_limit - 3:
					x = attenuation_distance_limit - 3
				if x < -attenuation_distance_limit + 3:
					x = -attenuation_distance_limit + 3
				y = (not tick & 1) * 16 + yc - 8
				swapped = False

				try:
					ordering = (1, 0, 2) if x > 0 else (1, 2, 0)
					while True:
						taken = get_skeleton(x, y)
						for xi in ordering:
							pos = (xi, *note_locations[tick // 2])
							if pos not in taken:
								taken.add(pos)
								raise StopIteration
						x += 3 if panning > 0 else -3
						if x > attenuation_distance_limit:
							if swapped or (tick & 1 and vel < 0.25):
								return
							x = -attenuation_distance_limit
							panning = -panning
							swapped = True
						elif x < -attenuation_distance_limit:
							if swapped or (tick & 1 and vel < 0.25):
								return
							x = attenuation_distance_limit
							panning = -panning
							swapped = True
				except StopIteration:
					pass
				blocks = get_note_block(
					note,
					pos,
					None,
					odd=tick & 1,
					ctx=ctx,
				)
				for coords, block, *kwargs in blocks:
					tile_entity = None
					if kwargs:
						if len(kwargs) > 1:
							tile_entity =  kwargs[1]
						target = litemapy.BlockState(f"minecraft:{block}", **{k: str(v) for k, v in kwargs[0].items()})
					else:
						target = static_blocks.get(block) or static_blocks.setdefault(block, litemapy.BlockState(f"minecraft:{block}"))
					if target.id in ("minecraft:note_block", "minecraft:command_block"):
						nc += 1
						if xi != 1:
							if tick & 8:
								before = (coords[0] + x - main.x, coords[1] + y - main.y, coords[2] + z - main.z + 1)
							else:
								before = (coords[0] + x - main.x, coords[1] + y - main.y, coords[2] + z - main.z - 1)
							if xi == 0:
								main = setblock(main, before, litemapy.BlockState("minecraft:wall_torch", facing="west"))
							elif xi == 2:
								main = setblock(main, before, litemapy.BlockState("minecraft:wall_torch", facing="east"))
					coords = (coords[0] + x - main.x, coords[1] + y - main.y, coords[2] + z - main.z)
					no_replace = ("minecraft:observer", "minecraft:repeater", "minecraft:wall_torch") if block == "tripwire" else ()
					main = setblock(main, coords, target, tile_entity=tile_entity, no_replace=no_replace)

			def add_recurring(half, ins, pitch, volume, panning):
				nonlocal main, nc
				# pitch = round(pitch)
				note_hash = ins ^ round(pitch) // 36
				if panning == 0:
					panning = 1 if note_hash & 1 else -1
				note = (ins, pitch, 2, 1, volume, panning)
				base, pitch = get_note_mat(note, odd=half)
				attenuation_multiplier = 16 if base in ("warped_trapdoor", "bamboo_trapdoor", "oak_trapdoor", "bamboo_fence_gate", "dropper") else 48
				y = yc - 14 if half else yc + 2
				swapped = False
				if half:
					volume = min(volume, 100)
					volume *= 1.2
				else:
					volume *= 0.8
				x = round(max(0, 1 - log2lin(volume / 100)) * (attenuation_multiplier / 3 - 1)) * (3 if panning > 0 else -3)
				vel = max(0, min(1, 1 - x / attenuation_distance_limit))
				if not backwards and segment >= half_segments - 2 and primary_width >= 12:
					if x <= -primary_width / 2:
						x = -x
					x = round(x / 3 - min(primary_width / 2 / 3, ((segment - half_segments + 2) * 32 + tick) / 2 / 3)) * 3
				elif backwards and segment < 2 and primary_width >= 12:
					if x >= primary_width / 2:
						x = -x
					x = round(x / 3 + min(primary_width / 2 / 3, ((1 - segment) * 32 + (32 - tick)) / 2 / 3)) * 3
				if x > attenuation_distance_limit - 3:
					x = attenuation_distance_limit - 3
				if x < -attenuation_distance_limit + 3:
					x = -attenuation_distance_limit + 3
				if x == 0 and not half:
					x += 3 if panning > 0 else -3
				try:
					ordering = (1, 0, 2) if x > 0 else (1, 2, 0)
					while True:
						taken = get_looping(x, y)
						for xi in ordering:
							pos = (xi, y, 0)
							if pos not in taken:
								taken.add(pos)
								raise StopIteration
						x += 3 if panning > 0 else -3
						if x > attenuation_distance_limit:
							if swapped or (half and vel < 0.25):
								return
							x = -attenuation_distance_limit
							panning = -panning
							swapped = True
						elif x < -attenuation_distance_limit:
							if swapped or (half and vel < 0.25):
								return
							x = attenuation_distance_limit
							panning = -panning
							swapped = True
						if x == 0 and not half:
							x += 3 if panning > 0 else -3
				except StopIteration:
					pass
				blocks = get_note_block(
					note,
					pos,
					None,
					odd=half,
					ctx=ctx,
				)
				for coords, block, *kwargs in blocks:
					tile_entity = None
					if kwargs:
						if len(kwargs) > 1:
							tile_entity = kwargs[1]
						target = litemapy.BlockState(f"minecraft:{block}", **{k: str(v) for k, v in kwargs[0].items()})
					else:
						target = static_blocks.get(block) or static_blocks.setdefault(block, litemapy.BlockState(f"minecraft:{block}"))
					for z2 in range(2):
						coords2 = (coords[0] + x - main.x, coords[1] - main.y + 1, coords[2] + z - main.z + 2 + z2 * 2)
						main = setblock(main, coords2, target, tile_entity=tile_entity)

			held_notes = [{}, {}]
			if not edge and len(notes) >= ticks_per_segment:
				for half in range(2):
					held = held_notes[half]
					indices = range(half, ticks_per_segment, 2)
					target = None
					for t in indices:
						beat = notes[t] = sorted((note for note in notes[t] if note[2] >= 0), key=lambda note: transport_note_priority(note), reverse=True)
						temp = {}
						for note in beat[:192]:
							ins, pitch, volume, panning = note[0], note[1], note[4], note[5]
							k = (ins, pitch)
							if t >= 2 and k not in held:
								continue
							try:
								target = temp[k]
							except KeyError:
								target = temp[k] = [0, 0, 0]
							target[0] += 1
							target[1] += volume ** 2
							target[2] += panning
						for k, v in temp.items():
							try:
								target = held[k]
							except KeyError:
								target = held[k] = [1, sqrt(v[1]), v[2] / v[0], sqrt(v[1])]
							else:
								target[0] += 1
								target[1] = min(sqrt(v[1]), target[1])
								target[2] += v[2] / v[0]
								target[3] = max(sqrt(v[1]), target[3])
						if not target:
							break
					for k, v in tuple(held.items()):
						if v[0] < ticks_per_half or v[1] > 110 and v[3] < 128:
							held.pop(k)
							continue
						v[2] /= ticks_per_half
					if held:
						for i in indices:
							beat = notes[i]
							removed = []
							for j, note in enumerate(beat):
								k = note[:2]
								if k in held:
									v = held[k][1]
									if note[4] - v <= 5:
										removed.append(j)
									else:
										beat[j] = (*note[:4], note[4] - min(100, v), *note[5:])
							if removed:
								removals = np.zeros(len(beat))
								removals[removed] = True
								notes[i] = [note for n, note in enumerate(beat) if not removals[n]]
			for tick in range(ticks_per_segment):
				if not notes:
					break
				beat = notes.popleft()
				for note in beat[:96]:
					add_note(note, tick)
			if any(held_notes):
				for half, held in enumerate(held_notes):
					for k, v in tuple(held.items()):
						add_recurring(half, *k, *v[1:3])

			for x, y in skeletons:
				if y >= yc and x in range(-3, 4):
					# We must get rid of copper bulbs too close to the player that would make a significant amount of sound. The new block needs to produce a shape update (trigger observers).
					# This block *cannot* be replaced with a note block, crafter etc that would be more quiet, because it must not power adjacent dust when strongly powered (conductive), must not be a QC component (pistons) which would get re-powered by the lane above, and all except the centre one must not be any rail as it would silently bud the rail lines below causing a clock!
					if x == 0:
						main[x + 1 - main.x, yc + 9, z] = litemapy.BlockState("minecraft:powered_rail", shape="ascending_south", powered="false")
					# TODO: Figure out what to do with segments 1 unit away from the player, as they are still (very barely) in range of copper bulb sound, but must not be a rail, meaning a hopper is the only other candidate, but we want to avoid this as it causes passive block entity ticking lag when in large amounts.
					else:
						main[x + 1 - main.x, yc + 9, z] = litemapy.BlockState("minecraft:hopper", facing="south")

			# Connect all modules in each segment
			layers = [[k for k in skeletons if k[1] < 16], [k for k in skeletons if k[1] >= 16]]
			layers[0].extend(k for k in loops if k[1] < 16)
			layers[1].extend(k for k in loops if k[1] >= 16)
			layers[0].sort()
			layers[1].sort()
			for yi, indices in enumerate(layers):
				if not indices:
					continue
				y = yi * 16 + yc - 10
				y2 = y + 1
				mx, Mx = indices[0][0], indices[-1][0]
				if mx < 0:
					if mx >= -7:
						# No extended lines necessary for short distances; default to powered rail
						for x in range(mx + 1, 1):
							primary = (x - main.x, y2, z)
							secondary = (x - main.x, y, z)
							main = setblock(main, secondary, litemapy.BlockState("minecraft:crimson_trapdoor", half="top", facing="west"), no_replace=["minecraft:observer"])
							main[primary] = litemapy.BlockState("minecraft:powered_rail", shape="east_west", powered="false")
					else:
						# Long range necessary; use instant wire design by Kahyzen (https://youtu.be/nJx-o9fDVm4) for lag optimisation
						for x in range(mx + 1, 1):
							primary = (x - main.x, y2, z)
							secondary = (x - main.x, y, z)
							thresh = mx + 8
							offset = x - thresh
							phase = offset % 9 if x <= -9 or offset < 9 else -1
							if phase == 1 or x == 0:
								main[x - main.x, y - 2, z] = litemapy.BlockState("minecraft:crimson_trapdoor", half="top", facing="west")
								main[x - main.x, y - 1, z] = litemapy.BlockState("minecraft:activator_rail", shape="east_west", powered="false")
								main[secondary] = litemapy.BlockState("minecraft:observer", facing="up")
								main[primary] = litemapy.BlockState("minecraft:activator_rail", shape="east_west", powered="true")
								continue
							if phase == 0 or x == -1:
								main[x - main.x, y - 1, z] = litemapy.BlockState("minecraft:crimson_trapdoor", half="top", facing="west")
								main[secondary] = litemapy.BlockState("minecraft:activator_rail", shape="ascending_west", powered="true")
								main[primary] = litemapy.BlockState("minecraft:observer", facing="east")
								main[x - main.x, y + 2, z] = black_carpet
								continue
							main = setblock(main, secondary, litemapy.BlockState("minecraft:crimson_trapdoor", half="top", facing="west"), no_replace=["minecraft:observer"])
							main[primary] = litemapy.BlockState("minecraft:activator_rail", shape="east_west", powered="true")
				if Mx > 0:
					if Mx <= 8:
						for x in range(2, Mx + 2):
							primary = (x - main.x, y2, z)
							secondary = (x - main.x, y, z)
							main = setblock(main, secondary, litemapy.BlockState("minecraft:crimson_trapdoor", half="top", facing="east"), no_replace=["minecraft:observer"])
							main[primary] = litemapy.BlockState("minecraft:powered_rail", shape="east_west", powered="false")
					else:
						for x in range(2, Mx + 2):
							primary = (x - main.x, y2, z)
							secondary = (x - main.x, y, z)
							thresh = Mx - 6
							offset = thresh - x
							phase = offset % 9 if x >= 11 or offset < 9 else -1
							if phase == 1 or x == 2:
								main[x - main.x, y - 2, z] = litemapy.BlockState("minecraft:crimson_trapdoor", half="top", facing="east")
								main[x - main.x, y - 1, z] = litemapy.BlockState("minecraft:activator_rail", shape="east_west", powered="false")
								main[secondary] = litemapy.BlockState("minecraft:observer", facing="up")
								main[primary] = litemapy.BlockState("minecraft:activator_rail", shape="east_west", powered="true")
								continue
							if phase == 0 or x == 3:
								main[x - main.x, y - 1, z] = litemapy.BlockState("minecraft:crimson_trapdoor", half="top", facing="east")
								main[secondary] = litemapy.BlockState("minecraft:activator_rail", shape="ascending_east", powered="true")
								main[primary] = litemapy.BlockState("minecraft:observer", facing="west")
								main[x - main.x, y + 2, z] = black_carpet
								continue
							main = setblock(main, secondary, litemapy.BlockState("minecraft:crimson_trapdoor", half="top", facing="east"), no_replace=["minecraft:observer"])
							main[primary] = litemapy.BlockState("minecraft:activator_rail", shape="east_west", powered="true")
		track1 = tile_region(track, (1, 1, half_segments - 2))
		if backwards:
			track1._Region__z = 2 * skeleton.length
		track1._Region__y = yc
		main = paste_region(main, track1, ignore_src_air=True)
		if backwards:
			secondary_width = main.max_x() + main.x
			tile_width = primary_width + secondary_width
			main = flip_region(main)
			main._Region__x -= tile_width
		else:
			primary_width = main.min_x() - main.x + 1
		main._Region__z = start + backwards
		master = paste_region(master, main, ignore_src_air=True, ignore_dst_non_air=True)
	end = half_segments * skeleton.length + 4

	def locate(pos):
		x, y, z = pos
		return (x - master.x, y - master.y, z - master.z)

	for yi in range(2):
		ym = yc - 12 + yi * 16
		master[locate((1, ym + 3, end - 2))] = litemapy.BlockState("minecraft:shroomlight")
		master[locate((1, ym + 2, end - 1))] = litemapy.BlockState("minecraft:quartz_slab", type="top")
		master[locate((1, ym + 3, end - 1))] = litemapy.BlockState("minecraft:redstone_wire", north="side", south="side")
		master[locate((1, ym + 3, end))] = litemapy.BlockState("minecraft:target")
		master[locate((1, ym + 4, end))] = black_carpet
		master[locate((1, ym + 3, end + 1))] = note_block
		master[locate((1, ym + 4, end + 1))] = black_carpet
		# Generate a new instant wire to connect ends
		for j in range(tile_width):
			phase = j % 9 if j < tile_width - 11 or tile_width < 9 else -1
			x, y, z = -j, ym, end + 1
			if phase == 1 or j == tile_width - 10:
				master[locate((x, y + 1, z))] = litemapy.BlockState("minecraft:warped_trapdoor", half="top", facing="south")
				master[locate((x, y + 2, z))] = litemapy.BlockState("minecraft:powered_rail", shape="ascending_west", powered="true")
				master[locate((x, y + 3, z))] = litemapy.BlockState("minecraft:observer", facing="east")
				master[locate((x, y + 4, z))] = black_carpet
				continue
			if phase == 0 or j == tile_width - 11:
				master[locate((x, y, z))] = litemapy.BlockState("minecraft:warped_trapdoor", half="top", facing="south")
				master[locate((x, y + 1, z))] = litemapy.BlockState("minecraft:powered_rail", shape="east_west", powered="false")
				master[locate((x, y + 2, z))] = litemapy.BlockState("minecraft:observer", facing="up")
				master[locate((x, y + 3, z))] = litemapy.BlockState("minecraft:powered_rail", shape="east_west", powered="true")
				continue
			master[locate((x, y + 2, z))] = litemapy.BlockState("minecraft:warped_trapdoor", half="top", facing="south")
			master[locate((x, y + 3, z))] = litemapy.BlockState("minecraft:powered_rail", shape="east_west", powered="true")
		master[locate((1 - tile_width, ym + 3, end))] = litemapy.BlockState("minecraft:observer", facing="south")
		master[locate((1 - tile_width, ym + 4, end))] = black_carpet
	x, y, z = 0, yc + 1, end - 22
	turn1, = litemapy.Schematic.load(f"{base_path}litematic/Hyperchoron V2 Turn 1.litematic").regions.values()
	turn2, = litemapy.Schematic.load(f"{base_path}litematic/Hyperchoron V2 Turn 2.litematic").regions.values()
	turn3, = litemapy.Schematic.load(f"{base_path}litematic/Hyperchoron V2 Turn 3.litematic").regions.values()
	turn1 = move_region(turn1, x - 1, y, z)
	master = paste_region(master, turn1, ignore_src_air=True)
	turn2 = move_region(turn2, x - tile_width, y - 1, z - 1)
	master = paste_region(master, turn2, ignore_src_air=True)
	turn3 = move_region(turn3, x - tile_width, y, 2)
	master = paste_region(master, turn3, ignore_src_air=True)
	for i in range(1, tile_width - 5):
		if i % 32 == 0:
			master[locate((-i, y + 1, z + 1))] = litemapy.BlockState("minecraft:redstone_block")
			master[locate((-i, y + 2, z + 1))] = litemapy.BlockState("minecraft:powered_rail", shape="east_west", powered="true")
		elif i >= tile_width - 13:
			master[locate((-i, y + 1, z + 1))] = litemapy.BlockState("minecraft:tinted_glass")
			master[locate((-i, y + 2, z + 1))] = litemapy.BlockState("minecraft:powered_rail", shape="east_west", powered="true")
		else:
			master[locate((-i, y + 1, z + 1))] = litemapy.BlockState("minecraft:tinted_glass")
			master[locate((-i, y + 2, z + 1))] = litemapy.BlockState("minecraft:rail", shape="south_west")
	for i in range(tile_width - 3, 8, -1):
		if i % 32 == 0 or i == 9:
			master[locate((-i, y + 1, 3))] = litemapy.BlockState("minecraft:redstone_block")
			master[locate((-i, y + 2, 3))] = litemapy.BlockState("minecraft:powered_rail", shape="east_west", powered="true")
		else:
			master[locate((-i, y + 1, 3))] = litemapy.BlockState("minecraft:tinted_glass")
			master[locate((-i, y + 2, 3))] = litemapy.BlockState("minecraft:rail", shape="south_east")
	# Very hacky way to reinstantiate the schematic from the loaded one; litemapy *will not* produce the signs properly otherwise!
	# I suspect this is due to litemapy exposing raw NBT data rather than parsing it to determine version, leading to exporting as an older schematic version and causing litematic to fallback to placing an empty sign.
	schem.author = DEFAULT_NAME
	schem.description = DEFAULT_DESCRIPTION
	schem.regions.clear()
	schem.regions["Hyperchoron"] = master = paste_region(master, header, ignore_src_air=True)
	master[locate((1, yc + 4, -3))] = litemapy.BlockState("minecraft:warped_sign", rotation="0")
	x, y, z = locate((1, yc + 4, -3))
	lines = []
	curr = ""
	tokens = name.replace("_", " ").split()
	while tokens:
		word = tokens.pop(0)
		if len(word) > 15:
			word, next_word = word[:15], word[15:]
			tokens.insert(0, next_word)
		if len(word) + len(curr) > 14:
			lines.append(curr)
			curr = ""
		if curr:
			curr += " "
		curr += word
	if word:
		lines.append(word)
	messages = []
	if len(lines) <= 2:
		messages.append(Compound({'': String('')}))
	for line in lines[:4]:
		texts = []
		for i, c in enumerate(line):
			r = i / (len(line) - 1)
			R, G, B = round(255 * r), 255, round(255 * (1 - r))
			colour = hex((R << 16) | (G << 8) | B)[2:].upper()
			while len(colour) < 6:
				colour = "0" + colour
			colour = "#" + colour
			texts.append(Compound({'bold': Byte(0), 'color': String(colour), 'text': String(c)}))
		messages.append(Compound({'extra': List[Compound](texts), 'text': String('')}))
	while len(messages) < 4:
		messages.append(Compound({'': String('')}))
	master.tile_entities.append(
		litemapy.TileEntity(
			Compound({'x': Int(x), 'y': Int(y), 'z': Int(z), 'is_waxed': Byte(1), 'id': String('minecraft:sign'), 'front_text': Compound({'color': String('black'), 'messages': List[Compound](messages), 'has_glowing_text': Byte(0)}), 'back_text': Compound({'color': String('black'), 'messages': List[String]([String(''), String(''), String(''), String('')]), 'has_glowing_text': Byte(0)})})
		)
	)
	extremities = get_bounding(master)
	# print(extremities)
	return schem, extremities, nc


def load_nbs(file):
	print("Importing NBS...")
	import pynbs
	if not isinstance(file, (str, bytes)):
		path = str(hash(file))
		tmpl = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
		fn = f"{temp_dir}{tmpl}.nbs"
		with open(fn, "wb") as f:
			f.write(file.read())
		file.close()
		file = fn
	nbs = pynbs.read(file)
	header = nbs.header
	events = [
		[0, 0, "header", 1, 9 + 1, 1],
		[1, 0, "tempo", 1000 * 1000 / header.tempo],
	]
	for i in range(len(midi_instrument_selection) - 1):
		events.append([i + 2, 0, "program_c", i + 10, midi_instrument_selection[i]])
	ticked = {}
	pannings = {}
	for tick, chord in nbs:
		for note in chord:
			instrument_name = nbs.layers[note.layer].name.rsplit("_", 1)[0]
			note_instrument = nbs_values.get(note.instrument, "snare")
			if note_instrument in ("basedrum", "snare", "hat"):
				match note_instrument:
					case "basedrum":
						pitch = 35
					case "snare":
						pitch = 38
					case "hat":
						pitch = 42
					case _:
						raise NotImplementedError(pitch)
				if note.panning not in range(-1, 2) or pannings.get(10):
					events.append([128, tick, "control_c", 10, 10, note.panning / 100 * 63 + 64])
					pannings[10] = note.panning
				volume = round(lin2log(note.velocity * 127 / 100))
				note_event = [128, tick, "note_on_c", 9, pitch, volume, 1, 1]
				events.append(note_event)
				continue
			default = instrument_codelist.index(default_instruments[note_instrument])
			try:
				ins = instrument_codelist.index(instrument_name)
			except ValueError:
				ins = default
			pitch = note.key + round_min(note.pitch / 100) - 33 + fs1 + pitches[note_instrument]
			bucket = (note.layer, pitch)
			if note.panning not in range(-1, 2) or pannings.get(ins + 10):
				events.append([ins + 2, tick, "control_c", ins + 10, 10, note.panning / 100 * 63 + 64])
				pannings[ins + 10] = note.panning
			volume = round(lin2log(note.velocity * 127 / 100))
			if note.panning & 1:
				if bucket in ticked:
					prev = ticked[bucket]
					if prev[5] == volume and tick - prev[1] < header.tempo / 2:
						prev[7] = max(prev[7], 1 + tick - prev[1])
						continue
			note_event = [ins + 2, tick, "note_on_c", ins + 10, pitch, volume, 1, 1]
			events.append(note_event)
			ticked[bucket] = note_event
	events.sort(key=lambda e: e[1])
	return events

def save_nbs(transport, output, speed_info, ctx):
	print("Exporting NBS...")
	out_name = output.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
	orig_ms_per_clock, real_ms_per_clock, scale, orig_step_ms, _orig_tempo = speed_info
	speed_ratio = real_ms_per_clock / scale / orig_ms_per_clock
	wait = round(orig_step_ms / speed_ratio / 50) * 50 if ctx.mc_legal else orig_step_ms / speed_ratio
	tempo = 1000 / wait
	import pynbs
	nbs = pynbs.new_file(
		song_name=out_name,
		tempo=round(tempo, 3),
	)
	nbs.header.song_origin = ctx.input[0].replace("\\", "/").rsplit("/", 1)[-1]
	nbs.header.song_author=DEFAULT_NAME
	nbs.header.description=DEFAULT_DESCRIPTION
	layer_poly = {}
	for i, beat in enumerate(transport):
		current_poly = {}
		beat.sort(key=lambda note_value: (note_value[2], note_value[1]), reverse=True)
		for note in beat:
			if note[2] < 0:
				continue
			ins = note[0]
			base, pitch = get_note_mat(note, odd=i & 1)
			if base == "PLACEHOLDER":
				continue
			instrument = instrument_names[base]
			if ins != -1:
				if not ctx.mc_legal:
					instrument2 = fixed_instruments[instrument_codelist[ins]]
					pitch2 = note[1] - pitches[instrument2] - fs1
					if -12 <= pitch2 < 48:
						pitch = pitch2
						instrument = instrument2
					else:
						pitch2 = note[1] - pitches[instrument] - fs1
						if -33 <= pitch2 < 55:
							pitch = pitch2
			nbi = nbs_names[instrument]
			try:
				current_poly[ins] += 1
			except KeyError:
				current_poly[ins] = 1
			volume = note[4] / 127 * 100
			if note[2] <= 0:
				volume *= min(1, 0.9 ** (tempo / 20))
			if ctx.command_blocks or ctx.command_blocks is None and not ctx.mc_legal:
				pass
			else:
				pitch = round(pitch)
			raw_key = round(pitch)
			kwargs = {}
			if volume != 100:
				kwargs["velocity"] = round(log2lin(volume / 100) * 100)
			panning = int(note[5] * 49) * 2 + (0 if note[2] > 0 else 1 if i & 1 else -1)
			if panning != 0:
				kwargs["panning"] = panning
			if raw_key != pitch:
				kwargs["pitch"] = round((pitch - raw_key) * 100)
			rendered = pynbs.Note(
				tick=i,
				layer=ins,
				key=raw_key + 33,
				instrument=nbi,
				**kwargs,
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

def save_litematic(transport, output, ctx):
	is_nbt = output.casefold().endswith(".nbt")
	if is_nbt:
		print("Exporting NBT...")
	else:
		is_mcfunction = output.casefold().endswith(".mcfunction")
		if is_mcfunction:
			print("Exporting MCFunction...")
		else:
			print("Exporting Litematic...")
	out_name = output.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
	schem, ext, nc = build_minecraft(transport, ctx=ctx, name=ctx.input[0].replace("\\", "/").rsplit("/", 1)[-1])
	if is_nbt:
		reg = get_region(schem)
		nbt = reg.to_structure_nbt(mc_version=4325)
		nbt.filename = out_name
		with open(output, "wb") as f:
			nbt.save(f)
	elif is_mcfunction:
		zrange = 16 * 16
		lines = ["gamerule maxCommandChainLength 2147483647\n"]
		lines.extend(f"forceload add ~{ext.mx} ~{z} ~{ext.Mx} ~{z + zrange}\n" for z in range(ext.mz, ext.Mz + zrange - 1, zrange))
		lines.extend((
			"gamerule commandModificationBlockLimit 2147483647\n",
			f"fill ~{ext.mx} ~{ext.my} ~{ext.mz} ~{ext.Mx} ~{ext.My} ~{ext.Mz} air strict\n",
		))
		reg = get_region(schem)
		blocks = reg._Region__blocks
		mask = blocks != 0
		for z in reg.range_z():
			for y in reg.range_y():
				for x in reg.range_x():
					pos = (x, y, z)
					if mask[pos] != 0:
						block = str(reg[pos]).removeprefix("minecraft:")
						if block not in ("end_rod[facing=up]", "wall_torch[facing=west]", "wall_torch[facing=east]", "sea_lantern"):
							block += " strict"
						lines.append(f"setblock ~{x + reg.x} ~{y + reg.y} ~{z + reg.z} {block}\n")
		lines.append("forceload remove all")
		with open(output, "w") as f:
			f.writelines(lines)
	else:
		schem.name = out_name
		schem.save(output)
	return nc