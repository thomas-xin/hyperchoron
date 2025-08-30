import concurrent.futures
import itertools
import os
import numpy as np
from . import midi
from .mappings import midi_instrument_selection
from .util import (
	__version__ as __version__, Transport, ts_us, round_min,
	resolve, event_types, to_numpy, transpose,
	get_ext, get_children, archive_formats, temp_dir,
	count_leaves, create_archive, get_parser,
)


decoder_mapping = dict((k, v) for line in """
txt | md | py : None
hpc : util.load_hpc
nbs : minecraft.load_nbs
mid | midi : midi.load_midi
csv : midi.load_csv
org : tracker.load_org
xm : tracker.load_xm
🗿 | moai : text.load_moai
wav | flac | mp3 | aac | ogg | opus | m4a | weba | webm : pcm.load_raw
_ : dawvert.load_arbitrary
""".splitlines() if line for exts, v in [line.rsplit(" : ", 1)] for k in exts.split(" | "))

encoder_mapping = dict((k, v) for line in """
hpc : util.save_hpc
nbs : minecraft.save_nbs
mid | midi | csv : midi.save_midi
org : tracker.save_org
nbt | mcfunction | litematic : minecraft.save_litematic
🗿 | moai : text.save_moai
skysheet : text.save_skysheet
genshinsheet : text.save_genshinsheet
deltarune : text.save_deltarune
wav | flac | mp3 | aac | ogg | opus | m4a | weba | webm : pcm.save_raw
_ : dawvert.save_arbitrary
""".splitlines() if line for exts, v in [line.rsplit(" : ", 1)] for k in exts.split(" | "))

scope = globals()

def load_file(fi, ctx) -> np.ndarray | Transport:
	name = fi.replace("\\", "/").rsplit("/", 1)[-1]
	ext = name.rsplit(".", 1)[-1].casefold()
	decoder = decoder_mapping.get(ext) or decoder_mapping["_"]
	if decoder == "None":
		return
	func = resolve(decoder, scope=scope)
	data = func(fi)
	if isinstance(data, Transport):
		if ctx.one_to_one and (not ctx.strict_tempo or ctx.resolution == 1 / data.tick_delay):
			return data
		# Reconstruct as MIDI events if further processing is necessary
		events = [
			[0, 0, event_types.HEADER, 1, len(midi_instrument_selection) + 1, 1, 0, 0, 0, 0, 0],
			[1, 0, event_types.TEMPO, data.tick_delay * 1000000, 0, 0, 0, 0, 0, 0, 0],
		]
		for i in range(len(midi_instrument_selection)):
			events.append([i + 2, 0, event_types.PROGRAM_C, i + 10, midi_instrument_selection[i], 0, 0, 0, 0, 0, 0])
		for tick, beat in enumerate(data):
			for note in beat:
				note_event = [note.instrument_class + 2, tick, event_types.NOTE_ON_C, note.instrument_class + 10, note.pitch, note.velocity * 127, note.priority, 1, note.panning, note.timing, note.modality]
				events.append(note_event)
		data = events
	return to_numpy(data, sort=False)

def export(transport, fo, instrument_activities, speed_info, key_info, ctx=None) -> str:
	ext = fo.rsplit(".", 1)[-1].casefold()
	encoder = encoder_mapping.get(ext) or encoder_mapping["_"]
	if encoder == "None":
		return
	func = resolve(encoder, scope=scope)
	nc = func(transport, fo, speed_info=speed_info, key_info=key_info, instrument_activities=instrument_activities, ctx=ctx)
	print(f"Saved to {fo}, Final note count: {nc}")
	return fo

def save_file(events, fo, ctx) -> str:
	if events is None:
		return
	if not isinstance(events, Transport):
		events = to_numpy(events)
		speed_info = midi.get_step_speed(events, ctx=ctx)
		transport, _nc, instrument_activities, speed_info = midi.deconstruct(events, speed_info, ctx=ctx)
	else:
		transport = events
		instrument_activities, speed_info = transport.get_metadata()
	key_info = transpose(transport, ctx)
	return export(transport, fo, instrument_activities, speed_info, key_info, ctx=ctx)

class ContextArgs:
	def get(self, k, default=None):
		return getattr(self, k, default)

def fix_args(ctx) -> ContextArgs:
	if not ctx.output:
		ctx.output = [ctx.input[0].rsplit(".", 1)[0]]
	formats = [get_ext(fo) if "." in fo else ctx.get("format", "nbs") for fo in ctx.output]
	if not ctx.get("format"):
		ctx.format = formats[0]
	for i, fo in enumerate(ctx.output):
		fo = fo.replace("\\", "/")
		if fo.endswith("/"):
			pass
		elif os.path.exists(fo) and os.path.isdir(fo):
			fo = fo + "/"
		elif "." not in fo:
			ctx.output[i] = f"{fo}.{formats[i]}"

	if ctx.get("mixing") is None:
		ctx.mixing = "IL"
	if ctx.get("volume") is None:
		ctx.volume = 1
	if ctx.get("speed") is None:
		ctx.speed = 1
	if ctx.get("resolution") is None:
		ctx.resolution = 12 if ctx.format in ("🗿", "moai", "skysheet", "genshinsheet") else 20 if ctx.format in ("nbt", "mcfunction", "litematic") else 40
	if ctx.get("strict_tempo") is None:
		ctx.strict_tempo = ctx.format in ("nbt", "mcfunction", "litematic")
	if ctx.get("transpose") is None:
		ctx.transpose = 0
	if ctx.get("invert_key") is None:
		ctx.invert_key = False
	if ctx.get("microtones") is None:
		ctx.microtones = ctx.format not in ("nbt", "mcfunction", "litematic", "org", "skysheet", "genshinsheet")
	if ctx.get("accidentals") is None:
		ctx.accidentals = ctx.format not in ("skysheet", "genshinsheet")
	if not ctx.accidentals:
		ctx.microtones = False
	if ctx.get("drums") is None:
		ctx.drums = True
	if ctx.get("max_distance") is None:
		ctx.max_distance = 42
	if ctx.get("minecart_improvements") is None:
		ctx.minecart_improvements = False
	if ctx.get("one_to_one") is None:
		ctx.one_to_one = False
	return ctx

def probe_paths(path, modes, depth=0) -> tuple:
	mode = modes[depth] if depth < len(modes) else modes[-1]
	if mode == "I":
		children = get_children(path)
		while len(children) == 1 and os.path.isdir(children[0]):
			children = get_children(children[0])
		if len(children) == 1:
			if depth:
				return children[0], [children[0]]
			return [children[0]], [children[0]]
		inputsi = []
		flati = []
		for c in children:
			inputs, flat = probe_paths(c, modes, depth + 1)
			inputsi.append(inputs)
			flati.extend(flat)
		return inputsi, flati
	elif mode in ("L", "C"):
		children = get_children(path)
		while len(children) == 1 and os.path.isdir(children[0]):
			children = get_children(children[0])
		if len(children) == 1:
			if depth:
				return children[0], [children[0]]
			return [children[0]], [children[0]]
		inputsi = []
		for c in children:
			inputs, flat = probe_paths(c, modes, depth + 1)
			inputsi.append(inputs)
		return inputsi, [path]
	raise NotImplementedError(mode)

def process_single(paths, ctx, depth=0) -> list:
	if type(paths) is str:
		return [load_file(paths, ctx)]
	return process_level(paths, ctx, min(depth + 1, len(ctx.mixing) - 1))

def process_level(paths, ctx, depth=0, executor=None) -> list:
	mode = ctx.mixing[depth]
	assert type(paths) is list, paths
	if len(paths) == 1:
		children = process_single(paths[0], ctx, depth)
	elif executor is not None:
		futures = [executor.submit(process_single, p, ctx, depth) for p in paths]
		children = itertools.chain.from_iterable(f.result() for f in futures)
	else:
		children = list(itertools.chain.from_iterable(process_single(p, ctx, depth) for p in paths))
	if type(children) is list and len(children) == 1:
		return children
	if mode == "I":
		return children
	if mode == "L":
		return [midi.stack_midis(children)]
	if mode == "C":
		return [midi.concat_midis(children)]
	raise NotImplementedError(mode)

def convert_files(**kwargs) -> list:
	ctx = ContextArgs()
	ctx.__dict__.update(kwargs)
	ctx = fix_args(ctx)
	inputs, outputs = probe_paths(ctx.input[0], ctx.mixing)
	count = len(outputs)
	archive_path = None
	if len(ctx.output) == 1 and get_ext(ctx.output[0]) in archive_formats:
		archive_path = ctx.output[0]
		ctx.output = [temp_dir + str(ts_us()) + "/"]
	out_format = ctx.get("format") or "mid"
	output_files = list(ctx.output)
	if len(output_files) == 1:
		if output_files[0].endswith("/"):
			if not os.path.exists(output_files[0]):
				os.mkdir(output_files[0])
			output_files = [output_files[0] + (file.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0] or "untitled") + "." + out_format for file in outputs]
		elif count > 1:
			raise ValueError('Expected new (suffixed with "/") or empty folder for multiple outputs.')
	elif count > 1:
		raise ValueError(f"Expected 1 or {count} outputs, got {len(output_files)}.")
	if count > 1:
		assert len(output_files) == count, f"Expected {count} outputs, got {len(output_files)}"
	filecount = count_leaves(inputs)
	print(f"Total input files: {filecount}, total inputs: {len(inputs)}, total outputs: {len(output_files)}")
	ctx.one_to_one = filecount == len(inputs)
	if ctx.one_to_one and len(inputs) == len(output_files) and all(fo.rsplit(".", 1)[-1] in "wav | flac | mp3 | aac | ogg | opus | m4a | weba | webm".split(" | ") for fo in output_files):
		from . import pcm
		removed = set()
		for i, (fi, fo) in enumerate(zip(inputs, output_files)):
			ofmt = fo.rsplit(".", 1)[-1]
			ofi = ".flac" if ofmt != "wav" else "." + ofmt
			match fi.rsplit(".", 1)[-1]:
				case "mid" | "midi":
					tmpl = temp_dir + str(ts_us()) + "1"
					out = tmpl + ofi
					pcm.render_midi([fi], [out], fmt=ofi[1:])
					pcm.mix_raw([out], fo)
					removed.add(i)
				case "nbs":
					tmpl = temp_dir + str(ts_us()) + "1"
					out = tmpl + ofi
					pcm.render_nbs([fi], [out], fmt=ofi[1:])
					pcm.mix_raw([out], fo)
					removed.add(i)
				case "org":
					tmpl = temp_dir + str(ts_us()) + "2"
					if ofi == ".flac":
						ofi = ".wav"
					out = tmpl + ofi
					pcm.render_org([fi], [out], fmt=ofi[1:])
					pcm.mix_raw([out], fo)
					removed.add(i)
				case "xm":
					tmpl = temp_dir + str(ts_us()) + "99"
					out = tmpl + ofi
					pcm.render_xm([fi], [out], fmt=ofi[1:])
					pcm.mix_raw([out], fo)
					removed.add(i)
		inputs = [fi for i, fi in enumerate(inputs) if i not in removed]
		output_files = [fo for i, fo in enumerate(output_files) if i not in removed]
	if len(inputs) == len(output_files) == 1:
		results = process_level(inputs, ctx=ctx)
		outputs = [save_file(results[0], output_files[0], ctx=ctx)]
	elif len(inputs):
		with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
			results = process_level(inputs, ctx=ctx, executor=executor)
			futures = []
			if type(results) is list and len(results) == 1 and len(output_files) > 1:
				midi_events = results[0]
				for fo in output_files:
					futures.append(executor.submit(save_file, midi_events, fo, ctx=ctx))
			else:
				for midi_events, fo in zip(results, output_files):
					futures.append(executor.submit(save_file, midi_events, fo, ctx=ctx))
			outputs = [fut.result() for fut in futures]
	if archive_path:
		outputs = [create_archive(ctx.output[0], archive_path)]
	return outputs


def main():
	parser = get_parser()
	args = parser.parse_args()
	import time
	t = time.time()
	outputs = convert_files(**vars(args))
	taken = time.time() - t
	print(f"{len(outputs)} files rendered in {round_min(round(taken, 3))} second(s).")
	clear_cache()

def clear_cache():
	try:
		import shutil
		return shutil.rmtree(temp_dir)
	except Exception:
		from traceback import print_exc
		print_exc()
		raise


if __name__ == "__main__":
	main()