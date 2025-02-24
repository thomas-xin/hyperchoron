# Coco eats a gold block on the 18/2/2025. Nom nom nom nom. Output sound weird? Sorgy accident it was the block I eated. Rarararrarrrr 😋

from math import inf
try:
	import tqdm
except ImportError:
	tqdm = None
else:
	import warnings
	warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
from lib import util


def export(transport, instrument_activities, speed_info, ctx=None):
	print("Saving...")
	block_replacements = {}
	if ctx.cheap:
		block_replacements.update()
	else:
		block_replacements.update()
	nc = 0
	for output in ctx.output:
		ext = output.rsplit(".", 1)[-1]
		match ext:
			case "nbs":
				from lib import minecraft
				nc += minecraft.save_nbs(transport, output, ctx=ctx)
			case "mid" | "midi" | "csv":
				from lib import midi
				nc += midi.save_midi(transport, output, instrument_activities=instrument_activities, speed_info=speed_info, ctx=ctx)
			case "org":
				from lib import tracker
				nc += tracker.save_org(transport, output, instrument_activities=instrument_activities, speed_info=speed_info, ctx=ctx)
			case "mcfunction":
				from lib import minecraft
				nc += minecraft.save_mcfunction(transport, output, ctx=ctx)
			case "litematic":
				from lib import minecraft
				nc += minecraft.save_litematic(transport, output, ctx=ctx)
			case _:
				from lib import dawvert
				nc += dawvert.save_arbitrary(transport, output, instrument_activities=instrument_activities, speed_info=speed_info, ctx=ctx)
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
	imported = []
	for file in inputs:
		if isinstance(file, str) and file.startswith("https://"):
			import io
			import urllib.request
			req = urllib.request.Request(file, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"})
			name = file.split("?", 1)[0].rsplit("/", 1)[-1]
			file = io.BytesIO(urllib.request.urlopen(req).read())
		elif isinstance(file, str):
			name = file.replace("\\", "/").rsplit("/", 1)[-1]
		else:
			name = file.name.replace("\\", "/").rsplit("/", 1)[-1]
		ext = name.rsplit(".", 1)[-1]
		match ext:
			case "nbs":
				from lib import minecraft
				data = minecraft.load_nbs(file)
			case "csv":
				from lib import midi
				data = midi.load_csv(file)
			case "org":
				from lib import tracker
				data = tracker.load_org(file)
			# case "xm":
			# 	from lib import tracker
			# 	data = tracker.load_xm(file)
			case "mid" | "midi":
				from lib import midi
				data = midi.load_midi(file)
			case _:
				from lib import dawvert
				data = dawvert.load_arbitrary(file, ext)
		imported.append(data)
	transport, instrument_activities, speed_info, note_candidates = util.merge_imports(imported, ctx)
	util.transpose(transport, ctx)
	print("Note candidates:", note_candidates)
	print("Note count:", sum(map(len, transport)))
	print("Max detected polyphony:", max(map(len, transport), default=0))
	print("Lowest note:", min(min(n[1] for n in b) for b in transport if b))
	print("Highest note:", max(max(n[1] for n in b) for b in transport if b))
	export(transport, instrument_activities, speed_info, ctx=ctx)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		prog="",
		description="MIDI-Tracker-DAW converter and Minecraft Note Block exporter",
	)
	parser.add_argument("-i", "--input", nargs="+", help="Input file (.zip | .mid | .csv | .nbs | .org | *)")
	parser.add_argument("-o", "--output", nargs="*", help="Output file (.mid | .csv | .nbs | .mcfunction | .litematic | .org | *)")
	parser.add_argument("-s", "--speed", nargs="?", type=float, default=1, help="Scales song speed up/down as a multiplier, applied before tempo sync; higher = faster. Defaults to 1")
	parser.add_argument("-t", "--transpose", nargs="?", type=int, default=0, help="Transposes song up/down a certain amount of semitones, applied before instrument material mapping; higher = higher pitched. Defaults to 0")
	parser.add_argument("-ik", "--invert-key", action=argparse.BooleanOptionalAction, default=False, help="Experimental: During transpose step, autodetects song key signature, then inverts it (e.g. C Major <=> C Minor). Defaults to FALSE")
	parser.add_argument("-sa", "--strum-affinity", nargs="?", default=1, type=float, help="Increases or decreases threshold for sustained notes to be cut into discrete segments; higher = more notes. Defaults to 1")
	parser.add_argument("-d", "--drums", action=argparse.BooleanOptionalAction, default=True, help="Allows percussion channel. If disabled, percussion channels will be treated as regular instrument channels. Defaults to TRUE")
	parser.add_argument("-c", "--cheap", action=argparse.BooleanOptionalAction, default=False, help="For Minecraft outputs: Restricts the list of non-instrument blocks to a more survival-friendly set. Also enables compatibility with previous versions of Minecraft. May cause spacing issues with the sand/snare drum instruments. Defaults to FALSE")
	parser.add_argument("-x", "--exclusive", action=argparse.BooleanOptionalAction, default=None, help="For non-Minecraft outputs: Disables speed re-matching and strum quantisation, increases pitch range limits. Defaults to TRUE.")
	args = parser.parse_args()
	convert_file(args)
