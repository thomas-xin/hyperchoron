# Coco eats a gold block on the 18/2/2025. Nom nom nom nom. Output sound weird? Sorgy accident it was the block I eated. Rarararrarrrr 😋

try:
	import tqdm
except ImportError:
	tqdm = None
else:
	import warnings
	warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
from hyperchoron import util


def export(transport, instrument_activities, speed_info, ctx=None):
	nc = 0
	for output in ctx.output:
		ext = output.rsplit(".", 1)[-1].casefold()
		match ext:
			case "nbs":
				from hyperchoron import minecraft
				nc += minecraft.save_nbs(transport, output, speed_info, ctx=ctx)
			case "mid" | "midi" | "csv":
				from hyperchoron import midi
				nc += midi.save_midi(transport, output, instrument_activities=instrument_activities, speed_info=speed_info, ctx=ctx)
			case "org":
				from hyperchoron import tracker
				nc += tracker.save_org(transport, output, instrument_activities=instrument_activities, speed_info=speed_info, ctx=ctx)
			case "nbt" | "mcfunction" | "litematic":
				from hyperchoron import minecraft
				nc += minecraft.save_litematic(transport, output, ctx=ctx)
			case "🗿" | "moai":
				from hyperchoron import text
				nc += text.save_thirtydollar(transport, output, speed_info=speed_info, ctx=ctx)
			case "zip":
				from hyperchoron import text
				nc += text.save_deltarune(transport, output, instrument_activities=instrument_activities, speed_info=speed_info, ctx=ctx)
			case _:
				from hyperchoron import dawvert
				nc += dawvert.save_arbitrary(transport, output, instrument_activities=instrument_activities, speed_info=speed_info, ctx=ctx)
	print("Final note count:", nc)

def convert_file(args):
	ctx = args
	default_resolution = 40
	fmt = ""
	if ctx.output and ((fmt := ctx.output[0].rsplit(".", 1)[-1].casefold()) in ("litematic", "mcfunction", "nbt")):
		if ctx.mc_legal is None:
			print("Auto-switching to Minecraft-Legal mode...")
			ctx.mc_legal = True
	if fmt == "🗿":
		default_resolution = 12
	elif ctx.mc_legal:
		default_resolution = 20
	if not ctx.resolution:
		ctx.resolution = default_resolution
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
		if isinstance(file, str) and (file.startswith("https://") or file.startswith("http://")):
			import urllib.request
			req = urllib.request.Request(file, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"})
			name = file.split("?", 1)[0].rsplit("/", 1)[-1].replace("__", "_")
			fn = util.temp_dir + name
			with open(fn, "wb") as f:
				f.write(urllib.request.urlopen(req).read())
			file = fn
		elif isinstance(file, str):
			name = file.replace("\\", "/").rsplit("/", 1)[-1]
		else:
			name = file.name.replace("\\", "/").rsplit("/", 1)[-1]
		ext = name.rsplit(".", 1)[-1].casefold()
		match ext:
			case "nbs":
				from hyperchoron import minecraft
				data = minecraft.load_nbs(file)
			case "csv":
				from hyperchoron import midi
				data = midi.load_csv(file)
			case "org":
				from hyperchoron import tracker
				data = tracker.load_org(file)
			case "xm":
				from hyperchoron import tracker
				data = tracker.load_xm(file)
			case "mid" | "midi":
				from hyperchoron import midi
				data = midi.load_midi(file)
			case "wav" | "flac" | "mp3" | "aac" | "ogg" | "opus" | "m4a" | "wma" | "weba" | "webm":
				from hyperchoron import pcm
				data = pcm.load_wav(file, ctx=ctx)
			case "🗿" | "moai":
				from hyperchoron import text
				data = text.load_thirtydollar(file)
			case _:
				from hyperchoron import dawvert
				data = dawvert.load_arbitrary(file, ext)
		imported.append(data)
	transport, instrument_activities, speed_info, note_candidates = util.merge_imports(imported, ctx)
	util.transpose(transport, ctx)
	print("Note candidates:", note_candidates)
	print("Note segment count:", sum(map(len, transport)))
	print("Max detected polyphony:", max(map(len, transport), default=0))
	# print("Lowest note:", min(min(n[1] for n in b) for b in transport if b))
	# print("Highest note:", max(max(n[1] for n in b) for b in transport if b))
	export(transport, instrument_activities, speed_info, ctx=ctx)


def main():
	parser = util.get_parser()
	args = parser.parse_args()
	convert_file(args)

if __name__ == "__main__":
	main()
