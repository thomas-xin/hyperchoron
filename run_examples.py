import os
import subprocess
import sys

def run_conversion(ctx, fi, *fo):
	args = [sys.executable, "hyperchoron.py", "-i", fi, "-o", *fo, "-t", str(ctx.transpose), "-s", str(ctx.speed), "-sa", str(ctx.strum_affinity)]
	if not ctx.drums:
		args.append("--no-drums")
	if ctx.exclusive:
		args.append("-x")
	if ctx.invert_key:
		args.append("-ik")
	print(args)
	return subprocess.Popen(args)
def wait_procs(procs):
	while len(procs) >= 12:
		try:
			procs[0].wait(timeout=2)
		except subprocess.TimeoutExpired:
			pass
		for i in range(len(procs) - 1, -1, -1):
			if procs[i].poll() is not None:
				procs.pop(i)

def convert_files(ctx):
	procs = []
	if not ctx.input:
		ctx.input = "examples/midi"
	if not ctx.output:
		ctx.output = ["examples/litematic", "examples/mcfunction", "examples/nbs"]
	for fold in ctx.output:
		if not os.path.exists(fold):
			os.mkdir(fold)
	fmts = [fold.rsplit("/", 1)[-1] for fold in ctx.output]
	min_timestamp = os.path.getmtime("hyperchoron.py")
	for fn in sorted(os.listdir(ctx.input), key=lambda fn: (fn.endswith(".zip"), os.path.getsize(f"{ctx.input}/{fn}")), reverse=True):
		if fn.rsplit(".", 1)[-1] not in ("mid", "midi", "nbs", "zip"):
			print(f"WARNING: File {repr(fn)} has unrecognised extension, skipping...")
			continue
		names = [fn.rsplit(".", 1)[0] + "." + fmt for fmt in fmts]
		fi = f"{ctx.input}/{fn}"
		fo = [f"{fold}/{n}" for fold, n in zip(ctx.output, names)]
		if any(not os.path.exists(f) or not os.path.getsize(f) or os.path.getmtime(f) < max(min_timestamp, os.path.getmtime(fi)) for f in fo):
			wait_procs(procs)
			procs.append(run_conversion(ctx, fi, *fo))
	for proc in procs:
		proc.wait()


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		prog="",
		description="Hyperchoron Multi-Input",
	)
	parser.add_argument("-i", "--input", nargs="+", help="Input file (.mid | .zip | .nbs | .csv)")
	parser.add_argument("-o", "--output", nargs="*", help="Output file (.mcfunction | .litematic | .nbs | .org | .csv | .mid)")
	parser.add_argument("-s", "--speed", nargs="?", type=float, default=1, help="Scales song speed up/down as a multiplier, applied before tempo sync; higher = faster. Defaults to 1")
	parser.add_argument("-t", "--transpose", nargs="?", type=int, default=0, help="Transposes song up/down a certain amount of semitones, applied before instrument material mapping; higher = higher pitched. Defaults to 0")
	parser.add_argument("-ik", "--invert-key", action=argparse.BooleanOptionalAction, default=False, help="Experimental: During transpose step, autodetects song key signature, then inverts it (e.g. C Major <=> C Minor). Defaults to FALSE")
	parser.add_argument("-sa", "--strum-affinity", nargs="?", default=1, type=float, help="Increases or decreases threshold for sustained notes to be cut into discrete segments; higher = more notes. Defaults to 1")
	parser.add_argument("-d", "--drums", action=argparse.BooleanOptionalAction, default=True, help="Allows percussion channel. If disabled, the default MIDI percussion channel will be treated as a regular instrument channel. Defaults to TRUE")
	parser.add_argument("-c", "--cheap", action=argparse.BooleanOptionalAction, default=False, help="Restricts the list of non-instrument blocks to a more survival-friendly set. Also enables compatibility with previous versions of Minecraft. May cause spacing issues with the sand/snare drum instruments. Defaults to FALSE")
	parser.add_argument("-x", "--exclusive", action=argparse.BooleanOptionalAction, default=None, help="Disables speed re-matching and strum quantisation, increases pitch bucket limit. Defaults to FALSE if outputting to any Minecraft-related format, and included for compatibility with other export formats.")
	args = parser.parse_args()
	convert_files(args)
