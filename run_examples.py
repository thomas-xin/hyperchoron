import os
import subprocess
import sys
from hyperchoron import util


def run_conversion(ctx, fi, *fo):
	args = [sys.executable, "hyperchoron.py", "-i", fi, "-o", *fo, "-t", str(ctx.transpose), "-s", str(ctx.speed), "-sa", str(ctx.strum_affinity), "-md", str(ctx.max_distance)]
	if not ctx.drums:
		args.append("--no-drums")
	if ctx.mc_legal:
		args.append("-ml")
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
		ctx.output = ["examples/nbs"]
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
	parser = util.get_parser()
	args = parser.parse_args()
	convert_files(args)
