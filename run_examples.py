import os
import subprocess
import sys
from hyperchoron import util
from hyperchoron.mappings import legal_inputs


def run_conversion(ctx, fi, *fo):
	args = [sys.executable, "hyperchoron.py", "-i", fi, "-o", *fo, "-t", str(ctx.transpose), "-s", str(ctx.speed), "-sa", str(ctx.strum_affinity), "-md", str(ctx.max_distance)]
	if not ctx.drums:
		args.append("--no-drums")
	if ctx.mc_legal:
		args.append("-ml")
	if ctx.invert_key:
		args.append("-ik")
	if ctx.minecart_improvements:
		args.append("-mi")
	if ctx.command_blocks:
		args.append("-cb")
	print(args)
	return subprocess.Popen(args)
def wait_procs(procs, max_workers=12):
	while len(procs) >= max_workers:
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
		ctx.input = ["examples/midi/" + f for f in os.listdir("examples/midi")]
	else:
		for fn in ctx.input:
			if os.path.isdir(fn):
				ctx.input.remove(fn)
				ctx.input.extend(fn + "/" + f for f in os.listdir(fn))
	if not ctx.output:
		ctx.output = ["examples/nbs"]
	for fold in ctx.output:
		if not os.path.exists(fold):
			os.mkdir(fold)
	fmts = [fold.rsplit("/", 1)[-1] for fold in ctx.output]
	# min_timestamp = os.path.getmtime("hyperchoron.py")
	for fn in sorted(ctx.input, key=lambda fn: (fn.endswith(".zip"), os.path.getsize(fn)), reverse=True):
		if fn.rsplit(".", 1)[-1] not in legal_inputs:
			print(f"WARNING: File {repr(fn)} has unrecognised extension, skipping...")
			continue
		names = [fn.rsplit(".", 1)[0] + "." + fmt for fmt in fmts]
		fo = [fold + "/" + n.replace("\\", "/").rsplit("/", 1)[-1] for fold, n in zip(ctx.output, names)]
		if any(not os.path.exists(f) or not os.path.getsize(f) or os.path.getmtime(f) < os.path.getmtime(fn) for f in fo):
			wait_procs(procs)
			procs.append(run_conversion(ctx, fn, *fo))
	for proc in procs:
		proc.wait()


if __name__ == "__main__":
	parser = util.get_parser()
	args = parser.parse_args()
	convert_files(args)
