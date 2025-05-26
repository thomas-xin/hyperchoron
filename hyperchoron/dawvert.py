# ruff: noqa: E402
import os
from .util import base_path
dawvert_path = base_path.rsplit("/", 2)[0] + "/DawVert"
print(dawvert_path)
assert os.path.exists(dawvert_path)
import subprocess
import sys
import tempfile
from .mappings import dawvert_inputs, dawvert_outputs
from . import midi


def load_arbitrary(file, ext):
	assert ext in dawvert_inputs, f"Input format {ext} currently unsupported."
	if isinstance(file, str):
		file = open(file, "rb")
	with tempfile.NamedTemporaryFile(delete=False) as f:
		with file:
			f.write(file.read())
		fo = f.name + ".mid"
		args = [sys.executable, "dawvert_cmd.py", "-it", dawvert_inputs[ext], "-i", f.name, "-ot", "midi", "-o", fo]
		subprocess.check_output(args, cwd=dawvert_path)
		assert os.path.exists(fo) and os.path.getsize(fo), "Unable to locate output."
	if os.path.exists(f.name):
		os.remove(f.name)
	try:
		return midi.load_midi(fo)
	finally:
		if os.path.exists(fo):
			os.remove(fo)


def save_arbitrary(transport, output, instrument_activities, speed_info, ctx):
	ext = output.rsplit(".", 1)[-1]
	assert ext in dawvert_outputs, f"Output format {ext} currently unsupported."
	mid = os.path.abspath(output.rsplit(".", 1)[0] + ".mid")
	nc = midi.save_midi(transport, mid, instrument_activities, speed_info, ctx)
	if os.path.exists(output):
		os.remove(output)
	args = [sys.executable, "dawvert_cmd.py", "-it", "midi", "-i", mid, "-ot", dawvert_outputs[ext], "-o", os.path.abspath(output)]
	print(mid)
	subprocess.check_output(args, cwd=dawvert_path)
	if os.path.exists(mid):
		os.remove(mid)
	assert os.path.exists(output) and os.path.getsize(output), "Unable to locate output."
	return nc