import os
from hyperchoron import convert_file

class Args:
	def __getattr__(self, k):
		return object.__getattribute__(self, "__dict__").get(k)

if not os.path.exists("examples/litematic"):
	os.mkdir("examples/litematic")
for fn in os.listdir("examples/midi"):
	name = fn.rsplit(".", 1)[0] + ".litematic"
	fi = f"examples/midi/{fn}"
	fo = f"examples/litematic/{name}"
	if not os.path.exists(fo) or not os.path.getsize(fo) or os.path.getmtime(fo) < os.path.getmtime(fi):
		args = Args()
		args.input = fi
		args.output = fo
		convert_file(args)

if not os.path.exists("examples/mcfunction"):
	os.mkdir("examples/mcfunction")
for fn in os.listdir("examples/midi"):
	name = fn.rsplit(".", 1)[0] + ".mcfunction"
	fi = f"examples/midi/{fn}"
	fo = f"examples/mcfunction/{name}"
	if not os.path.exists(fo) or not os.path.getsize(fo) or os.path.getmtime(fo) < os.path.getmtime(fi):
		args = Args()
		args.input = fi
		args.output = fo
		convert_file(args)