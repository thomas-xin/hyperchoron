import functools
from math import cbrt, erf
from .mappings import (
	instrument_names, midi_instrument_selection,
	fixed_instruments, instrument_codelist,
	thirtydollar_names, thirtydollar_volumes,
	nbs2thirtydollar, thirtydollar_unmap, thirtydollar_drums,
	percussion_mats, material_map, pitches, default_instruments,
	fs1, fs4,
)
from .util import round_min, log2lin, lin2log, temp_dir, base_path


@functools.lru_cache(maxsize=256)
def get_note_mat_ex(note, odd=False):
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
	if mat.endswith("+"):
		mat = mat[:-1]
		mod += 12
	return mat, mod


def load_thirtydollar(file):
	print("Importing 🗿...")
	with open(file, "r", encoding="utf-8") as f:
		actions = f.read().replace("\n", "|").split("|")
	events = [
		[0, 0, "header", 1, 9 + 1, 1],
		[1, 0, "tempo", 1000 * 1000 / (300 / 60)],
	]
	for i in range(len(midi_instrument_selection) - 1):
		events.append([i + 2, 0, "program_c", i + 10, midi_instrument_selection[i]])
	tick = 0
	vm = 100
	pm = 0
	for action in actions:
		match action:
			case _ if action.startswith("!speed@"):
				action = action.split("@", 1)[-1]
				speed = float(action)
				events.append([1, 0, "tempo", 1000 * 1000 / (speed / 60)])
			case _ if action.startswith("!stop@"):
				action = action.split("@", 1)[-1]
				stop = float(action)
				tick += stop
			case _ if action.startswith("!volume@"):
				action = action.split("@", 1)[-1]
				add = action.endswith("+")
				if add:
					action = action[:-1]
					vm += float(action)
				else:
					vm = float(action)
			case _ if action.startswith("!transpose@"):
				action = action.split("@", 1)[-1]
				add = action.endswith("+")
				if add:
					action = action[:-1]
					pm += float(action)
				else:
					pm = float(action)
			case "!combine":
				tick -= 1
			case "_pause":
				tick += 1
			case _:
				rep = 1
				velocity = 100
				pitch = 0
				vol = 1
				ins = None
				pitch_override = None
				if "=" in action:
					action, rep = action.rsplit("=", 1)
				if "%" in action:
					action, velocity = action.rsplit("%", 1)
				if "@" in action:
					action, pitch = action.rsplit("@", 1)
				try:
					tup = thirtydollar_unmap[action]
					if len(tup) >= 3:
						ins, po, vol = tup
					else:
						ins, po = tup
					po += 24
				except KeyError:
					if action in thirtydollar_drums:
						ins, pitch_override = thirtydollar_drums[action]
					elif action.startswith("noteblock_"):
						action = action.split("_", 1)[-1]
						ins = instrument_codelist.index(default_instruments[action])
						po = pitches[action]
				if ins is not None:
					if pitch_override is None:
						pitch_override = float(pm) + float(pitch) + float(po) + fs1 + 12
					elif pitch_override == -1:
						tick += int(rep)
						continue
					for i in range(int(rep)):
						note_event = [int(ins) + 2, tick, "note_on_c", int(ins) + 10, pitch_override, lin2log(float(vm) / 100 * float(velocity) / 100) / float(vol) * 127, 1, 1]
						events.append(note_event)
						tick += 1
	events.sort(key=lambda e: e[1])
	return events

def save_thirtydollar(transport, output, speed_info, ctx):
	print("Exporting 🗿...")
	orig_ms_per_clock, real_ms_per_clock, scale, orig_step_ms, _orig_tempo = speed_info
	speed_ratio = real_ms_per_clock / scale / orig_ms_per_clock
	wait = max(1, round(orig_step_ms / speed_ratio))
	bpm = 60 * 1000 / wait
	nc = 0
	events = [f"!speed@{round_min(round(bpm, 2))}"]
	div = 0
	for i, beat in enumerate(transport):
		temp = []
		for note in sorted(beat, key=lambda note: note[0]):
			itype, pitch, priority, _long, vel, pan = note
			if priority < 0:
				continue
			name = thirtydollar_names[itype]
			if name.startswith("noteblock_"):
				mat, mod = get_note_mat_ex(note, odd=i & 1)
				match mat:
					case  "PLACEHOLDER":
						continue
					case "cobblestone":
						name = "🥁"
					case "black_concrete_powder":
						name = "hammer"
					case "creeper_head":
						name = "celeste_diamond"
						mod += 12
					case "skeleton_skull":
						name = "celeste_spring"
						mod += 12
					case "obsidian" | "netherrack":
						name = "🪘"
					case "dropper":
						name = "hitmarker"
						mod += 12
					case _ if "stained_glass" in mat:
						name = "rdclap"
						mod += 12
					case _ if "fence_gate" in mat or "trapdoor" in mat:
						name = "skipshot"
						mod += 12
					case _:
						instrument = instrument_names[mat]
						if itype != -1:
							if not ctx.mc_legal:
								instrument2 = fixed_instruments[instrument_codelist[itype]]
								pitch2 = note[1] - pitches[instrument2] - fs1
								if pitch2 in range(-12, 48):
									mod = pitch2
									instrument = instrument2
								else:
									pitch2 = note[1] - pitches[instrument] - fs1
									if pitch2 in range(-33, 55):
										mod = pitch2
						if instrument in nbs2thirtydollar:
							name = nbs2thirtydollar[instrument]
						else:
							name = f"noteblock_{instrument}"
				pitch = mod + fs4 - 12
			p = pitch - fs4
			# Get rid of instruments that don't exist
			if name == "noteblock_iron_xylophone":
				name = "noteblock_harp"
			elif name == "noteblock_cowbell":
				name = "noteblock_xylophone"
				p -= 12
			elif name == "meowsynth":
				p += 6
			elif name == "🦴":
				p += 16
			elif name == "noteblock_didgeridoo":
				name = "noteblock_harp"
				p -= 24
				if p < -24:
					name = "noteblock_bass"
					p += 24
			if name in ("stylophone", "fnf_up", "mariopaint_flower") and p < -12:
				name = "fnf_down"
				p += 12
				if p < -12:
					name = "noteblock_harp"
					p -= 12
					if p < -24:
						name = "noteblock_bass"
						p += 24
			elif name == "meowsynth" and p < -12:
				name = "🦴"
				p += 24 - 8 - 6
			elif name == "🦴" and p >= 12:
				name = "amogus_emergency"
				p -= 16 - 7.5
			elif name == "noteblock_bit" and p >= 24:
				name = "amogus_emergency"
				p -= -7.5
			if name == "stylophone":
				p += 1 / 3
			text = name
			if p != 0:
				text += f"@{round_min(round(p, 2))}"
			v = log2lin(vel / 127) * 100
			if priority == 0:
				v *= min(1, 0.9 ** (50 / wait))
			v *= thirtydollar_volumes.get(name, 1)
			v = round(v)
			if v != 0:
				text += f"%{v}"
			temp.append(text)
		if temp:
			events.append("|!combine|".join(temp))
		elif events[-1].startswith("!stop@") and not events[-1].endswith("\n"):
			n = int(events[-1].split("@", 1)[-1])
			events[-1] = f"!stop@{n + 1}"
		elif events[-1] == "_pause":
			events[-1] = "!stop@2"
		else:
			events.append("_pause")
		if i & 15 == 15 and i != len(transport) - 1:
			events[-1] += "|!divider\n"
			div += 1
		nc += len(temp)
	with open(output, "w", encoding="utf-8") as thirtydollar:
		thirtydollar.write("|".join(events))
	return nc


def save_deltarune(transport, output, instrument_activities, speed_info, ctx):
	tmpl = output.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
	full_midi = f"{temp_dir}{tmpl}-full.mid"
	from hyperchoron import midi
	instruments, wait, resolution = list(midi.build_midi(transport, instrument_activities, speed_info, ctx=ctx))
	nc = midi.proceed_save_midi(full_midi, f"{tmpl}-full", False, instruments, wait, resolution)
	all_notes = []
	for ins in instruments:
		all_notes.extend(ins.notes)
	all_notes.sort(key=lambda note: (note.tick, note.instrument_type in (29, 30, 31, 9, -1), -cbrt(note.pitch) * (min(16, note.length / resolution) * erf((100 - note.volume) / 40) + note.volume)))
	max_duration = all_notes[-1].aligned
	seconds_per_tick = 1000000 / wait * resolution
	special = False
	drum_pos = [0, 0, 0]
	lead_pos = [0, 0]
	lead_pitch = -1
	last_pos = -1
	vocals_pos = 0
	vocals_pitch = -1
	lead_direction = 0
	vocals_direction = 1
	lead = []
	vocals = []
	drums = []
	for note in all_notes:
		if note.instrument_type == -1:
			special = 0
			if note.pitch in (27, 32, 33, 35, 36, 41, 42, 43, 44, 45, 47, 48, 50, 60, 61, 62, 64, 65, 66, 85, 86, 87):
				direction = 0
			elif note.pitch in (28, 29, 30, 31, 37, 38, 39, 40, 58, 63, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 79, 82):
				direction = 1
			elif note.pitch in (49, 51, 52, 57, 59, 81, 83, 84):
				direction = 2
			else:
				direction = 1
				special = 1
			if drum_pos[direction] > note.tick:
				continue
			duration = (round_min(round(note.aligned / seconds_per_tick, 3)) if note.pitch in (49, 57, 58, 84) else 0)
			rhythm_note = [
				round_min(round(note.tick / seconds_per_tick, 3)),
				direction,
				duration,
				special,
			]
			drum_pos[direction] = note.tick + max(duration, resolution) / resolution
			drums.append(rhythm_note)
			instrument = instruments[note.instrument_id]
			continue
		direction = int(not lead_direction if abs(note.pitch - lead_pitch) >= 1 else lead_direction)
		if lead_pos[direction] > note.tick and lead_pos[not direction] <= note.tick and note.volume >= 127:
			direction = int(not direction)
		if lead_pos[direction] > note.tick or (note.instrument_type == 9 or note.volume < 127) and lead_pos[not direction] > note.tick:
			diff = note.pitch - vocals_pitch
			if diff >= 5 or diff >= 2 and vocals_direction >= 1:
				direction = 2
			elif diff <= -5 or diff <= -2 and vocals_direction <= 1:
				direction = 0
			elif diff:
				direction = 1
			else:
				direction = vocals_direction
			if vocals_pos > note.tick:
				continue
			vocals_direction = direction
			instrument = instruments[note.instrument_id]
			rhythm_note = [
				round_min(round(note.tick / seconds_per_tick, 3)),
				direction,
				round_min(round(note.aligned / seconds_per_tick, 3)),
				int(note.length >= resolution * 16 and note.volume >= 127),
			]
			vocals.append(rhythm_note)
			vocals_pitch = note.pitch
			vocals_pos = note.tick + note.length
			continue
		lead_direction = direction
		instrument = instruments[note.instrument_id]
		special = int(not special and (note.length >= resolution * 16 or note.volume >= 127))
		pos = round_min(round(note.tick / seconds_per_tick, 3))
		if last_pos == pos:
			lead[-1][2] = round_min(round((note.tick - resolution) / seconds_per_tick, 3))
		duration = (round_min(round(note.aligned / seconds_per_tick, 3)) if note.length >= resolution * 4 and instrument.sustain else 0)
		last_pos = pos + duration if duration > 0 else 0
		rhythm_note = [
			pos,
			direction,
			duration,
			special,
		]
		lead.append(rhythm_note)
		lead_pitch = note.pitch
		lead_pos[direction] = note.tick + (note.length if instrument.sustain else max(resolution * 2, min(note.length, resolution * 24)))
		instrument.notes.remove(note)
	# import json
	# lead_json = json.dumps(lead)
	# vocals_json = json.dumps(vocals)
	# drums_json = json.dumps(drums)
	lead_fn = "\n".join(f"\ta({n[0]},{n[1]},{n[2]},{n[3]});" for n in lead)
	vocals_fn = "\n".join(f"\ta({n[0]},{n[1]},{n[2]},{n[3]});" for n in vocals)
	drums_fn = "\n".join(f"\ta({n[0]},{n[1]},{n[2]},{n[3]});" for n in drums)
	code = [
		"""function scr_rhythmgame_notechart(arg0 = "lead", arg1 = 0) {
	script_execute("scr_rhythmgame_notechart_" + arg0, arg1);
}""",
		"function scr_rhythmgame_notechart_lead(arg0 = 0) {",
		"\tvar a = scr_rhythmgame_addnote;",
		lead_fn,
		"}",
		"function scr_rhythmgame_notechart_vocals(arg0 = 0) {",
		"\tvar a = scr_rhythmgame_addnote;",
		vocals_fn,
		"}",
		"function scr_rhythmgame_notechart_drums(arg0 = 0) {",
		"\tvar a = scr_rhythmgame_addnote;",
		drums_fn,
		"}",
		"function scr_rhythmgame_notechart_lyrics(arg0 = 0) {}",
		"function scr_rhythmgame_notechart_lead_solo(arg0) {}",
		"function scr_rhythmgame_notechart_lead_finale() {}",
		"""function scr_rhythmgame_notechart_clear() {
	notetime = [];
	notetype = [];
	noteend = [];
	noteanim = [];
	notealive = [];
	notescore = [];
	maxnote = 0;
	minnote = 0;
}""",
	]
	rhythmgame_notes = f"{temp_dir}{tmpl}-rhythmgame_notechart.gml"
	with open(rhythmgame_notes, "w") as f:
		f.write("\n".join(code))
	code = [
		"""function scr_rhythmgame_load_song(arg0 = 0, arg1 = true, arg2 = true, arg3 = false) {
	if (song_loaded && !arg3) scr_rhythmgame_song_reset();
	arg0 = 5;
	obj_rhythmgame.song_id = arg0;
	song_id = arg0;
	notetime[0] = 0;
	notespeed = 150;
	loop = false;
	scr_debug_print("trying to load");
""" + f"""
	track1_id = "{tmpl}-base.ogg";
	track2_id = "{tmpl}-full.ogg";
	bpm = {60000000 / wait / resolution / 4};
	notespacing = 60 / bpm;
	meter = notespacing * 4;
	trackstart = scr_round_to_beat(-1 * (240 / bpm), bpm);
	oneAtATime = false;
	track_length = {max_duration / seconds_per_tick};
	notespeed = 144;
""" + """
	if (arg3) {
	} else if (arg0 >= 0) {
		scr_debug_print("loading da other stuff");
		scr_rhythmgame_notechart_lead(arg0);
		if (tutorial > 0)
			scr_rhythmgame_toggle_notes(false);
		musicm.bpm = bpm;
		musicm.beat_offset += bpm / 2;
		if (arg1) {
			with (drums) {
				scr_rhythmgame_notechart_drums(arg0);
				bpm = other.bpm;
				notespacing = other.notespacing;
				meter = other.meter;
				loop = other.loop;
			}
		} if (arg2) {
			with (vocals) {
				scr_rhythmgame_notechart_vocals(arg0);
				bpm = other.bpm;
				notespacing = other.notespacing;
				meter = other.meter;
				loop = other.loop;
				if (i_ex(obj_rhythmgame)) {
					if (maxnote == 0 || chart_start > notetime[0] || (chart_start + (meter * 2)) < notetime[0]) performer.sprite_index = spr_ralsei_sing_clap;
				}
			}
		}
		scr_rhythmgame_load_events(arg0);
	}
}""",
	]
	rhythmgame_load = f"{temp_dir}{tmpl}-rhythmgame_song_load.gml"
	with open(rhythmgame_load, "w") as f:
		f.write("\n".join(code))
	base_midi = f"{temp_dir}{tmpl}-base.mid"
	midi.proceed_save_midi(base_midi, f"{tmpl}-base", False, instruments, wait, resolution)
	import os
	import subprocess
	base_flac = f"{temp_dir}{tmpl}-base.flac"
	args = [os.path.abspath(base_path + "/fluidsynth/fluidsynth"), "-g", "1", "-F", base_flac, "-c", "64", "-r", "48000", "-n", os.path.abspath(base_path + "/fluidsynth/gm64.sf2"), base_midi]
	subprocess.run(args)
	base_ogg = f"{temp_dir}{tmpl}-base.ogg"
	args = ["ffmpeg", "-y", "-i", base_flac, "-c:a", "libvorbis", "-b:a", "128k", base_ogg]
	proc = subprocess.Popen(args)
	full_flac = f"{temp_dir}{tmpl}-full.flac"
	args = [os.path.abspath(base_path + "/fluidsynth/fluidsynth"), "-g", "1", "-F", full_flac, "-c", "64", "-r", "48000", "-n", os.path.abspath(base_path + "/fluidsynth/gm64.sf2"), full_midi]
	subprocess.run(args)
	proc.wait()
	full_ogg = f"{temp_dir}{tmpl}-full.ogg"
	args = ["ffmpeg", "-y", "-i", full_flac, "-c:a", "libvorbis", "-b:a", "128k", full_ogg]
	subprocess.run(args)
	import zipfile
	with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as z:
		z.write(rhythmgame_notes, arcname="rhythmgame_notechart.gml")
		z.write(rhythmgame_load, arcname="rhythmgame_song_load.gml")
		z.write(base_ogg, arcname=f"{tmpl}-base.ogg")
		z.write(full_ogg, arcname=f"{tmpl}-full.ogg")
	return nc