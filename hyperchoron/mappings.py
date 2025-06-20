import math


falling_blocks = ("sand", "red_sand", "black_concrete_powder", "gravel")
# Predefined list attempting to match instruments across pitch ranges
material_map = [
	["bamboo_planks", "black_wool", "black_wool+", "amethyst_block+", "gold_block", "gold_block+"],
	["bamboo_planks", "bamboo_planks+", "glowstone", "glowstone+", "gold_block", "gold_block+"],
	["pumpkin", "pumpkin+", "amethyst_block", "clay", "clay+", "packed_ice+"],
	["pumpkin", "pumpkin+", "emerald_block", "emerald_block+", "gold_block", "gold_block+"],
	["bamboo_planks", "bamboo_planks+", "iron_block", "iron_block+", "soul_sand+", "bone_block+"],
	["bamboo_planks", "black_wool", "emerald_block", "emerald_block+", "packed_ice", "packed_ice+"],
	["netherrack", "netherrack+", "red_stained_glass", "red_stained_glass+", "red_sand", "red_sand+"],
	["bamboo_planks", "black_wool", "hay_block", "hay_block+", "gold_block", "gold_block+"],
	["bamboo_planks", "black_wool", "hay_block", "hay_block+", "gold_block", "gold_block+"],
	["pumpkin", "pumpkin+", "amethyst_block", "clay", "clay+", "packed_ice+"],
	["pumpkin", "pumpkin+", "black_wool+", "clay", "clay+", "packed_ice+"],
	["pumpkin", "pumpkin+", "emerald_block", "clay", "clay+", "gold_block+"],
	["pumpkin", "pumpkin+", "emerald_block", "emerald_block+", "packed_ice", "packed_ice+"],
	["bamboo_planks", "black_wool", "emerald_block", "emerald_block+", "gold_block", "gold_block+"],
	None
]
default_instruments = dict(
	harp="Plucked",						# 0
	pling="Piano",						# 1
	flute="Wind",						# 2
	bit="Square Synth",					# 3
	iron_xylophone="Pitched Percussion",# 4
	chime="Bell",						# 5
	basedrum="Unpitched Percussion",	# 6
	banjo="String",						# 7
	u1="Banjo",							# 8
	u2="Voice",							# 9
	didgeridoo="Brass",					#10
	u3="Saw Synth",						#11
	u4="Organ",							#12
	u5="Overdrive Guitar",				#13
	creeper="Drumset",					#-1
)
instrument_codelist = list(default_instruments.values())
default_instruments.update(dict(
	snare="Unpitched Percussion",
	hat="Unpitched Percussion",
	bell="Plucked",
	cow_bell="Pitched Percussion",
	guitar="Plucked",
	bass="Plucked",
	xylophone="Pitched Percussion",
))
fixed_instruments = {v: k for k, v in reversed(default_instruments.items())}
instrument_names = dict(
	amethyst_block="harp",
	bamboo_planks="bass",
	red_sand="snare",
	black_concrete_powder="snare",
	heavy_core="snare",
	blue_stained_glass="hat",
	red_stained_glass="hat",
	obsidian="basedrum",
	netherrack="basedrum",
	cobblestone="basedrum",
	gold_block="bell",
	clay="flute",
	packed_ice="chime",
	black_wool="guitar",
	bone_block="xylophone",
	iron_block="iron_xylophone",
	soul_sand="cow_bell",
	pumpkin="didgeridoo",
	emerald_block="bit",
	hay_block="banjo",
	glowstone="pling",
	skeleton_skull="skeleton",
	wither_skeleton_skull="wither_skeleton",
	zombie_head="zombie",
	creeper_head="creeper",
	piglin_head="piglin",
	warped_trapdoor="trapdoor",
	bamboo_trapdoor="trapdoor",
	oak_trapdoor="trapdoor",
	bamboo_fence_gate="fence_gate",
	dropper="dispenser",
)
nbs_names = {k: i for i, k in enumerate([
	"harp",
	"bass",
	"basedrum",
	"snare",
	"hat",
	"guitar",
	"flute",
	"bell",
	"chime",
	"xylophone",
	"iron_xylophone",
	"cow_bell",
	"didgeridoo",
	"bit",
	"banjo",
	"pling",
])}
nbs_values = {v: k for k, v in nbs_names.items()}
for unsupported in ("skeleton", "wither_skeleton", "zombie", "creeper", "piglin", "trapdoor", "fence_gate", "dispenser"):
	nbs_names[unsupported] = nbs_names["snare"]
nbs_names.update(dict(
	u1=14,
	u2=6,
	u3=13,
	u4=13,
	u5=5,
))
nbs_values.update({i: "creeper" for i in range(16, 32)})
pitches = dict(
	harp=24,
	bass=0,
	basedrum=0,
	snare=48,
	hat=24,
	guitar=12,
	flute=36,
	bell=48,
	chime=48,
	xylophone=48,
	iron_xylophone=24,
	cow_bell=36,
	didgeridoo=0,
	bit=24,
	banjo=24,
	pling=24,
	creeper=0,
	u1=24,
	u2=36,
	u3=24,
	u4=24,
	u5=12,
)
sustain_map = [
	0,
	0,
	1,
	1,
	0,
	0,
	0,
	1,
	0,
	1,
	1,
	1,
	1,
	1,
	0,
]
instrument_mapping = [
	 1, 1, 1, 4, 3, 3, 0, 3, # Piano
	 5, 5, 5, 4, 4, 5, 5, 1, # CP
	10, 1,11,12,10,11, 7, 3, # Organ
	 0, 0, 0, 1, 1,13,13, 7, # Guitar
	 0, 0, 0, 0, 4, 4,11,10, # Bass
	 7, 7, 7, 7, 3, 0, 0, 6, # Strings
	 2, 2, 2, 2, 9, 9, 9,12, # Ensemble
	10,10,10, 7,10,10,11,10, # Brass
	 7, 7, 7,10,10,10, 2,10, # Reed
	 2, 2, 2, 2, 2, 2, 2, 2, # Pipe
	 3,11,10,11, 3, 9,12,11, # SL
	 2,10,11, 9, 9, 2,10,10, # SP
	 2,12, 2, 2, 2, 2, 9, 2, # SE
	 0, 8, 8, 8, 1,11, 7, 7, # Ethnic
	 0, 6, 6, 6, 6, 6, 6, 6, # Percussive
	 6, 2, 6, 6, 6, 6, 6, 6, # Percussive
]
midi_instrument_selection = [
	46,
	0,
	73,
	80,
	11,
	14,
	-2,
	48,
	105,
	52,
	56,
	81,
	19,
	30,
	-1,
]
org_instrument_mapping = [
	 0, 0, 1, 3, 2, 9, 9, 3, 3, 7,
	 1, 2, 3, 3, 2, 2, 3, 7, 7, 7,
	 3, 3, 3, 3, 3,13,13, 7, 7, 7,
	 1, 1, 1, 7, 2, 2, 2, 3, 7, 7,
	 7, 3, 3, 3, 3, 3, 7, 2, 3, 2,
	 3, 7, 7, 7, 7, 7, 7,10, 2, 3,
	12, 3, 3,12, 3, 7,11,11, 2, 7,
	 0, 1, 2, 3, 2, 3, 7, 5, 4, 5,
	 7, 3, 2, 1, 2, 7, 7, 7, 7, 7,
	 4,10,10, 7, 7, 7, 7, 3, 3, 3,
]
org_instrument_selection = [
	1,
	10,
	47,
	20,
	78,
	76,
	-2,
	66,
	62,
	6,
	57,
	66,
	60,
	25,
	-1,
]
org_octave = 60
nbs2thirtydollar = dict(
	u1="noteblock_banjo",
	u2="fnf_up",
	u3="meowsynth",
	u4="mariopaint_car",
)
thirtydollar_names = [
	"noteblock_harp",
	"noteblock_pling",
	"noteblock_flute",
	"noteblock_bit",
	"noteblock_xylophone",
	"noteblock_chime",
	"🥁",
	"stylophone",
	"noteblock_banjo",
	"fnf_up",
	"mariopaint_flower",
	"meowsynth",
	"mariopaint_car",
	"🦴",
	"noteblock_",
]
thirtydollar_volumes = {
	"stylophone": 0.875,
	"fnf_up": 0.75,
	"fnf_down": 0.75,
	"🦴": 0.75,
	"mariopaint_flower": 0.8,
	"mariopaint_car": 0.8,
	"amogus_emergency": 0.8,
}
thirtydollar_unmap = {
	"noteblock_banjo": (8, 0),
	"stylophone": (7, -1 / 3, 0.875),
	"meowsynth": (11, -6),
	"fnf_up": (9, 0, 0.75),
	"🦴": (11, -16, 0.75),
	"mariopaint_flower": (10, 0, 0.8),
	"bong": (5, -13),
	"🔔": (5, 22),
	"🚫": (12, -6),
	"🤬": (3, 17),
	"buzzer": (11, -20),
	"airhorn": (10, -3),
	"puyo": (8, 8),
	"robtopphone": (4, 12),
	"🎸": (0, -36),
	"dimrainsynth": (12, 0),
	"hoenn": (10, 0),
	"🎺": (10, 0),
	"obama": (9, 0),
	"taunt": (10, 6),
	"samurai": (0, -9),
	"familyguy": (10, 6),
	"ultrainstinct": (7, 3),
	"morshu": (9, -10),
	"bup": (9, 3),
	"mariopaint_mario": (4, 12),
	"mariopaint_luigi": (3, 0),
	"mariopaint_star": (1, 24),
	"mariopaint_gameboy": (3, 0),
	"mariopaint_swan": (12, 12),
	"mariopaint_plane": (0, 0),
	"mariopaint_car": (12, 0, 0.8),
	"tab_sounds": (5, 18),
	"choruskid": (9, 0),
	"builttoscale": (1, -6),
	"fnf_left": (9, 0, 0.75),
	"fnf_down": (9, -12, 0.75),
	"fnf_right": (9, 0, 0.75),
	"gd_quit": (2, 3),
	"bwomp": (11, -36),
	"YOU": (5, -5),
	"terraria_guitar": (0, -6),
	"terraria_axe": (0, -42),
	"amogus_emergency": (3, -7.5, 0.8),
	"amongdrip": (11, -30),
	"minecraft_bell": (5, -25),
	"💡": (11, 14),
}
thirtydollar_drums = {
	"_pause": (-1, -1),
	"boom": (-1, 35),
	"👏": (-1, 39),
	"💨": (-1, 66),
	"🏏": (-1, 48),
	"shatter": (-1, 57),
	"👌": (-1, 27),
	"🖐": (-1, 39),
	"pan": (-1, 50),
	"hitmarker": (-1, 32),
	"dodgeball": (-1, 53),
	"whipcrack": (-1, 40),
	"taiko_don": (-1, 41),
	"taiko_ka": (-1, 63),
	"tf2_crit": (-1, 38),
	"smw_stomp2": (-1, 39),
	"sm64_painting": (-1, 84),
	"shaker": (-1, 82),
	"🥁": (-1, 35),
	"hammer": (-1, 38),
	"🪘": (-1, 47),
	"sidestick": (-1, 40),
	"ride2": (-1, 59),
	"buttonpop": (-1, 75),
	"skipshot": (-1, 58),
	"tab_rows": (-1, 36),
	"tab_actions": (-1, 62),
	"tab_rooms": (-1, 40),
	"preecho": (-1, 74),
	"rdclap": (-1, 42),
	"midspin": (-1, 40),
	"adofai_fire": (-1, 36),
	"adofai_ice": (-1, 38),
	"adofaikick": (-1, 35),
	"adofaicymbal": (-1, 55),
	"cowbell": (-1, 56),
	"karateman_throw": (-1, 65),
	"karateman_offbeat": (-1, 66),
	"karateman_hit": (-1, 36),
	"karateman_bulb": (-1, 58),
	"undertale_hit": (-1, 47),
	"undertale_crack": (-1, 71),
	"lancersplat": (-1, 74),
	"isaac_mantle": (-1, 87),
	"DEFEAT": (-1, 45),
	"vvvvvv_flash": (-1, 52),
	"celeste_dash": (-1, 73),
	"celeste_death": (-1, 47),
	"celeste_spring": (-1, 74),
	"celeste_diamond": (-1, 49),
	"amogus_kill": (-1, 55),
	"noteblock_snare": (-1, 40),
	"noteblock_click": (-1, 44),
}
percussion_mats = {int((data := line.split("#", 1)[0].strip().split("\t"))[0]): (data[1], int(data[2])) for line in """
0	PLACEHOLDER	0
27	obsidian	24	# High Q
28	black_concrete_powder	6	# Slap
29	warped_trapdoor	0	# Scratch Push
30	bamboo_trapdoor	0	# Scratch Pull
31	black_concrete_powder	16	# Sticks
32	dropper	0	# Square Click
33	blue_stained_glass	8	# Metronome Click
34	gold_block	18	# Metronome Bell
35	cobblestone	4	# Acoustic Bass Drum
36	cobblestone	8	# Bass Drum 1
37	blue_stained_glass	4	# Side Stick
38	black_concrete_powder	4	# Acoustic Snare
39	oak_trapdoor	0	# Hand Clap
40	black_concrete_powder	8	# Electric Snare
41	obsidian	0	# Low Floor Tom
42	blue_stained_glass	18	# Closed Hi-Hat
43	obsidian	6	# High Floor Tom
44	blue_stained_glass	21	# Pedal Hi-Hat
45	obsidian	12	# Low Tom
46	black_concrete_powder	18	# Open Hi-Hat
47	obsidian	16	# Low-Mid Tom
48	obsidian	20	# Hi-Mid Tom
49	creeper_head	0	# Crash Cymbal 1
50	obsidian	24	# High Tom
51	black_concrete_powder	23	# Ride Cymbal 1
52	black_concrete_powder	20	# Chinese Cymbal
53	black_concrete_powder	19	# Ride Bell
54	blue_stained_glass	19	# Tambourine
55	black_concrete_powder	17	# Splash Cymbal
56	soul_sand	15	# Cowbell
57	creeper_head	0	# Crash Cymbal 2
58	skeleton_skull	0	# Vibraslap
59	black_concrete_powder	24	# Ride Cymbal 2
60	cobblestone	23	# Hi Bongo
61	cobblestone	21	# Low Bongo
62	cobblestone	10	# Mute Hi Conga
63	soul_sand	7	# Open Hi Conga
64	soul_sand	2	# Low Conga
65	obsidian	22	# High Timbale
66	obsidian	6	# Low Timbale
67	bone_block	11	# High Agogo
68	soul_sand	18	# Low Agogo
69	blue_stained_glass	20	# Cabasa
70	blue_stained_glass	22	# Maracas
71	gold_block	20	# Short Whistle
72	packed_ice	17	# Long Whistle
73	blue_stained_glass	12	# Short Guiro
74	bamboo_fence_gate	0	# Long Guiro
75	bone_block	19	# Claves
76	bone_block	14	# Hi Wood Block
77	bone_block	7	# Low Wood Block
78	piglin_head	0	# Mute Cuica
79	zombie_head	0	# Open Cuica
80	bone_block	24	# Mute Triangle
81	packed_ice	24	# Open Triangle
82	blue_stained_glass	16	# Shaker
83	skeleton_skull	0	# Jingle Bell
84	packed_ice	20	# Bell Tree
85	dropper	0	# Castanets
86	obsidian	8	# Mute Surdo
87	obsidian	0	# Open Surdo
""".strip().splitlines()}
non_note_blocks = {
	"warped_trapdoor",
	"bamboo_trapdoor",
	"oak_trapdoor",
	"bamboo_fence_gate",
	"dropper",
}

# Remapping of midi note range to note block note range
c4 = 60
fs4 = c4 + 6
fs1 = fs4 - 36
c1 = c4 - 36
c0 = c4 - 48
c_1 = 0

note_names = [
	"C",
	"C#",
	"D",
	"D#",
	"E",
	"F",
	"F#",
	"G",
	"G#",
	"A",
	"A#",
	"B",
]

harmonics = dict(
	default=[(round(math.log(n, 2) * 12), 1 / n ** 2) for n in range(1, 17)][1:],
	triangle=[(round(math.log(n, 2) * 12), 1 / n ** 2) for n in range(1, 17, 2)][1:],
	square=[(round(math.log(n, 2) * 12), 1 / n) for n in range(1, 17, 2)][1:],
	saw=[(round(math.log(n, 2) * 12), 1 / n) for n in range(1, 17)][1:],
)

dawvert_inputs = dict(
	mid="midi",
	flp="flp",
	als="ableton",
	rpp="reaper",
	dawproject="dawproject",
	mmp="lmms",
	mmpz="lmms",
	ssp="serato",
	mod="mod",
	xm="xm",
	s3m="s3m",
	it="it",
	umx="umx",
	org="orgyana",
	ptcop="ptcop",
	json="jummbox",
	txt="famistudio_txt",
	dmf="deflemask",
	tbm="trackerboy",
	ibd="1bitdragon",
	jsonl="lovelycompressor",
	piximod="pixitracker",
	song="audiosauna",
	sequence="onlineseq",
	sng="soundation",
	caustic="caustic",
	mmf="mmf",
	note="notessimo_v3",
	msq="mariopaint_msq",
	mss="mariopaint_mss",
	ftr="fruitytracks",
	pmd="piyopiyo",
	squ="temper",
	sn2="soundclub2",
	rol="adlib_rol",
	sop="adlib_sop",
	fmf="flipper",
)
dawvert_outputs = dict(
	als="ableton",
	amped="amped",
	dawproject="dawproject",
	flp="flp",
	mmp="lmms",
	mid="midi",
	muse="muse",
	sequence="onlineseq",
	rrp="reaper",
	soundation="soundation",
)
legal_inputs = "zip nbs csv org xm mid midi wav flac mp3 aac ogg opus m4a wma weba webm 🗿 moai".split() + list(dawvert_inputs)