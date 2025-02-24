# Predefined list attempting to match instruments across pitch ranges
material_map = [
	["bamboo_planks", "black_wool", "black_wool+", "amethyst_block+", "gold_block", "gold_block+"],
	["bamboo_planks", "bamboo_planks+", "glowstone", "glowstone+", "gold_block", "gold_block+"],
	["pumpkin", "pumpkin+", "amethyst_block", "clay", "clay+", "packed_ice+"],
	["pumpkin", "pumpkin+", "emerald_block", "emerald_block+", "gold_block", "gold_block+"],
	["bamboo_planks", "bamboo_planks+", "iron_block", "iron_block+", "gold_block", "gold_block+"],
	["bamboo_planks", "black_wool", "amethyst_block", "amethyst_block+", "packed_ice", "packed_ice+"],
	["cobblestone", "cobblestone+", "red_stained_glass", "red_stained_glass+", "heavy_core", "heavy_core+"],
	["bamboo_planks", "black_wool", "hay_block", "hay_block+", "soul_sand+", "bone_block+"],
	None
]
default_instruments = dict(
	harp="Plucked",
	pling="Keyboard",
	flute="Wind",
	bit="Synth",
	iron_xylophone="Pitched Percussion",
	chime="Bell",
	basedrum="Unpitched Percussion",
	banjo="String",
	creeper="Drumset",
)
instrument_codelist = list(default_instruments.values())
default_instruments.update(dict(
	snare="Unpitched Percussion",
	hat="Unpitched Percussion",
	bell="Plucked",
	cow_bell="Pitched Percussion",
	didgeridoo="Wind",
	guitar="Plucked",
	bass="Plucked",
))
instrument_names = dict(
	amethyst_block="harp",
	bamboo_planks="bass",
	heavy_core="snare",
	black_concrete_powder="snare",
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
for unsupported in ("skeleton", "wither_skeleton", "zombie", "creeper", "piglin"):
	nbs_names[unsupported] = nbs_names["snare"]
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
]
instrument_mapping = [
	1, 1, 1, 4, 3, 3, 0, 1, # Piano
	5, 5, 5, 4, 4, 5, 5, 1, # CP
	2, 1, 2, 2, 7, 7, 7, 3, # Organ
	0, 0, 0, 1, 1, 3, 3, 7, # Guitar
	0, 0, 0, 0, 4, 4, 3, 3, # Bass
	7, 7, 7, 7, 3, 0, 0, 6, # Strings
	2, 2, 2, 2, 2, 2, 2, 3, # Ensemble
	2, 2, 2, 2, 2, 2, 2, 2, # Brass
	7, 7, 7, 7, 7, 2, 2, 3, # Reed
	2, 2, 2, 2, 2, 2, 2, 2, # Pipe
	3, 3, 3, 3, 3, 3, 3, 3, # SL
	2, 2, 2, 2, 2, 2, 2, 2, # SP
	2, 2, 2, 2, 2, 2, 2, 2, # SE
	0, 7, 2, 0, 1, 3, 3, 7, # Ethnic
	0, 6, 6, 6, 6, 6, 6, 6, # Percussive
	6, 6, 6, 6, 6, 6, 6, 6, # Percussive
]
midi_instrument_selection = [
	46,
	0,
	73,
	80,
	13,
	14,
	-2,
	48,
	-1,
]
org_instrument_mapping = [
	0, 0, 1, 3, 2, 2, 2, 3, 3, 7,
	1, 2, 3, 3, 2, 2, 3, 7, 7, 7,
	3, 3, 3, 3, 3, 3, 3, 7, 7, 7,
	1, 1, 1, 7, 2, 2, 2, 3, 7, 7,
	7, 3, 3, 3, 3, 3, 7, 2, 3, 2,
	3, 7, 7, 7, 7, 7, 7, 2, 2, 3,
	3, 3, 3, 3, 3, 7, 7, 7, 2, 7,
	0, 1, 2, 3, 2, 3, 7, 5, 4, 5,
	7, 3, 2, 1, 2, 7, 7, 7, 7, 7,
	4, 2, 7, 7, 7, 7, 7, 3, 3, 3,
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
	-1,
	60,
	-1,
]
org_octave = 60
percussion_mats = {int((data := line.split("#", 1)[0].strip().split("\t"))[0]): (data[1], int(data[2])) for line in """
0	PLACEHOLDER	0
31	heavy_core	16	# Sticks
32	blue_stained_glass	24	# Square Click
33	blue_stained_glass	8	# Metronome Click
34	gold_block	18	# Metronome Bell
35	netherrack	4	# Acoustic Bass Drum
36	netherrack	8	# Bass Drum 1
37	blue_stained_glass	4	# Side Stick
38	heavy_core	4	# Acoustic Snare
39	heavy_core	12	# Hand Clap
40	heavy_core	8	# Electric Snare
41	obsidian	0	# Low Floor Tom
42	blue_stained_glass	18	# Closed Hi-Hat
43	obsidian	6	# High Floor Tom
44	blue_stained_glass	21	# Pedal Hi-Hat
45	obsidian	12	# Low Tom
46	heavy_core	18	# Open Hi-Hat
47	obsidian	16	# Low-Mid Tom
48	obsidian	20	# Hi-Mid Tom
49	creeper_head	0	# Crash Cymbal 1
50	obsidian	24	# High Tom
51	heavy_core	23	# Ride Cymbal 1
52	heavy_core	20	# Chinese Cymbal
53	heavy_core	19	# Ride Bell
54	blue_stained_glass	19	# Tambourine
55	heavy_core	17	# Splash Cymbal
56	soul_sand	15	# Cowbell
57	creeper_head	0	# Crash Cymbal 2
58	skeleton_skull	0	# Vibraslap
59	heavy_core	24	# Ride Cymbal 2
60	netherrack	23	# Hi Bongo
61	netherrack	21	# Low Bongo
62	netherrack	10	# Mute Hi Conga
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
74	skeleton_skull	0	# Long Guiro
75	bone_block	19	# Claves
76	bone_block	14	# Hi Wood Block
77	bone_block	7	# Low Wood Block
78	piglin_head	0	# Mute Cuica
79	zombie_head	0	# Open Cuica
80	bone_block	24	# Mute Triangle
81	packed_ice	24	# Open Triangle
82	blue_stained_glass	16	# Closed Hi-Hat
83	skeleton_skull	0	# Jingle Bell
84	packed_ice	20	# Bell Tree
""".strip().splitlines()}

# Remapping of midi note range to note block note range
c4 = 60
fs4 = c4 + 6
fs1 = fs4 - 36
c1 = c4 - 36
c0 = c4 - 48
c_1 = 0

MAIN = 4
SIDE = 2
DIV = 4
BAR = 32

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