# Hyperchoron - Making Music Multidimensional
Hyperchoron (pun on chorus, and -choron, suffix for 4-dimensional polytopes) was originally exclusively designed as a MIDI to Minecraft Note Block export tool. <br />
Hyperchoron is a script and library that uses a combination of theory and heuristics to analyse, quantise and convert between various music sheet formats, mostly relevant to various video games.<br />
It intelligently works around limitations of each format, and adapts songs in natively supported formats to very accurate degrees, closely resembling how the end result would sound if transcribed manually by a human well-versed in music.<br />
For those who simply want to port or listen to songs across different platforms, this tool can save you hours with each transcription, and it finishes before you can blink!<br />
The decoders are optimised to handle black MIDIs with up to hundreds of millions of notes (depending on your system's specs) within minutes, and to parallelise conversions of entire folders with a single command.

- If there is a format that you don't see here, and is typically difficult to convert, feel free to open an issue or contact me on Discord to request it as a feature! I am always open to suggestions to expand this library.

![thumbnail](https://raw.githubusercontent.com/thomas-xin/hyperchoron/refs/heads/main/thumb.jpg)

- Disclaimer: Hyperchoron does not apply generative AI to produce songs; all notes are produced from the inputs only, and are not manifested or extrapolated by neural networks. This means that if you want the best results, you should find existing versions or transcriptions of the song that are of high quality.
# Instructions for use
## Installation
- Install [python](https://www.python.org) and [pip](https://pip.pypa.io/en/stable/)
- Install Hyperchoron as a package:
`pip install hyperchoron`
- (Optional) Install dependencies for PCM-related music features (this takes much more disk space due to the use of neural networks and larger audio frameworks):<br />
`pip install hyperchoron[pcm]`
- (Optional) Install dependencies for non-natively supported formats: [DawVert](https://github.com/SatyrDiamond/DawVert):<br />
`git clone https://github.com/SatyrDiamond/DawVert`<br />
## Usage
```ini
usage: hyperchoron [-h] [-V] -i INPUT [INPUT ...] -o OUTPUT [OUTPUT ...] [-f FORMAT] [-x [MIXING]] [-v [VOLUME]] [-s [SPEED]] [-r [RESOLUTION]] [-st | --strict-tempo | --no-strict-tempo] [-t [TRANSPOSE]] [-ik | --invert-key | --no-invert-key]
                   [-mt | --microtones | --no-microtones] [-ac | --accidentals | --no-accidentals] [-av | --apply-volumes | --no-apply-volumes] [-er | --extended-ranges | --no-extended-ranges] [-tc | --tempo-changes | --no-tempo-changes]
                   [-d | --drums | --no-drums] [-md [MAX_DISTANCE]] [-mi | --minecart-improvements | --no-minecart-improvements]

MIDI-Tracker-DAW converter and Minecraft Note Block exporter

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  -i, --input INPUT [INPUT ...]
                        Input file (.zip | .mid | .csv | .nbs | .org | *)
  -o, --output OUTPUT [OUTPUT ...]
                        Output file (.mid | .csv | .nbs | .nbt | .mcfunction | .litematic | .schem | .schematic | .org | .skysheet | .genshinsheet | *)
  -f, --format FORMAT   Output format (mid | csv | nbs | nbt | mcfunction | litematic | .schem | .schematic | org | skysheet | genshinsheet | deltarune | *)
  -x, --mixing [MIXING]
                        Behaviour when importing multiple files. "I" to process individually, "L" to layer/stack, "C" to concatenate. If multiple digits are inputted, this will be interpreted as a hierarchy. For example, for a 3-deep nested zip
                        folder where pairs of midis at the bottom layer should be layered, then groups of those layers should be concatenated, and there are multiple of these groups to process independently, input "ICL". Defaults to "IL"
  -v, --volume [VOLUME]
                        Scales volume of all notes up/down as a multiplier, applied before note quantisation. Defaults to 1
  -s, --speed [SPEED]   Scales song speed up/down as a multiplier, applied before tempo sync; higher = faster. Defaults to 1
  -r, --resolution [RESOLUTION]
                        Target time resolution of data, in hertz (per-second). Defaults to 12 for .🗿, .skysheet and .genshinsheet outputs, 20 for .nbt, .mcfunction and .litematic outputs, 40 otherwise
  -st, --strict-tempo, --no-strict-tempo
                        Snaps the song's tempo to the target specified by --resolution, being more lenient in allowing misaligned notes to compensate. Defaults to TRUE for .nbt, .mcfunction and .litematic outputs, FALSE otherwise
  -t, --transpose [TRANSPOSE]
                        Transposes song up/down a certain amount of semitones, applied before instrument material mapping; higher = higher pitched. Defaults to 0
  -ik, --invert-key, --no-invert-key
                        Experimental: During transpose step, autodetects song key signature, then inverts it (e.g. C Major <=> C Minor). Defaults to FALSE
  -mt, --microtones, --no-microtones
                        Allows microtones/pitchbends. If disabled, all notes are clamped to integer semitones. For Minecraft outputs, defers affected notes to command blocks. Has no effect if --accidentals is FALSE. Defaults to FALSE for .nbt,
                        .mcfunction, .litematic, .org, .skysheet and .genshinsheet outputs, TRUE otherwise
  -ac, --accidentals, --no-accidentals
                        Allows accidentals. If disabled, all notes are clamped to the closest key signature. Warning: Hyperchoron is currently only implemented to autodetect a single key signature per song. Defaults to FALSE for .skysheet and
                        .genshinsheet outputs, TRUE otherwise
  -av, --apply-volumes, --no-apply-volumes
                        Applies note voluming. If disabled, all notes are either 0% or 100% volume with no inbetween. Not currently implemented for all formats. Defaults to TRUE
  -er, --extended-ranges, --no-extended-ranges
                        Extends instrument ranges for formats with limitations. Defaults to TRUE for .nbs, FALSE otherwise
  -tc, --tempo-changes, --no-tempo-changes
                        Allows tempo changes. If disabled, all notes are moved to an approximate relative tick based on their real time as calculated with the tempo change, but without tempo changes in the output. CURRENTLY UNIMPLEMENTED; ALWAYS
                        FALSE.
  -d, --drums, --no-drums
                        Allows percussion channel. If disabled, percussion channels will be discarded. Defaults to TRUE
  -md, --max-distance [MAX_DISTANCE]
                        For Minecraft outputs only: Restricts the maximum block distance the notes may be placed from the centre line of the structure, in increments of 3 (one module). Decreasing this value makes the output more compact, at the
                        cost of note volume accuracy. Defaults to 42
  -mi, --minecart-improvements, --no-minecart-improvements
                        For Minecraft outputs only: Assumes the server is running the [Minecart Improvements](https://minecraft.wiki/w/Minecart_Improvements) version(s). Less powered rails will be applied on the main track, to account for the
                        increased deceleration. Currently only semi-functional; the rail section connecting the midway point may be too slow.
```
### Examples
Converting a MIDI file into a Minecraft Litematica schematic:
- `hyperchoron -i input.mid -o output.litematic`

Converting a MIDI file into a Minecraft WorldEdit schematic:
- `hyperchoron -i input.mid -o output.schem`

Converting a recursively nested .7z archive of .xm files into a MIDI file, stacking them together as if they were individual instrument parts (Refer below for additional requirements for .xm):
- `hyperchoron -i input.7z -x L -o output.mid`

Converting all elements (of various file types) from the folder of example inputs into .skysheet keeping their original filenames otherwise, placing them into a new folder:
- `hyperchoron -i examples/input -a -f skysheet -o examples/skysheet/`
  - Warning: May consume several gigabytes of RAM!

Converting a MIDI file stored on a remote server, accessible via HTTP:
- `hyperchoron -i https://mizabot.xyz/u/5OK0rYLJEBCJYQYRKBJbaAAQAYBYZA/mk7_rainbow_road.mid -o output.nbt`

Converting a FL Studio project into a Minecraft Note Block Studio project, ignoring vanilla Minecraft limitations (Refer below for additional requirements for .flp):
- `hyperchoron -i input.flp -mt -o output.nbs`

Converting a raw audio file into a .csv, a .🗿 and a .org file, transcribing all notes up a Major 3rd (Refer below for additional requirements for .wav):
- `hyperchoron -i input.wav -t 4 -o output.csv output.🗿 output.org`
  - Warning: Performing multiple conversions of different formats will cause default parameters to take on the values of those for the first format!
### As a library
Hyperchoron can be imported as a Python library for integration with other tools, and the following useful functions are exposed:
- `hyperchoron.convert_files(**kwargs) -> list`: Runs a conversion identical to the one you would get from running the cli tool as a standalone, passing the same optional arguments. This is included as an alternative to the cli, for the sake of performance if you're running multiple conversions. Supports the same multiple input/output/mixing modes the cli tool does, including the folder hierarchy unpacking, and parallelisation of multiple conversions where applicable.
- `hyperchoron.clear_cache()`: Attempts to delete the folder used for writing temporary data to disk during processing. Useful if you're not using the default handler.
- `hyperchoron.decoder_mapping`: A dictionary containing the available decoders, "_" representing the fallback decoder.
- `hyperchoron.encoder_mapping`: A dictionary containing the available encoders, "_" representing the fallback encoder.
- `hyperchoron.load_file(fi, ctx) -> "np.ndarray"`: Loads a single file from any of the decodable formats, returning a numpy array of the decoded events as MIDI, similar to tools such as midicsv.
- `hyperchoron.save_file(midi_events, fo, ctx) -> str`: Saves a list or numpy array of MIDI-like events to a single file from any of the encodable formats. Performs conversions as necessary.
- `hyperchoron.util.get_children(path) -> list`: Retrieves the list of "children" of a path. This may contain only a single element for regular files or folders of a single element, but will additionally resolve URLs and compressed archives to their own temporary folders, returning the new path of each.
- `hyperchoron.util.Transport(self, notes=None, tick_delay=0)`: The main class for temporary storage of music in the intermediate stage of conversion, also used for the `.hpc` extension. Intended to be serialisable, and auto-compressed in memory.
- `hyperchoron.fastmidi.parse_midi_events(path)`: The main MIDI parser as a standalone tool, now rewritten in rust for blazing fast speeds!! 🦀 (Significantly faster than midicsv implementations, returns a 6-column numpy array of all MIDI events with track numbers removed, and each note event instead contains the note's duration in ticks as the last element.)
### Input Formats
- Native support:
  - .hpc: The program's own internal memory storage format. Useful if you want to "cut down" an input with hundreds of millions of notes into a cache, to postprocess into multiple formats with different settings later.
  - .mid/.midi: Musical Instrument Digital Interface, a common music communication protocol and project file format supporting up to 15 melodic and 1 percussion instrument channel, polyphony, sustain, volume, panning and pitchbends.
  - .csv: Comma Separated Value table representing MIDI data.
  - .nbs: Minecraft Note Block Studio project files, used to represent songs composed of Minecraft sound effects. Supports up to 65535 channels in theory, volume and panning, with various limitations on pitch representations.
  - .org: Cave Story Organya tracker files; a song format supporting 8-bit instruments, with up to 8 channels, sustain, volume and panning.
  - .🗿/.moai: Discrete sound effect representation functionally similar to .nbs, supports most Minecraft note block instruments, used for https://thirtydollar.website.
- WIP (Currently in the works, not yet supported/functional):
  - .xm: Extended Module tracker; a popular project format used in video games, supporting 16-bit instruments, with up to 32 channels, sustain, volume, panning and pitchbends. Requires `pcm` dependencies.
  - .wav: Raw pulse-code modulation representation of audio, including all rendered songs and audio recordings. Compressed audio is also supported. Requires `pcm` dependencies.
- Limited:
  - Any format supported by [DawVert](https://github.com/SatyrDiamond/DawVert) if installed. Note that this converter may not always preserve format-specific features in a faithful way, meaning things like pitchbends may be lost.
  - Due to the limitations of DawVert, direct support for formats is slowly being implemented and will take priority where available, to better maintain accuracy in conversion.
  - Please see the project's page and documentation for additional formats supported!
### Output Formats
- Native support:
  - .hpc
  - .mid/.midi
  - .csv
  - .nbs
    - If you want to ensure that the output stays vanilla Minecraft compliant, be sure to use the `--strict-tempo` argument, and optionally `--no-microtones` if you would like to output to also be survival-legal. These options are normally enabled by default for `.litematic`, `.mcfunction` and `.nbt` outputs. If not specified, Hyperchoron will attempt to utilise the full capacity of the `.nbs` format including non-integer tick rates, and will only switch instruments if notes fall outside even the extended range provided by Note Block Studio.
  - .mcfunction: A list of Minecraft `/setblock` commands, to be run through a modded client or a datapack. The notes will be mapped to a multi-layered structure enabling 20Hz playback, but with limitations on polyphony, volume and pan control.
  - .litematic: Similar output to `.mcfunction`, but more easily viewed and pasted using the [Litematica](https://modrinth.com/mod/litematica) mod.
  - .nbt: A Minecraft NBT structure file, normally intended for use with structure blocks. However, in practice the outputs are usually too large, and will need a third-party mod (litematica included) to paste properly.
  - .org
  - .🗿/.moai
  - `deltarune`: A list of files that enable playing the song in the Deltarune rhythm game. Included will be three .ogg files which must be placed in the `mus` folder, as well as `rhythmgame_notechart.gml` and `rhythmgame_song_load.gml` files, which must be imported into the `data.win` code using a tool such as UndertaleModTool. See https://youtu.be/rSE3DecbFsM as a guide for this! Requires `pcm` dependencies. Note: This format does not have an extension, and must be manually specified using `-f`/`--format`. The output may be an empty folder or an archive.
  - .skysheet: A recreation and representation of the playable music sheets used in Sky: Children of the Light. The format supported by Hyperchoron is specifically for the https://sky-music.specy.app editor.
  - .genshinsheet: Similar to .skysheet, but for the game Genshin Impact. See https://genshin-music.specy.app for the corresponding editor.
  - .wav (+ FFmpeg outputs): Requires PCM dependencies. Renders songs as raw audio, possibly layered depending on options.
- WIP:
  - .xm
  - .schematic (+ other Minecraft formats)
- Limited:
  - Once again, DawVert-supported formats are also automatically supported by Hyperchoron.

## HPC Format
If you are looking to read/write Hyperchoron's compact memory format for any reason, the following details may be helpful:
- Header Magic: 4 bytes, string, always `~HPC`
- Version: 4 bytes, unsigned integer, currently 1
- Metadata:
  - LEB128-encoded size of data
  - Binary or text data, currently unused
- Data Sections (Possibly multiple):
  - LEB128-encoded size of section
  - Tick Count: 4 bytes, unsigned integer
  - Tick Delay: 8 bytes, fraction, represents delay of each tick in seconds:
    - Numerator: 4 bytes, unsigned integer
    - Denominator: 4 bytes, unsigned integer
  - LZMA2-compressed note data:
    - Note Column/Tick (Possibly multiple):
      - LEB128-encoded note count
        - Contains an additional condition that if this number is artificially extended to end in a `0x00` byte (does not normally occur with LEB128 encoding), the column is in "full" mode rather than "compact" mode, allowing out-of-range notes, finetunes, extremely high or low volumes, or chords with mixed modality, at the cost of slight compression efficiency. All notes encoded inside this column, defined below, will follow this mode.
      - Modality (Optional, only included in "compact" mode): 1 byte, unsigned integer, representing what format the note(s) originated from. Used for rendering raw audio.
        - Currently `0` for MIDI, `1` for NBS, `16` for ORG (other formats are not yet implemented, and will be interpreted as one of the aforementioned modalities).
      - Notes (Possibly multiple):
        - Full Note (9 bytes each):
          - Priority: 0.5 bytes, signed integer:
            - 2 for known note heads, 1 for standard notes, 0 for note trails, and -1 for deprioritised notes to be discarded in discrete outputs
          - Modality: 0.5 bytes, unsigned integer
          - Instrument ID: 1 byte, unsigned integer, with 255 representing -1; the instrument ID from the original format/modality
          - Instrument Class: 1 byte, signed integer; the instrument ID within Hyperchoron's known classes of instruments, used to assist mappings during conversions
            - 0: Plucked
            - 1: Piano
            - 2: Wind
            - 3: Square Synth
            - 4: Pitched Percussion
            - 5: Bell
            - 6: Unpitched Percussion
            - 7: String
            - 8: Banjo
            - 9: Voice
            - 10: Brass
            - 11: Saw Synth
            - 12: Organ
            - 13: Overdrive Guitar
            - -1: MIDI-mapped Drumset
          - Pitch: 2 bytes, float, in semitones starting from C0
          - Volume: 2 bytes, float, in logarithmic scale normalised at 1.0
          - Panning: 1 byte, signed integer, centred at 0 with -127 representing maximum left and 127 representing maximum right
          - Timing Offset: 1 byte, unsigned integer (irrelevant for most scenarios)
        - Compact Note (6 bytes each):
          - Priority: 0.5 bytes, signed integer
          - Timing Offset: 0.5 bytes, unsigned integer
          - Instrument ID: 1 byte, unsigned integer
          - Instrument Class: 1 byte, signed integer
          - Pitch: 1 byte, unsigned integer, in semitones starting from C0
          - Volume: 1 byte, unsigned integer, in logarithmic scale normalised at 255
          - Panning: 1 byte, signed integer

## Minecraft info
Odds are, most people finding their way to this repository will be mainly interested in the Minecraft export capabilities and instructions. As such, all information listed below will specifically be about exporting to Minecraft note blocks.
- If exporting to `.mcfunction`, you will need to make some sort of template datapack to be able to load it in. When pasting for the first time, it is recommended to perform the `/gamerule maxCommandChainLength 2147483647` command prior to pasting the note blocks to avoid longer songs being cut off. Alternatively, you may run the mcfunction twice, which will do this automatically.
- As of 2025/05, the structure has been redesigned to enable support for note volume and panning, alongside a new `--max-distance` parameter; this controls the maximum distance notes may be placed from the centreline where the player will travel. If the size of the output is too large for your use case, you may decrease this for a more compact structure. However, this comes at the cost of decreasing volume accuracy.

### MAESTRO support
- Hyperchoron supports exporting to `.nbs` files that are playable in [MAESTRO](https://www.youtube.com/watch?v=G78AnHpIw5w). For best results, include the arguments `-r 20 -st --no-extended-ranges --no-apply-volumes`, which will restrict the output to be survival-legal.

### What is the purpose of another Minecraft exporter like this?
- Converting music to Minecraft note blocks programmatically has been a thing for a long time, the most popular program being Note Block Studio. This program is not intended to entirely replace them, and is meant to be a standalone feature.
- Hyperchoron's intent is to expand on the exporting capabilities of note block programs, using a more adaptable algorithm to produce more accurate recreations of songs while still staying within the boundaries of vanilla Minecraft.
  - Note Sustain: Automatically cut excessively long notes into segments to be replayed several times, preventing notes that are supposed to be sustained from fading out quickly. This adapts based on the volume of the current and adjacent notes, and will automatically offset chords into a strum to avoid excessive sudden loudness.
  - Instrument Mapping: Rather than map all MIDI instruments to a list of blocks and call it a day, this program has the full 6-octave range for all instruments, automatically swapping the materials if a note exceeds the current range. This is very important for most songs as the default range of 2 octaves in vanilla Minecraft means notes will frequently fall out of range. Clipping, dropping or wrapping notes at the boundary are valid methods of handling this, but they do not permit an accurate recreation of the full song.
  - Adding to the previous point, all drums are implemented separately, rather than using only the three drum instruments in vanilla (which end up drowning each other out), some percussion instruments are mapped to other instruments or mob heads, which allows for a greater variety of sound effects.
  - Pitch Bends: Interpret pitch bends from MIDI files, and automatically convert them into grace notes. This is very important for any MIDI file that includes this mechanic, as the pitch of notes involved will often be very wrong without it. As of 2025/08, pitchbends and finetunes can now be ported over if the `--microtones` parameter is set, although this will make the structure illegal for survival mode, as it utilises command blocks.
  - Polyphonic Budget: If there are too many notes being played at any given moment, the quietest notes and intermediate notes used for sustain will be discarded first. Depending on the `--max-distance` parameter, the schematic allows for up to 87 notes at any point in time, which decreases if a more compact structure is desired.
  - Full Tick Rate: Using the piston-and-leaves circuit, a 5-gametick delay can be achieved, which allows offsetting from the (usual) even-numbered ticks redstone components are capable of operating at. This means the full 20Hz tick speed of vanilla Minecraft can be accessed, allowing much greater timing precision.
  - Tempo Alignment: The tempo of songs is automatically synced to Minecraft's default tick rate not using the song's time signature, but rather the greatest common denominator of the notes' timestamps, pruning outliers as necessary. This allows keeping in sync with songs with triplets, quintuplets, or any other measurement not divisible by a power of 2. The algorithm falls back to unsynced playback if a good timing candidate cannot be found, which allows songs with tempo changes or that do not follow their defined time signature at all to still function.
  - Note Volume: Notes are automatically spread out further from the centreline where the player is, depending on their volume/velocity. This additionally respects the direction of the notes' panning if specified.

### Additional notes
- Hyperchoron's design was originally focused on importing and exporting to vanilla Minecraft as accurately and as reasonably as possible. That means, there may be limitations when attempting to convert songs with a much higher speed or many stacked notes; in cases where the target resolution is lower, the notes will be compacted and quantised to fit.
- Please note that conversion quality may vary significantly between different versions of Hyperchoron. This is an unavoidable nuance that comes with attempting to provide a one-size-fits-all solution for sheet music of many kinds; often making one thing sound better will make another sound worse.
- Here is a video showcasing some of the example outputs in an earlier version: https://youtu.be/Vtmh1Qi0w9s, and here is an example of the latest version with volume control: https://www.youtube.com/watch?v=qK1n4zVSjM0