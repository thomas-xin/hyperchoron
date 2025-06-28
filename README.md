# Hyperchoron - Making Music Multidimensional

![thumbnail](https://raw.githubusercontent.com/thomas-xin/hyperchoron/refs/heads/main/thumb.jpg)
# Instructions for use
## Installation
- Install [git](https://github.com/git-guides/install-git), [python](https://www.python.org) and [pip](https://pip.pypa.io/en/stable/)
- Clone this repo:
`git clone https://github.com/thomas-xin/hyperchoron`
- Install this as a package:
`pip install -e .`
- (Optional) Install [DawVert](https://github.com/SatyrDiamond/DawVert) in the same directory:<br />
`git clone https://github.com/SatyrDiamond/DawVert`<br />
`python3 -m pip install -r DawVert/requirements.txt`
## Usage
```ini
usage: hyperchoron [-h] [-V] -i INPUT [INPUT ...] -o [OUTPUT ...] [-r [RESOLUTION]] [-s [SPEED]] [-v [VOLUME]]
                   [-t [TRANSPOSE]] [-ik | --invert-key | --no-invert-key] [-sa [STRUM_AFFINITY]]
                   [-d | --drums | --no-drums] [-md [MAX_DISTANCE]] [-ml | --mc-legal | --no-mc-legal]

MIDI-Tracker-DAW converter and Minecraft Note Block exporter

options:
usage: hyperchoron [-h] [-V] -i INPUT [INPUT ...] -o [OUTPUT ...] [-r [RESOLUTION]] [-s [SPEED]] [-v [VOLUME]]
                   [-t [TRANSPOSE]] [-ik | --invert-key | --no-invert-key] [-sa [STRUM_AFFINITY]]
                   [-d | --drums | --no-drums] [-ml | --mc-legal | --no-mc-legal] [-md [MAX_DISTANCE]]
                   [-cb | --command-blocks | --no-command-blocks]
                   [-mi | --minecart-improvements | --no-minecart-improvements]

MIDI-Tracker-DAW converter and Minecraft Note Block exporter

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version and exit
  -i, --input INPUT [INPUT ...]
                        Input file (.zip | .mid | .csv | .nbs | .org | *)
  -o, --output [OUTPUT ...]
                        Output file (.mid | .csv | .nbs | .nbt | .mcfunction | .litematic | .org | *)
  -r, --resolution [RESOLUTION]
                        Target time resolution of data, in hertz (per-second). Defaults to 20 for Minecraft outputs,
                        40 otherwise
  -s, --speed [SPEED]   Scales song speed up/down as a multiplier, applied before tempo sync; higher = faster.
                        Defaults to 1
  -v, --volume [VOLUME]
                        Scales volume of all notes up/down as a multiplier, applied before note quantisation. Defaults
                        to 1
  -t, --transpose [TRANSPOSE]
                        Transposes song up/down a certain amount of semitones, applied before instrument material
                        mapping; higher = higher pitched. Defaults to 0
  -ik, --invert-key, --no-invert-key
                        Experimental: During transpose step, autodetects song key signature, then inverts it (e.g. C
                        Major <=> C Minor). Defaults to FALSE
  -sa, --strum-affinity [STRUM_AFFINITY]
                        Increases or decreases threshold for sustained notes to be cut into discrete segments; higher
                        = more notes. Defaults to 1
  -d, --drums, --no-drums
                        Allows percussion channel. If disabled, percussion channels will be treated as regular
                        instrument channels. Defaults to TRUE
  -ml, --mc-legal, --no-mc-legal
                        Forces song to be vanilla Minecraft compliant. Defaults to TRUE for .litematic, .mcfunction
                        and .nbt outputs, FALSE otherwise
  -md, --max-distance [MAX_DISTANCE]
                        For Minecraft outputs only: Restricts the maximum block distance the notes may be placed from
                        the centre line of the structure, in increments of 3 (one module). Decreasing this value makes
                        the output more compact, at the cost of note volume accuracy. Defaults to 42
  -cb, --command-blocks, --no-command-blocks
                        For Minecraft outputs only: Enables the use of command blocks in place of notes that require
                        finetune/pitchbend to non-integer semitones. Not survival-legal. Defaults to FALSE if --mc-
                        legal is set, TRUE otherwise
  -mi, --minecart-improvements, --no-minecart-improvements
                        For Minecraft outputs only: Assumes the server is running the [Minecart
                        Improvements](https://minecraft.wiki/w/Minecart_Improvements) version(s). Less powered rails
                        will be applied on the main track, to account for the increased deceleration. Currently only
                        semi-functional; the rail section connecting the midway point may be too slow.
```
### Examples
Converting a MIDI file into a Minecraft Litematica schematic:
- `hyperchoron -i input.mid -o output.litematic`

Converting a FL Studio project into a Minecraft Note Block Studio project, ignoring vanilla Minecraft limitations (Refer below for additional requirements):
- `hyperchoron -i input.flp -x -o output.nbs`

Converting a raw audio file into a MIDI project file, transcribing all notes up a Major 3rd (Refer below for additional requirements):
- `hyperchoron -i input.wav -t 4 -o output.mid`
## Project info
- Hyperchoron (pun on chorus, and -choron, suffix for 4-dimensional polytopes) was originally exclusively designed as a MIDI to Minecraft Note Block export tool. Over time, support for other formats are being added, and it can now be considered a multipurpose project import helper.
- The program takes one or more input and output files, with `.zip` file inputs being treated as multiple inputs in one.
- Multiple inputs are treated as a stacked MIDI project, and the exported notes will be combined into a single output stream.
- The ability to export to more formats than just Minecraft schematics started being added once it was realised that the heuristics behind tempo sync, note splicing and sorting, and envelope approximation can be repurposed into a converter for music formats not typically compatible due to limitations.
### Input Formats
- Native support:
  - .mid/.midi: Musical Instrument Digital Interface, a common music communication protocol and project file format supporting up to 15 melodic and 1 percussion instrument channel, polyphony, sustain, volume, panning and pitchbends.
  - .csv: Comma Separated Value table representing MIDI data.
  - .nbs: Minecraft Note Block Studio project files, used to represent songs composed of Minecraft sound effects. Supports up to 65535 channels in theory, volume and panning, with various limitations on pitch representations.
  - .org: Cave Story Organya tracker files; a song format supporting 8-bit instruments, with up to 8 channels, sustain, volume and panning.
  - .🗿/.moai: Discrete sound effect representation functionally similar to .nbs, supports most Minecraft note block instruments, used for https://thirtydollar.website.
- WIP (Currently in the works, not yet supported/functional):
  - .xm: Extended Module tracker; a popular project format used in video games, supporting 16-bit instruments, with up to 32 channels, sustain, volume, panning and pitchbends.
  - .wav: Raw pulse-code modulation representation of audio, including all rendered songs and audio recordings. This means all compressed variants will also gain support, although they will require [FFmpeg](https://www.ffmpeg.org) installed to provide coverage;
    - .mp3
    - .flac
    - .wmv
    - .opus
    - .ogg
    - .m4a
    - And far more than can be listed here!
- Limited:
  - Any format supported by [DawVert](https://github.com/SatyrDiamond/DawVert) if installed. Note that this converter may not always preserve format-specific features in a faithful way, meaning things like pitchbends may be lost.
  - Due to the limitations of DawVert, direct support for formats is slowly being implemented and will take priority where available, to better maintain accuracy in conversion.
  - As of 2025/02, this includes the following formats:
    - .flp
    - .als
    - .mmp/.mmpz
    - .rpp
    - .dawproject
    - .ssp
    - .mod
    - .xm
    - .s3m
    - .it
    - .umx
    - .txt
    - .dmf
    - .tbm
    - .ibd
    - .jsonl
    - .piximod
    - .json
    - .song
    - .sequence
    - .sng
    - .caustic
    - .mmf
    - .note
    - .msq
    - .mss
    - .ftr
    - .pmd
    - .squ
    - .ptcop
    - .sn2
    - .rol
    - .sop
    - .fmf
### Output Formats
- Native support:
  - .mid/.midi
  - .csv
  - .nbs
    - If you want to ensure that the output stays vanilla Minecraft compliant, be sure to use the `--mc-legal` argument. This option is normally enabled by default for `.litematic`, `.mcfunction` and `.nbt` outputs. If not specified, Hyperchoron will attempt to utilise the full capacity of the `.nbs` format including non-integer tick rates, and will only switch instruments if notes fall outside even the extended range provided by Note Block Studio.
  - .mcfunction: A list of Minecraft `/setblock` commands, to be run through a modded client or a datapack. The notes will be mapped to a multi-layered structure enabling 20Hz playback, but with limitations on polyphony, volume and pan control.
  - .litematic: Similar output to `.mcfunction`, but more easily viewed and pasted using the [Litematica](https://modrinth.com/mod/litematica) mod.
  - .nbt: A Minecraft NBT structure file, normally intended for use with structure blocks. However, in practice the outputs are usually too large, and will need a third-party mod (litematica included) to paste properly.
  - .org
  - .🗿/.moai
  - .zip (currently placeholder): An archive containing files that enable playing the song in the Deltarune rhythm game. Included will be three .ogg files which must be placed in the `mus` folder, as well as `rhythmgame_notechart.gml` and `rhythmgame_song_load.gml` files, which must be imported into the `data.win` code using a tool such as UndertaleModTool. See https://youtu.be/rSE3DecbFsM as a guide for this! Requires FFmpeg installed in PATH, as Deltarune requires vorbis-encoded audio.
- WIP:
  - .xm
  - .wav (+ FFmpeg outputs)
- Limited:
  - Once again, DawVert-supported formats are also automatically supported by Hyperchoron. As of 2025/02, this includes the following formats:
    - .ableton
    - .amped
    - .dawproject
    - .flp
    - .lmms
    - .muse
    - .onlineseq
    - .reaper
    - .soundation

## Minecraft info
Odds are, most people finding their way to this repository will be mainly interested in the Minecraft export capabilities and instructions. As such, all information listed below will specifically be about exporting to Minecraft note blocks.
- If exporting to `.mcfunction`, you will need to make some sort of template datapack to be able to load it in. When pasting for the first time, it is recommended to perform the `/gamerule maxCommandChainLength 2147483647` command prior to pasting the note blocks to avoid longer songs being cut off. Alternatively, you may run the mcfunction twice, which will do this automatically.
- As of 2025/05, the structure has been redesigned to enable support for note volume and panning, alongside a new `--max-distance` parameter; this controls the maximum distance notes may be placed from the centreline where the player will travel. If the size of the output is too large for your use case, you may decrease this for a more compact structure. However, this comes at the cost of decreasing volume accuracy.

### What is the purpose of another Minecraft exporter like this?
- Converting music to Minecraft note blocks programmatically has been a thing for a long time, the most popular program being Note Block Studio. This program is not intended to entirely replace them, and is meant to be a standalone feature.
- Hyperchoron's intent is to expand on the exporting capabilities of note block programs, using a more adaptable algorithm to produce more accurate recreations of songs while still staying within the boundaries of vanilla Minecraft.
  - Note Sustain: Automatically cut excessively long notes into segments to be replayed several times, preventing notes that are supposed to be sustained from fading out quickly. This adapts based on the volume of the current and adjacent notes, and will automatically offset chords into a strum to avoid excessive sudden loudness.
  - Instrument Mapping: Rather than map all MIDI instruments to a list of blocks and call it a day, this program has the full 6-octave range for all instruments, automatically swapping the materials if a note exceeds the current range. This is very important for most songs as the default range of 2 octaves in vanilla Minecraft means notes will frequently fall out of range. Clipping, dropping or wrapping notes at the boundary are valid methods of handling this, but they do not permit an accurate recreation of the full song.
  - Adding to the previous point, all drums are implemented separately, rather than using only the three drum instruments in vanilla (which end up drowning each other out), some percussion instruments are mapped to other instruments or mob heads, which allows for a greater variety of sound effects.
  - Pitch Bends: Interpret pitch bends from MIDI files, and automatically convert them into grace notes. This is very important for any MIDI file that includes this mechanic, as the pitch of notes involved will often be very wrong without it. As of 2025/06, pitchbends and finetunes can now be ported over if the `--command-blocks` parameter is set, although this will make the structure illegal for survival mode.
  - Polyphonic Budget: If there are too many notes being played at any given moment, the quietest notes and intermediate notes used for sustain will be discarded first. Depending on the `--max-distance` parameter, the schematic allows for up to 87 notes at any point in time, which decreases if a more compact structure is desired.
  - Full Tick Rate: Using the piston-and-leaves circuit, a 5-gametick delay can be achieved, which allows offsetting from the (usual) even-numbered ticks redstone components are capable of operating at. This means the full 20Hz tick speed of vanilla Minecraft can be accessed, allowing much greater timing precision.
  - Tempo Alignment: The tempo of songs is automatically synced to Minecraft's default tick rate not using the song's time signature, but rather the greatest common denominator of the notes' timestamps, pruning outliers as necessary. This allows keeping in sync with songs with triplets, quintuplets, or any other measurement not divisible by a power of 2. The algorithm falls back to unsynced playback if a good timing candidate cannot be found, which allows songs with tempo changes or that do not follow their defined time signature at all to still function.
  - Note Volume: Notes are automatically spread out further from the centreline where the player is, depending on their volume/velocity. This additionally respects the direction of the notes' panning if specified.

### Additional notes
- Hyperchoron's design is focused on importing and exporting to vanilla Minecraft as accurately and as reasonably as possible. That means, there are limitations when attempting to convert songs with a much higher speed or many stacked notes; in both cases the notes will be compacted and quantised.
- Please note that conversion quality may vary significantly between different versions of Hyperchoron. This is an unavoidable nuance that comes with attempting to provide a one-size-fits-all solution for MIDI files; often making one thing sound better will make another sound worse.
- Here is a video showcasing some of the example outputs in an earlier version: https://youtu.be/Vtmh1Qi0w9s, and here is an example of the latest version with volume control: https://youtu.be/XJlirm0joc4
