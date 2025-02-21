![thumbnail](https://raw.githubusercontent.com/thomas-xin/hyperchoron/refs/heads/main/thumb.jpg)
# Instructions for use
### Installation
- Install [git](https://github.com/git-guides/install-git) and [python](https://www.python.org).
- Clone this repo:
`git clone https://github.com/thomas-xin/hyperchoron`
- Install dependencies:
`py -m pip install -r requirements.txt`
### Usage
```ini
py hyperchoron.py -h
usage:  [-h] [-i INPUT [INPUT ...]] [-o [OUTPUT ...]] [-t [TRANSPOSE]] [-s [SPEED]] [-sa [STRUM_AFFINITY]]
        [-d | --drums | --no-drums] [-c | --cheap | --no-cheap] [-x | --exclusive | --no-exclusive]

MIDI to Minecraft Note Block Converter

options:
  -h, --help            show this help message and exit
  -i, --input INPUT [INPUT ...]
                        Input file (.mid | .zip | .nbs)
  -o, --output [OUTPUT ...]
                        Output file (.mcfunction | .litematic | .nbs | .org)
  -t, --transpose [TRANSPOSE]
                        Transposes song up/down a certain amount of semitones, applied before instrument material mapping;
                        higher = higher pitched
  -s, --speed [SPEED]   Scales song speed up/down as a multiplier, applied before tempo sync; higher = faster
  -sa, --strum-affinity [STRUM_AFFINITY]
                        Increases or decreases threshold for sustained notes to be cut into discrete segments; higher = more
                        notes
  -d, --drums, --no-drums
                        Allows percussion channel. If disabled, the default MIDI percussion channel will be treated as a
                        regular instrument channel. Defaults to TRUE
  -c, --cheap, --no-cheap
                        Restricts the list of non-instrument blocks to a more survival-friendly set. Also enables compatibility
                        with previous versions of minecraft. May cause spacing issues with the sand/snare drum instruments.
                        Defaults to FALSE
  -x, --exclusive, --no-exclusive
                        Disables speed re-matching and strum quantisation, increases pitch bucket limit. Defaults to FALSE if
                        outputting to any Minecraft-related format, and included for compatibility with other export formats.
```
### Additional info
- The program takes one or more input and output files, and currently supports outputting to `.mcfunction` (a list of `/setblock` commands), `.litematic` (used by the litematica mod), or `.nbs` (Note Block Studio project).
- Multiple MIDI inputs (or a single zip input containing multiple MIDI files) will be treated as a stacked MIDI project, and the exported notes will be combined into a single output stream.
- Note that if exporting to `.mcfunction`, you will need to make some sort of template datapack to be able to load it in. It is recommended to perform the `/gamerule maxCommandChainLength 2147483647` command prior to pasting the note blocks to avoid longer songs being cut off.
- If you are intending to build the output schematic in vanilla survival mode, the `-c` option will force the program to use cobblestone for most of the structure. This removes the excessive use of decorative blocks such as beacons, crying obsidian and froglights, as well as heavy core as an instrument. Note that the latter may cause alignment issues arbitrarily depending on the complexity of the song, as sand requires a supporting block, and there are no blocks (besides customised player heads) in vanilla that can perform this role without also possibly silencing a note block that happens to be directly below.
- If you are exporting to `.nbs`, the output will be intercepted immediately before the schematic formatting stage, and the raw note blocks will be written. The file will be playable in Note Block Studio, however please note that its export feature does not support exporting 20Hz songs, meaning you will not be able to use it in vanilla Minecraft. Support for importing `.nbs` files into Hyperchoron is currently in progress.

# What is the purpose of another exporter like this?
- Converting music to minecraft note blocks programmatically has been a thing for a long time, the most popular program being Note Block Studio. This program is not intended to entirely replace them, and is meant to be a standalone feature.
- Hyperchoron's intent is to expand on the exporting capabilities of note block programs, using a more adaptable algorithm to produce more accurate recreations of songs while still staying within the boundaries of vanilla Minecraft.
  - Note Sustain: Automatically cut excessively long notes into segments to be replayed several times, preventing notes that are supposed to be sustained from fading out quickly. This adapts based on the volume of the current and adjacent notes, and will automatically offset chords into a strum to avoid excessive sudden loudness.
  - Instrument Mapping: Rather than map all MIDI instruments to a list of blocks and call it a day, this program has the full 6-octave range for all instruments, automatically swapping the materials if a note exceeds the current range. This is very important for most songs as the default range of 2 octaves in vanilla Minecraft means notes will frequently fall out of range. Clipping, dropping or wrapping notes at the boundary are valid methods of handling this, but they do not permit an accurate recreation of the full song.
  - Adding to the previous point, all drums are implemented separately, rather than using only the three drum instruments in vanilla (which end up drowning each other out), some percussion instruments are mapped to other instruments or mob heads, which allows for a greater variety of sound effects.
  - Pitch Bends: Interpret pitch bends from MIDI files, and automatically convert them into grace notes. This is very important for any MIDI file that includes this mechanic, as the pitch of notes involved will often be very wrong without it.
  - Polyphonic Budget: If there are too many notes being played at any given moment, the quietest notes and intermediate notes used for sustain will be discarded first.
  - Full Tick Rate: Using the scaffolding-on-trapdoor circuit, a 3-gametick delay can be achieved, which allows offsetting from the (usual) even-numbered ticks redstone components are capable of operating at. This means the full 20Hz tick speed of vanilla Minecraft can be accessed, allowing much greater timing precision.
  - Tempo Alignment: The tempo of songs is automatically synced to Minecraft's default tick rate not using the song's time signature, but rather the greatest common denominator of the notes' timestamps, pruning outliers as necessary. This allows keeping in sync with songs with triplets, quintuplets, or any other measurement not divisible by a power of 2. The algorithm falls back to unsynced playback if a good timing candidate cannot be found, which allows songs with tempo changes or that do not follow their defined time signature at all to still function.
- Hyperchoron's design is focused on importing and exporting to vanilla Minecraft as accurately and as reasonably as possible. That means, there are limitations when attempting to convert songs with a much higher speed or many stacked notes; in both cases the notes will be compacted and quantised. If your intent is not to play the song in Minecraft but rather just to make a cover of a song using Minecraft sounds, then this tool is most likely not for you.
- Screenshots and example exported outputs are provided; credit goes out to the original creators of the songs as well as MIDI transcriptions where applicable.
  - Most of the miscellaneous MIDI files can be found on https://www.vgmusic.com; the black MIDIs' debuts can be found by searching their name on YouTube, the Cave Story MIDIs are directly exported from the original game's files, and the Undertale MIDIs can be found here: https://youtu.be/n138Qs-pvb8
  - The examples are for demonstration purposes only; the creator of this repo does not claim ownership over the songs and transcriptions involved.
  - Please note that conversion quality may vary significantly between the examples, particularly across different versions of Hyperchoron. This is an unavoidable nuance that comes with attempting to provide a one-size-fits-all solution for MIDI files; often making one thing sound better will make another sound worse.
  - Here is a video showcasing some of the examples: https://youtu.be/Vtmh1Qi0w9s
