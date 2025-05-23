from pathlib import Path

from setuptools import setup

import versioneer

long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name="hyperchoron",
    python_requires=">=3.10",
    description="Lightweight MIDI-Tracker-DAW converter and Minecraft Note Block exporter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Thomas Xin",
    url="https://github.com/thomas-xin/hyperchoron",
    packages=[
        "hyperchoron",
    ],
    entry_points={
        "console_scripts": [
            "hyperchoron = hyperchoron.cli:main",
        ]
    },
    license="MIT",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=[
        "audio-separator>=0.33.0",
        "json5>=0.12.0",
        "librosa>=0.11.0",
        "litemapy>=0.10.0b0",
        "nbtlib>=2.0.4",
        "py_midicsv>=4.1.2",
        "pynbs>=1.1.0",
        "tqdm>=4.67.1"
    ],
)
