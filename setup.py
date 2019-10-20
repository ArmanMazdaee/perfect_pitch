import sys
from setuptools import setup, find_packages

if "--gpu" in sys.argv:
    GPU = True
    sys.argv.remove("--gpu")
else:
    GPU = False

setup(
    name="perfectpitch-gpu" if GPU else "perfectpitch",
    version="0.1.0",
    description="Automatic Music Transcription using deeplearning",
    long_description="",
    url="https://github.com/ArmanMazdaee/perfectpitch",
    author="Arman Mazdaee",
    author_email="arman.maz1373@gmail.com",
    license="MIT",
    packages=find_packages(),
    entry_points={"console_scripts": ["perfectpitch = perfectpitch:main"]},
    python_requires=">=3.6,<4.0",
    install_requires=[
        "tensorflow-gpu>=2.0.0,<3.0.0" if GPU else "tensorflow>=2.0.0,<3.0.0",
        "librosa>=0.7.0,<0.8.0",
        "mir_eval>=0.5.0,<0.6.0",
    ],
)
