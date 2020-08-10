from setuptools import setup, find_packages

setup(
    name="perfectpitch",
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
        "numpy>=1.18.0",
        "soundfile >= 0.9.0",
        "numba==0.48.0",  # pin numba version for librosa
        "librosa>=0.8.0",
        "mido>=1.2.0",
        "torch>=1.5.0",
        "axial-positional-embedding>=0.2.0",
        "reformer_pytorch>=1.1.0",
        "mir_eval>=0.6.0",
        "tqdm>=4.42.0",
    ],
)
