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
        "torch>=1.5.0,<2.0.0",
        "torchaudio>=0.5.0,<0.6.0",
        "tqdm>=4.42.0,<5.0.0",
        "mido>=1.2.0,<2.0.0",
        "mir_eval>=0.5.0,<0.6.0",
    ],
)
