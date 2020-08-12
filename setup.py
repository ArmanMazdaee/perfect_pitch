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
        "numpy>=1.18.0,<1.19.0",
        "tensorflow>=2.3.0",
        "note-seq",
        "librosa>=0.8.0",
        "mir_eval>=0.6.0",
        "tqdm>=4.42.0",
    ],
)
