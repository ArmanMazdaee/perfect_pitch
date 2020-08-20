import numpy as np
import tensorflow as tf
import librosa
import note_seq

from perfectpitch import constants


def load_spec(path):
    with tf.io.gfile.GFile(path, "rb") as wav_file:
        audio, _ = librosa.load(wav_file, sr=constants.SAMPLE_RATE)

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=constants.SAMPLE_RATE,
        hop_length=constants.SPEC_HOP_LENGTH,
        fmin=30.0,
        n_mels=constants.SPEC_DIM,
        htk=True,
    )
    return spec.T


def load_transcription(path):
    with tf.io.gfile.GFile(path, "rb") as midi_file:
        midi = midi_file.read()

    notesequence = note_seq.midi_io.midi_to_note_sequence(midi)
    notesequence = note_seq.apply_sustain_control_changes(notesequence)

    return {
        "pitches": np.array([note.pitch for note in notesequence.notes], np.int8),
        "start_times": np.array(
            [note.start_time for note in notesequence.notes], np.float32
        ),
        "end_times": np.array(
            [note.end_time for note in notesequence.notes], np.float32
        ),
        "velocities": np.array([note.velocity for note in notesequence.notes], np.int8),
    }
