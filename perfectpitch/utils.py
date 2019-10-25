import operator
import io
import math

import numpy as np
import librosa
import mido

from perfectpitch import constants
from perfectpitch.protobuf import music_pb2


def audio_from_string(serialized):
    audio_file = io.BytesIO(serialized)
    wav, _ = librosa.load(audio_file, constants.SAMPLE_RATE)
    return librosa.to_mono(wav)


def spec_from_audio(audio):
    mel = librosa.feature.melspectrogram(
        audio,
        constants.SAMPLE_RATE,
        hop_length=constants.SPEC_HOP_LENGTH,
        fmin=30.0,
        n_mels=constants.SPEC_N_BINS,
        htk=True,
    )
    return mel.T


def velocity_range_from_string(serialized):
    velocity_range = music_pb2.VelocityRange.FromString(serialized)
    return (
        np.array(velocity_range.min, dtype=np.int8),
        np.array(velocity_range.max, dtype=np.int8),
    )


def notesequence_from_string(serialized):
    transcription = music_pb2.NoteSequence.FromString(serialized)

    sustains_ranges = []
    sustain_start = None
    control_changes = sorted(
        transcription.control_changes, key=operator.attrgetter("time")
    )
    for control_change in control_changes:
        if control_change.control_number != 64:
            continue
        if sustain_start is None and control_change.control_value >= 64:
            sustain_start = control_change.time
        elif sustain_start is not None and control_change.control_value < 64:
            sustains_ranges.append((sustain_start, control_change.time))
            sustain_start = None
    if sustain_start is not None:
        sustain_end = max(sustain_start, transcription.total_time)
        sustains_ranges.append((sustain_start, sustain_end))
    sustains_ranges.append((math.inf, math.inf))

    active_sustain = 0
    sustain_start, sustain_end = sustains_ranges[active_sustain]
    notes = sorted(transcription.notes, key=operator.attrgetter("end_time"))
    pitches = np.zeros([len(notes)], dtype=np.int8)
    intervals = np.zeros([len(notes), 2], dtype=np.float32)
    velocities = np.zeros([len(notes)], dtype=np.int8)

    for i, note in enumerate(notes):
        while note.end_time >= sustain_end:
            active_sustain += 1
            sustain_start, sustain_end = sustains_ranges[active_sustain]

        pitches[i] = note.pitch
        intervals[i, 0] = note.start_time
        if sustain_start < note.end_time < sustain_end:
            intervals[i, 1] = sustain_end
        else:
            intervals[i, 1] = note.end_time
        velocities[i] = note.velocity

    return pitches, intervals, velocities


def notesequence_to_midi(path, pitches, intervals, velocities):
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)

    messages = []
    for pitch, (start, end), velocity in zip(pitches, intervals, velocities):
        messages.append(
            mido.Message("note_on", note=pitch, time=start, velocity=velocity)
        )
        messages.append(
            mido.Message("note_off", note=pitch, time=end, velocity=velocity)
        )
    messages.sort(key=operator.attrgetter("time"))

    time = 0
    for message in messages:
        time_delta = message.time - time
        tick = int(mido.second2tick(time_delta, 480, 500000))
        track.append(message.copy(time=tick))
        time = message.time

    midi.save(path)
