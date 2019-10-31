import operator

import numpy as np
import librosa
import mido

from perfectpitch import constants


def audio_to_spec(audio):
    mel = librosa.feature.melspectrogram(
        audio,
        constants.SAMPLE_RATE,
        hop_length=constants.SPEC_HOP_LENGTH,
        fmin=30.0,
        n_mels=constants.SPEC_N_BINS,
        htk=True,
    )
    return mel.T


def save_notesequence(path, pitches, intervals, velocities):
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


def notesequence_to_pianoroll(pitches, intervals, velocities, max_velocity):
    frame_duration = constants.SPEC_HOP_LENGTH / constants.SAMPLE_RATE
    num_frames = int(intervals.max() / frame_duration) + 1
    num_pitches = constants.MAX_PITCH - constants.MIN_PITCH + 1

    notes = sorted(
        [
            (pitch, start_time, end_time, velocity)
            for (pitch, (start_time, end_time), velocity) in zip(
                pitches, intervals, velocities
            )
        ],
        key=operator.itemgetter(1),
    )

    active_frames = np.zeros([num_frames, num_pitches], dtype=np.float32)
    onset_frames = np.zeros_like(active_frames)
    offset_frames = np.zeros_like(active_frames)
    velocity_frames = np.zeros_like(active_frames)

    for pitch, start_time, end_time, velocity in notes:
        if pitch > constants.MAX_PITCH or pitch < constants.MIN_PITCH:
            continue

        pitch_index = pitch - constants.MIN_PITCH

        start_frame = int(start_time / frame_duration)
        end_frame = int(end_time / frame_duration)
        if start_frame == end_frame:
            continue

        active_frames[start_frame:end_frame, pitch_index] = 1
        onset_frames[start_frame, pitch_index] = 1
        offset_frames[end_frame, pitch_index] = 1
        velocity_frames[start_frame:end_frame, pitch_index] = velocity / max_velocity

    return {
        "actives": active_frames,
        "onsets": onset_frames,
        "offsets": offset_frames,
        "velocities": velocity_frames,
    }


def pianoroll_to_notesequence(actives, onsets, offsets, velocities):
    frame_duration = constants.SPEC_HOP_LENGTH / constants.SAMPLE_RATE

    notes = []
    pitch_start_frame = {}

    def start_note(pitch, start_frame):
        pitch_start_frame[pitch] = start_frame

    def end_note(pitch, end_frame):
        start_frame = pitch_start_frame.pop(pitch)
        if start_frame == end_frame:
            return

        notes.append(
            (
                pitch + constants.MIN_PITCH,
                start_frame * frame_duration,
                end_frame * frame_duration,
                np.clip(velocities[start_frame, pitch], 0, 1) * 80 + 10,
            )
        )

    for frame in range(actives.shape[0]):
        for pitch in range(actives.shape[1]):
            is_onset = onsets[frame, pitch] > 0.5
            is_previous_onset = onsets[frame - 1, pitch] if frame > 0 else False
            is_offset = offsets[frame, pitch] > 0.5
            is_started = pitch in pitch_start_frame

            is_active = actives[frame, pitch] > 0.5
            is_active = is_active and not is_offset
            is_active = is_active or is_onset

            if is_onset and not is_started:
                start_note(pitch, frame)
            elif is_onset and is_started and not is_previous_onset:
                end_note(pitch, frame)
                start_note(pitch, frame)
            elif not is_active and is_started:
                end_note(pitch, frame)

    for pitch in list(pitch_start_frame.keys()):
        end_note(pitch, actives.shape[0])

    return {
        "pitches": np.array([note[0] for note in notes], dtype=np.int8),
        "intervals": np.array([(note[1], note[2]) for note in notes], dtype=np.float32),
        "velocities": np.array([note[3] for note in notes], dtype=np.int8),
    }
