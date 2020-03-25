import operator

import numpy as np
import librosa
import mido

from perfectpitch import constants


def load_spec(path):
    audio, _ = librosa.load(path, constants.SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        audio,
        constants.SAMPLE_RATE,
        hop_length=constants.SPEC_HOP_LENGTH,
        fmin=30.0,
        n_mels=constants.SPEC_N_BINS,
        htk=True,
    )
    return mel.astype(np.float32).T


def load_notesequence(path):
    midi = mido.MidiFile(path)
    if midi.type != 0:
        raise NotImplementedError("midi file type is not 0")

    notes = []
    actived = {}
    sustain = False
    time = 0

    for event in midi:
        time += event.time
        if event.type == "note_on":
            pitch = event.note
            if pitch in actived:
                onset = actived[pitch][0]
                offset = time
                velocity = actived[pitch][2]
                notes.append((pitch, onset, offset, velocity))
            actived[pitch] = (time, None, event.velocity)
        elif event.type == "note_off":
            pitch = event.note
            onset = actived[pitch][0]
            offset = time
            velocity = actived[pitch][2]
            actived[pitch] = (onset, offset, velocity)
            if not sustain:
                notes.append((pitch, onset, offset, velocity))
                del actived[pitch]
        elif (
            event.type == "control_change" and event.control == 64 and event.value >= 64
        ):
            sustain = True
        elif (
            event.type == "control_change" and event.control == 64 and event.value < 64
        ):
            sustain = False
            pitches = [p for p, v in actived.items() if v[1] is not None]
            for pitch in pitches:
                onset, offset, velocity = actived[pitch]
                notes.append((pitch, onset, offset, velocity))
                del actived[pitch]

    for pitch, (onset, _, velocity) in actived.items():
        notes.append((pitch, onset, time, velocity))

    return {
        "pitches": np.array([note[0] for note in notes], dtype=np.int8),
        "intervals": np.array([(note[1], note[2]) for note in notes], dtype=np.float32),
        "velocities": np.array([note[3] for note in notes], dtype=np.int8),
    }


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


def notesequence_to_pianoroll(pitches, intervals, velocities):
    frame_duration = constants.SPEC_HOP_LENGTH / constants.SAMPLE_RATE
    num_frames = int(intervals.max() / frame_duration) + 1
    num_pitches = constants.MAX_PITCH - constants.MIN_PITCH + 1
    velocity_max = velocities.max().tolist()

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
        velocity_frames[start_frame:end_frame, pitch_index] = velocity / velocity_max

    return {
        "actives": active_frames,
        "onsets": onset_frames,
        "offsets": offset_frames,
        "velocities": velocity_frames,
    }


def pianoroll_to_notesequence(actives, onsets, offsets, velocities):
    frame_duration = constants.SPEC_HOP_LENGTH / constants.SAMPLE_RATE
    notes = []
    start_frame = None

    for pitch in range(actives.shape[1]):
        start_frame = None
        for frame in range(actives.shape[0]):
            is_onset = onsets[frame, pitch] >= 0.5
            is_previous_onset = onsets[frame - 1, pitch] >= 0.5 if frame > 0 else False
            is_offset = offsets[frame, pitch] >= 0.5
            is_started = start_frame is not None

            is_active = actives[frame, pitch] >= 0.5
            is_active = is_active and not is_offset
            is_active = is_active or is_onset

            if is_onset and not is_started:
                start_frame = frame
            elif is_onset and is_started and not is_previous_onset:
                notes.append(
                    (
                        pitch + constants.MIN_PITCH,
                        start_frame * frame_duration,
                        frame * frame_duration,
                        np.clip(velocities[start_frame, pitch], 0, 1) * 80 + 10,
                    )
                )
                start_frame = frame
            elif not is_active and is_started:
                notes.append(
                    (
                        pitch + constants.MIN_PITCH,
                        start_frame * frame_duration,
                        frame * frame_duration,
                        np.clip(velocities[start_frame, pitch], 0, 1) * 80 + 10,
                    )
                )
                start_frame = None
        if start_frame is not None:
            notes.append(
                (
                    pitch + constants.MIN_PITCH,
                    start_frame * frame_duration,
                    actives.shape[0] * frame_duration,
                    np.clip(velocities[start_frame, pitch], 0, 1) * 80 + 10,
                )
            )

    return {
        "pitches": np.array([note[0] for note in notes], dtype=np.int8),
        "intervals": np.array([(note[1], note[2]) for note in notes], dtype=np.float32),
        "velocities": np.array([note[3] for note in notes], dtype=np.int8),
    }
