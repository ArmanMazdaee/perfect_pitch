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
    return mel.astype(np.float32)


def load_notesequence(path):
    midi = mido.MidiFile(path)

    if midi.type != 0 and midi.type != 1:
        raise NotImplementedError("midi file type should be 0 or 1")

    time = 0
    events = []
    for event in midi:
        time += event.time
        if event.type == "note_on" and event.velocity == 0:
            events.append(mido.Message(type="note_off", note=event.note, time=time))
        else:
            events.append(event.copy(time=time))
    events.append(mido.Message(type="control_change", control=64, value=0, time=time))

    notes = []
    actived = {}
    sustained = {}
    sustain = False
    for index, event in enumerate(events):
        if event.type == "note_on":
            if event.note in actived:
                continue
            if sustain and event.note in sustained:
                onset_event = events[sustained[event.note]]
                notes.append(
                    (event.note, onset_event.time, event.time, onset_event.velocity)
                )
                del sustained[event.note]
            actived[event.note] = index

        elif event.type == "note_off":
            if event.note not in actived:
                continue
            if sustain:
                sustained[event.note] = actived[event.note]
            else:
                onset_event = events[actived[event.note]]
                notes.append(
                    (event.note, onset_event.time, event.time, onset_event.velocity)
                )
            del actived[event.note]

        elif event.type == "control_change" and event.control == 64:
            if event.value >= 64 and not sustain:
                sustain = True
            elif event.value < 64 and sustain:
                for onset_index in sustained.values():
                    onset_event = events[onset_index]
                    notes.append(
                        (
                            onset_event.note,
                            onset_event.time,
                            event.time,
                            onset_event.velocity,
                        )
                    )
                sustained.clear()

    if len(sustained) != 0:
        raise RuntimeError("Some notes left sustained at the end of midi file")
    if len(actived) != 0:
        raise RuntimeError("Some notes left actived at the end of midi file")

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

    active_frames = np.zeros([num_pitches, num_frames], dtype=np.float32)
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

        active_frames[pitch_index, start_frame:end_frame] = 1
        onset_frames[pitch_index, start_frame] = 1
        offset_frames[pitch_index, end_frame] = 1
        velocity_frames[pitch_index, start_frame:end_frame] = velocity / velocity_max

    return {
        "actives": active_frames,
        "onsets": onset_frames,
        "offsets": offset_frames,
        "velocities": velocity_frames,
    }


def pianoroll_to_notesequence(actives, onsets, offsets, velocities):
    frame_duration = constants.SPEC_HOP_LENGTH / constants.SAMPLE_RATE
    notes = []

    for pitch in range(actives.shape[0]):
        start_frame = None
        for frame in range(actives.shape[1]):
            is_onset = onsets[pitch, frame] >= 0.5
            is_previous_onset = onsets[pitch, frame - 1] >= 0.5 if frame > 0 else False
            is_offset = offsets[pitch, frame] >= 0.5 or actives[pitch, frame] < 0.5

            if (is_offset and start_frame is not None) or (
                is_onset and start_frame is not None and not is_previous_onset
            ):
                notes.append(
                    (
                        pitch + constants.MIN_PITCH,
                        start_frame * frame_duration,
                        frame * frame_duration,
                        np.clip(velocities[pitch, start_frame], 0, 1) * 80 + 10,
                    )
                )
                start_frame = None

            if is_onset and start_frame is None:
                start_frame = frame

        if start_frame is not None:
            notes.append(
                (
                    pitch + constants.MIN_PITCH,
                    start_frame * frame_duration,
                    actives.shape[1] * frame_duration,
                    np.clip(velocities[pitch, start_frame], 0, 1) * 80 + 10,
                )
            )

    return {
        "pitches": np.array([note[0] for note in notes], dtype=np.int8),
        "intervals": np.array([(note[1], note[2]) for note in notes], dtype=np.float32),
        "velocities": np.round([note[3] for note in notes]).astype(np.int8),
    }
