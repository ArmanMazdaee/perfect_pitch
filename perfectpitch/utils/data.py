import math
import operator

import torch
import torchaudio
import mido

from perfectpitch import constants


def load_audio(path):
    audio, sample_rate = torchaudio.load(path)
    audio = audio.mean(dim=0, keepdim=True)
    audio = torchaudio.compliance.kaldi.resample_waveform(
        waveform=audio, orig_freq=sample_rate, new_freq=constants.SAMPLE_RATE
    )
    return audio.view([-1])


def save_audio(path, audio):
    torchaudio.save(path, audio.view([1, -1]), constants.SAMPLE_RATE)


def audio_to_spec(audio):
    win_length = 1024
    pad = (
        math.floor((win_length - constants.SPEC_HOP_LENGTH) / 2),
        math.ceil((win_length - constants.SPEC_HOP_LENGTH) / 2),
    )
    audio = torch.nn.functional.pad(audio.view([1, 1, -1]), pad, "reflect")

    pitches = torch.arange(constants.MIN_PITCH, constants.MAX_PITCH + 1)
    frequencies = 440 * (2 ** ((pitches - 69) / 12.0))
    real_points = torch.stack(
        [
            torch.linspace(
                start=0,
                end=f * 2 * math.pi * win_length / constants.SAMPLE_RATE,
                steps=win_length,
            )
            for f in frequencies
        ]
    )
    imag_points = real_points + (math.pi / 2)

    real_kernel = (torch.sin(real_points) * torch.hann_window(win_length)).unsqueeze(1)
    imag_kernel = (torch.sin(imag_points) * torch.hann_window(win_length)).unsqueeze(1)

    real_spec = torch.nn.functional.conv1d(
        audio, real_kernel, stride=constants.SPEC_HOP_LENGTH
    ).squeeze(0)
    imag_spec = torch.nn.functional.conv1d(
        audio, imag_kernel, stride=constants.SPEC_HOP_LENGTH
    ).squeeze(0)
    spec = torch.sqrt((real_spec ** 2) + (imag_spec ** 2))
    return spec


def spec_to_audio(spec):
    repeated_spec = spec.repeat_interleave(constants.SPEC_HOP_LENGTH, dim=1)
    pitches = torch.arange(constants.MIN_PITCH, constants.MAX_PITCH + 1)
    frequencies = 440 * (2 ** ((pitches - 69) / 12.0))
    points = torch.stack(
        [
            torch.linspace(
                start=0,
                end=f * 2 * math.pi * repeated_spec.shape[1] / constants.SAMPLE_RATE,
                steps=repeated_spec.shape[1],
            )
            for f in frequencies
        ]
    )
    sins = torch.sin(points)
    audio = (sins * repeated_spec).sum(dim=0)
    audio = audio - audio.mean()
    audio = audio / audio.abs().max()
    return audio


def load_transcription(path):
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
        "pitches": torch.ByteTensor([note[0] for note in notes]),
        "intervals": torch.FloatTensor([(note[1], note[2]) for note in notes]),
        "velocities": torch.ByteTensor([note[3] for note in notes]),
    }


def save_transcription(path, pitches, intervals, velocities):
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)

    messages = []
    for index in range(len(pitches)):
        pitch = pitches[index].item()
        start = intervals[index][0].item()
        end = intervals[index][1].item()
        velocity = velocities[index].item()
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


def transcription_to_pianoroll(pitches, intervals, velocities, num_frames=None):
    frame_duration = constants.SPEC_HOP_LENGTH / constants.SAMPLE_RATE
    num_pitches = constants.MAX_PITCH - constants.MIN_PITCH + 1
    if num_frames is None:
        num_frames = int(intervals.max() / frame_duration) + 1

    pitches = pitches - constants.MIN_PITCH
    intervals = (intervals // frame_duration).long().clamp_max(num_frames - 1)
    velocities = velocities.float() / velocities.max()

    valid_indices = (
        (pitches >= 0) & (pitches < num_pitches) & (intervals[:, 0] != intervals[:, 1])
    )
    pitches = pitches[valid_indices]
    intervals = intervals[valid_indices]
    velocities = velocities[valid_indices]

    active_frames = torch.zeros([num_pitches, num_frames])
    onset_frames = torch.zeros_like(active_frames)
    offset_frames = torch.zeros_like(active_frames)
    velocity_frames = torch.zeros_like(active_frames)

    for index in range(len(pitches)):
        pitch = pitches[index].item()
        onset = intervals[index][0].item()
        offset = intervals[index][1].item()
        velocity = velocities[index].item()

        active_frames[pitch, onset:offset] = 1
        onset_frames[pitch, onset] = 1
        offset_frames[pitch, offset] = 1
        velocity_frames[pitch, onset:offset] = velocity

    return {
        "actives": active_frames,
        "onsets": onset_frames,
        "offsets": offset_frames,
        "velocities": velocity_frames,
    }


def pianoroll_to_transcription(actives, onsets, offsets, velocities):
    frame_duration = constants.SPEC_HOP_LENGTH / constants.SAMPLE_RATE
    num_pitches = actives.shape[0]
    num_frames = actives.shape[1]
    notes = []

    actives = actives.tolist()
    onsets = onsets.tolist()
    offsets = offsets.tolist()
    velocities = velocities.tolist()

    for pitch in range(num_pitches):
        start_frame = None
        for frame in range(num_frames):
            is_onset = onsets[pitch][frame] >= 0.5
            is_previous_onset = onsets[pitch][frame - 1] >= 0.5 if frame > 0 else False
            is_offset = offsets[pitch][frame] >= 0.5 or actives[pitch][frame] < 0.5

            if (is_offset and start_frame is not None) or (
                is_onset and start_frame is not None and not is_previous_onset
            ):
                notes.append(
                    (pitch, start_frame, frame, velocities[pitch][start_frame])
                )
                start_frame = None

            if is_onset and start_frame is None:
                start_frame = frame

        if start_frame is not None:
            notes.append(
                (pitch, start_frame, num_frames, velocities[pitch][start_frame])
            )

    pitches = torch.ByteTensor([note[0] for note in notes]) + constants.MIN_PITCH
    intervals = (
        torch.FloatTensor([(note[1], note[2]) for note in notes]) * frame_duration
    )
    velocities = torch.FloatTensor([note[3] for note in notes]).clamp(0, 1) * 80 + 10

    return {
        "pitches": pitches,
        "intervals": intervals,
        "velocities": velocities,
    }
