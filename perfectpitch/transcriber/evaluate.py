from collections import defaultdict

import mir_eval
from tqdm import tqdm

from perfectpitch.dataset.transcription_dataset import TranscriptionDataset
from perfectpitch.utils.data import notesequence_to_pianoroll, pianoroll_to_notesequence


def _evalute(ref_notesequence, est_pianoroll):
    ref_pitches = mir_eval.util.midi_to_hz(ref_notesequence["pitches"].numpy())
    ref_intervals = ref_notesequence["intervals"].numpy()
    ref_velocities = ref_notesequence["velocities"].numpy()

    est_notesequence = pianoroll_to_notesequence(
        est_pianoroll["actives"],
        est_pianoroll["onsets"],
        est_pianoroll["offsets"],
        est_pianoroll["velocities"],
    )
    est_pitches = mir_eval.util.midi_to_hz(est_notesequence["pitches"].numpy())
    est_intervals = est_notesequence["intervals"].numpy()
    est_velocities = est_notesequence["velocities"].numpy()

    results = {}
    score = mir_eval.transcription.evaluate(
        ref_intervals, ref_pitches, est_intervals, est_pitches
    )
    results.update({f"transcription/{key}": value for key, value in score.items()})
    score = mir_eval.transcription_velocity.evaluate(
        ref_intervals,
        ref_pitches,
        ref_velocities,
        est_intervals,
        est_pitches,
        est_velocities,
    )
    results.update(
        {f"transcription_velocity/{key}": value for key, value in score.items()}
    )
    return results


def evaluate_transcriber(dataset_path):
    dataset = TranscriptionDataset(dataset_path)
    results = defaultdict(list)
    for sample in tqdm(dataset, desc=f"evaluating {dataset_path}"):
        notesequence = sample["notesequence"]
        pianoroll = notesequence_to_pianoroll(
            notesequence["pitches"],
            notesequence["intervals"],
            notesequence["velocities"],
        )
        score = _evalute(notesequence, pianoroll)
        for key, value in score.items():
            results[key].append(value)

    print("{: >10} {: >10} {: >10}".format("name", "mean", "min"))
    for key, value in results.items():
        print(
            "{: >10} {: >10} {: >10}".format(key, sum(value) / len(value), min(value))
        )
