from collections import defaultdict

import mir_eval
from tqdm import tqdm

from perfectpitch.dataset.transcription_dataset import TranscriptionDataset
from .transcriber import Transcriber


def _evalute(ref_notesequence, est_notesequence):
    ref_pitches = mir_eval.util.midi_to_hz(ref_notesequence["pitches"].numpy())
    ref_intervals = ref_notesequence["intervals"].numpy()

    est_pitches = mir_eval.util.midi_to_hz(est_notesequence["pitches"].numpy())
    est_intervals = est_notesequence["intervals"].numpy()

    results = {}
    score = mir_eval.transcription.evaluate(
        ref_intervals, ref_pitches, est_intervals, est_pitches
    )
    results.update({f"transcription/{key}": value for key, value in score.items()})
    return results


def evaluate_transcriber(dataset_path, onsets_detector_path):
    dataset = TranscriptionDataset(dataset_path, shuffle=False)
    num_steps = sum(1 for _ in dataset)
    transcriber = Transcriber(onsets_detector_path)
    results = defaultdict(list)
    for spec, ref_notesequence in tqdm(dataset, desc="evaluating", total=num_steps):
        est_notesequence = transcriber(spec)
        score = _evalute(ref_notesequence, est_notesequence)
        for key, value in score.items():
            results[key].append(value)

    print("{: >50} {: >10} {: >10}".format("name", "mean", "min"))
    for key, value in results.items():
        print(
            "{: >50} {: >10} {: >10}".format(key, sum(value) / len(value), min(value))
        )
