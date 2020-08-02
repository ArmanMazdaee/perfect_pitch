from collections import defaultdict

import mir_eval
from tqdm import tqdm

from perfectpitch.dataset.transcription_dataset import TranscriptionDataset
from .transcriber import Transcriber


def _evalute(ref_transcription, est_transcription):
    ref_pitches = mir_eval.util.midi_to_hz(ref_transcription["pitches"])
    ref_intervals = ref_transcription["intervals"]

    est_pitches = mir_eval.util.midi_to_hz(est_transcription["pitches"])
    est_intervals = est_transcription["intervals"]

    results = {}
    score = mir_eval.transcription.evaluate(
        ref_intervals, ref_pitches, est_intervals, est_pitches
    )
    results.update({f"transcription/{key}": value for key, value in score.items()})
    return results


def evaluate_transcriber(dataset_path, onsets_detector_path, device):
    dataset = TranscriptionDataset(dataset_path, shuffle=False)
    transcriber = Transcriber(onsets_detector_path, device)
    results = defaultdict(list)
    for sample in tqdm(dataset, desc="evaluating"):
        spec = sample["spec"]
        ref_transcription = sample["transcription"]
        est_transcription = transcriber(spec)
        score = _evalute(ref_transcription, est_transcription)
        for key, value in score.items():
            results[key].append(value)

    print("{: >50} {: >10} {: >10}".format("name", "mean", "min"))
    for key, value in results.items():
        print(
            "{: >50} {: >10} {: >10}".format(key, sum(value) / len(value), min(value))
        )
