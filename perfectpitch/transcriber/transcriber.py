import torch

from perfectpitch.onsetsdetector.model import OnsetsDetector
from perfectpitch.utils.data import pianoroll_to_notesequence


class Transcriber:
    def __init__(self, onsets_detector_path):
        self.onsets_detector = OnsetsDetector()
        self.onsets_detector.load_state_dict(
            torch.load(onsets_detector_path, map_location=torch.device("cpu"))
        )

    def __call__(self, spec):
        spec = spec.unsqueeze(0)

        onsets_logits = self.onsets_detector(spec)
        onsets_logits = onsets_logits.squeeze(0)
        onsets = torch.zeros_like(onsets_logits)
        onsets[onsets_logits > 0] = 1

        actives = torch.zeros_like(onsets)
        offsets = torch.zeros_like(onsets)
        velocities = torch.zeros_like(onsets)

        return pianoroll_to_notesequence(actives, onsets, offsets, velocities)
