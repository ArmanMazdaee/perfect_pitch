import torch

from perfectpitch.onsetsdetector.model import OnsetsDetector
from perfectpitch.utils.data import pianoroll_to_transcription


class Transcriber:
    def __init__(self, onsets_detector_path, device):
        self._device = torch.device(device)

        self._onsets_detector = OnsetsDetector()
        self._onsets_detector.load_state_dict(
            torch.load(onsets_detector_path, map_location=torch.device("cpu"))
        )
        self._onsets_detector.to(self._device)
        self._onsets_detector.eval()

    def __call__(self, spec):
        with torch.no_grad():
            spec = spec.to(self._device)
            spec = spec.unsqueeze(1)

            onsets_logits = self._onsets_detector(spec)
            onsets_logits = onsets_logits.squeeze(1)
            onsets = torch.zeros_like(onsets_logits)
            onsets[onsets_logits > 0] = 1

        onsets = onsets.to(torch.device("cpu"))
        actives = torch.zeros_like(onsets)
        offsets = torch.zeros_like(onsets)
        velocities = torch.zeros_like(onsets)

        return pianoroll_to_transcription(actives, onsets, offsets, velocities)
