import numpy as np
import torch

from perfectpitch.onsetsdetector.model import OnsetsDetector
from perfectpitch.utils.transcription import pianoroll_to_transcription


class Transcriber:
    def __init__(self, onsets_detector_path, device):
        self._device = torch.device(device)

        self._onsets_detector = OnsetsDetector()
        self._onsets_detector.load_state_dict(
            torch.load(onsets_detector_path, map_location=torch.device("cpu"))
        )
        self._onsets_detector.to(self._device)
        self._onsets_detector.eval()

    def __call__(self, spec, posenc):
        with torch.no_grad():
            spec = torch.from_numpy(spec).to(self._device).unsqueeze(1)
            posenc = torch.from_numpy(posenc).to(self._device).unsqueeze(1)

            onsets_logits = self._onsets_detector(spec, posenc).squeeze(1).cpu().numpy()

        onsets = np.zeros_like(onsets_logits)
        onsets[onsets_logits > 0] = 1

        actives = np.zeros_like(onsets)
        offsets = np.zeros_like(onsets)
        velocities = np.zeros_like(onsets)

        return pianoroll_to_transcription(actives, onsets, offsets, velocities)
