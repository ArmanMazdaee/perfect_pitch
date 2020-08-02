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

    def _get_splits(self, length):
        step = 2000
        pad = 1000
        for start in range(0, length, step):
            end = min(start + step, length)
            pad_before = min(start, pad)
            pad_after = min(length - end, pad)
            yield (start - pad_before, end + pad_after, pad_before, pad_after)

    def _forward(self, spec):
        with torch.no_grad():
            spec = torch.from_numpy(spec).to(self._device).unsqueeze(1)

            onsets_logits = self._onsets_detector(spec).squeeze(1).cpu().numpy()

        onsets = np.zeros_like(onsets_logits)
        onsets[onsets_logits > 0] = 1

        actives = np.zeros_like(onsets)
        offsets = np.zeros_like(onsets)
        velocities = np.zeros_like(onsets)
        return actives, onsets, offsets, velocities

    def __call__(self, spec):
        actives_parts = []
        onsets_parts = []
        offsets_parts = []
        velocities_parts = []
        for start, end, pad_before, pad_after in self._get_splits(len(spec)):
            actives_part, onsets_part, offsets_part, velocities_part = self._forward(
                spec[start:end]
            )
            actives_parts.append(actives_part[pad_before:-pad_after])
            onsets_parts.append(onsets_part[pad_before:-pad_after])
            offsets_parts.append(offsets_part[pad_before:-pad_after])
            velocities_parts.append(velocities_part[pad_before:-pad_after])

        actives = np.concatenate(actives_parts)
        onsets = np.concatenate(onsets_parts)
        offsets = np.concatenate(offsets_parts)
        velocities = np.concatenate(velocities_parts)
        return pianoroll_to_transcription(actives, onsets, offsets, velocities)
