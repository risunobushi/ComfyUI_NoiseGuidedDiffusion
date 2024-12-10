import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from noise_guided_diffusion import NoiseGuidedDiffusion

NODE_CLASS_MAPPINGS = {
    "NoiseGuidedDiffusion": NoiseGuidedDiffusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseGuidedDiffusion": "Noise Guided Diffusion",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
