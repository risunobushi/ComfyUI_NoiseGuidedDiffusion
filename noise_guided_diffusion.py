import torch
import numpy as np
from scipy.spatial import Voronoi
import cv2

class NoiseGuidedDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "noise_type": (["perlin", "voronoi", "simplex", "white"],),
            "noise_scale": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1
            }),
            "noise_frequency": ("FLOAT", {
                "default": 1.0,
                "min": 0.1,
                "max": 10.0,
                "step": 0.1
            }),
            "octaves": ("INT", {
                "default": 3,
                "min": 1,
                "max": 8,
                "step": 1
            }),
            "seed": ("INT", {
                "default": 0,
                "min": 0,
                "max": 0xffffffffffffffff
            })
        }}
    
    RETURN_TYPES = ("MODEL", "MASK",)  # Added MASK for preview
    RETURN_NAMES = ("model", "noise_preview",)
    FUNCTION = "apply"
    CATEGORY = "sampling"
    
    def __init__(self):
        self.noise_map = None
        self.current_seed = None
        self.current_scale = None
        self.current_freq = None
        self.current_type = None
        self.current_octaves = None

    def generate_perlin_noise(self, h, w):
        x = np.linspace(0, self.current_freq, w)
        y = np.linspace(0, self.current_freq, h)
        xx, yy = np.meshgrid(x, y)
        
        noise = np.zeros((h, w))
        amplitude = 1.0
        frequency = 1.0
        
        for _ in range(self.current_octaves):
            phase_x = np.random.rand() * 100
            phase_y = np.random.rand() * 100
            noise += amplitude * np.sin(2 * np.pi * (xx * frequency + phase_x)) * \
                    np.sin(2 * np.pi * (yy * frequency + phase_y))
            amplitude *= 0.5
            frequency *= 2
        
        return noise

    def generate_voronoi_noise(self, h, w):
        # Generate random points
        num_points = int(max(h, w) * self.current_freq / 2)
        points = np.random.rand(num_points, 2)
        points[:, 0] *= w
        points[:, 1] *= h
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        noise = np.zeros((h, w))
        
        # Calculate distances to nearest points
        for px, py in points:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            noise = np.maximum(noise, 1 / (1 + dist * 0.1 * self.current_freq))
        
        return noise

    def generate_simplex_noise(self, h, w):
        # Simple approximation of simplex-like noise
        noise = np.zeros((h, w))
        scale = self.current_freq * 0.1
        
        for octave in range(self.current_octaves):
            freq = (2 ** octave) * scale
            amplitude = 0.5 ** octave
            
            # Generate base noise
            x = np.linspace(0, w * freq, w)
            y = np.linspace(0, h * freq, h)
            xx, yy = np.meshgrid(x, y)
            
            # Add rotated waves
            angles = np.linspace(0, np.pi, 3)
            for angle in angles:
                rotated_x = xx * np.cos(angle) - yy * np.sin(angle)
                rotated_y = xx * np.sin(angle) + yy * np.cos(angle)
                noise += amplitude * np.sin(rotated_x + rotated_y)
        
        return noise

    def generate_white_noise(self, h, w):
        return np.random.rand(h, w)

    def generate_noise_map(self, shape):
        np.random.seed(self.current_seed)
        h, w = shape[-2:]
        
        # Select noise generation method
        if self.current_type == "perlin":
            noise = self.generate_perlin_noise(h, w)
        elif self.current_type == "voronoi":
            noise = self.generate_voronoi_noise(h, w)
        elif self.current_type == "simplex":
            noise = self.generate_simplex_noise(h, w)
        else:  # white noise
            noise = self.generate_white_noise(h, w)

        # Normalize to [0, 1] range
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Apply scale factor
        noise = noise * self.current_scale
        
        # Convert to torch tensor
        noise_map = torch.from_numpy(noise).float()
        if len(shape) > 2:
            noise_map = noise_map.unsqueeze(0).unsqueeze(0)
            noise_map = noise_map.expand(shape)
        
        return noise_map
    
    def apply(self, model, noise_type, noise_scale, noise_frequency, octaves, seed):
        model = model.clone()
        self.current_seed = seed
        self.current_scale = noise_scale
        self.current_freq = noise_frequency
        self.current_type = noise_type
        self.current_octaves = octaves
        
        # Generate a preview noise map (1-channel, 512x512)
        preview_shape = (1, 1, 512, 512)
        preview_noise = self.generate_noise_map(preview_shape)
        
        # Ensure preview noise is on the same device as the model
        preview_noise = preview_noise.to(device=model.device)
        
        model.set_model_denoise_mask_function(self.forward)
        return (model, preview_noise)
    
    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        
        # Generate or retrieve noise map
        if (self.noise_map is None or 
            self.noise_map.shape != denoise_mask.shape):
            self.noise_map = self.generate_noise_map(denoise_mask.shape)
            # Ensure noise map is on the same device as denoise_mask
            self.noise_map = self.noise_map.to(device=denoise_mask.device)
        
        # Calculate sigma parameters
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]
        
        # Convert to timesteps
        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])
        
        # Calculate threshold based on current timestep
        base_threshold = (current_ts - ts_to) / (ts_from - ts_to)
        
        # Modify threshold using noise map
        modified_threshold = base_threshold + self.noise_map * (1 - base_threshold)
        
        # Return binary mask
        return (denoise_mask >= modified_threshold).to(denoise_mask.dtype)

NODE_CLASS_MAPPINGS = {
    "NoiseGuidedDiffusion": NoiseGuidedDiffusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseGuidedDiffusion": "Noise Guided Diffusion",
}
