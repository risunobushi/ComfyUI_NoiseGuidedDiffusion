import torch
import numpy as np
import cv2
from scipy.spatial import Voronoi

class NoiseGuidedDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "image": ("IMAGE",),  # Added input image for detail detection
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
            "detail_strength": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 5.0,
                "step": 0.1
            }),
            "black_level": ("FLOAT", {
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            }),
            "white_level": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
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
    
    RETURN_TYPES = ("MODEL", "MASK", "MASK",)
    RETURN_NAMES = ("model", "noise_preview", "detail_mask",)
    FUNCTION = "apply"
    CATEGORY = "sampling"

    def __init__(self):
        self.noise_map = None
        self.detail_mask = None
        self.current_seed = None
        self.current_scale = None
        self.current_freq = None
        self.current_type = None
        self.current_octaves = None
        self.black_level = None
        self.white_level = None

    def get_detail_mask(self, image):
        # Convert from torch tensor to numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            image = (image[0].transpose(1, 2, 0) * 255).astype(np.uint8)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Calculate edges using multiple methods
        edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
        
        edges_laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        
        # Combine edge detection methods
        detail_mask = (edges_sobel + edges_laplacian) / 2
        
        # Normalize and invert (high values = less detail)
        detail_mask = 1.0 - cv2.normalize(detail_mask, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply Gaussian blur to smooth the mask
        detail_mask = cv2.GaussianBlur(detail_mask, (0, 0), sigmaX=2)
        
        return detail_mask

    def apply_levels(self, noise):
        # Clip the noise to the specified black and white levels
        noise = np.clip(noise, self.black_level, self.white_level)
        # Renormalize to full range
        if self.white_level > self.black_level:
            noise = (noise - self.black_level) / (self.white_level - self.black_level)
        return noise

    def generate_noise_map(self, shape, detail_mask=None):
        np.random.seed(self.current_seed)
        h, w = shape[-2:]
        
        base_noise = None
        if self.current_type == "perlin":
            base_noise = self.generate_perlin_noise(h, w)
        elif self.current_type == "voronoi":
            base_noise = self.generate_voronoi_noise(h, w)
        elif self.current_type == "simplex":
            base_noise = self.generate_simplex_noise(h, w)
        else:  # white noise
            base_noise = self.generate_white_noise(h, w)

        # Normalize base noise
        base_noise = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min())
        
        if detail_mask is not None:
            # Adjust noise frequency based on detail mask
            high_freq_noise = self.generate_noise_map((1, 1, h, w))
            low_freq_noise = cv2.GaussianBlur(base_noise, (0, 0), sigmaX=3)
            
            # Combine noises based on detail mask
            base_noise = detail_mask * low_freq_noise + (1 - detail_mask) * base_noise
        
        # Apply levels adjustment
        base_noise = self.apply_levels(base_noise)
        
        # Apply scale factor
        base_noise = base_noise * self.current_scale
        
        # Convert to torch tensor
        noise_map = torch.from_numpy(base_noise).float()
        if len(shape) > 2:
            noise_map = noise_map.unsqueeze(0).unsqueeze(0)
            noise_map = noise_map.expand(shape)
        
        return noise_map

    def apply(self, model, image, noise_type, noise_scale, noise_frequency, detail_strength, 
             black_level, white_level, octaves, seed):
        model = model.clone()
        self.current_seed = seed
        self.current_scale = noise_scale
        self.current_freq = noise_frequency
        self.current_type = noise_type
        self.current_octaves = octaves
        self.black_level = black_level
        self.white_level = white_level
        
        # Generate detail mask
        detail_mask = self.get_detail_mask(image)
        detail_mask = np.power(detail_mask, detail_strength)  # Adjust strength
        
        # Store detail mask for later use
        self.detail_mask = torch.from_numpy(detail_mask).float().unsqueeze(0).unsqueeze(0)
        
        # Generate preview noise map with detail mask
        preview_shape = (1, 1, 512, 512)
        preview_noise = self.generate_noise_map(preview_shape, detail_mask)
        
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'device'):
                device = model.model.device
            elif hasattr(model, 'inner_model') and hasattr(model.inner_model, 'device'):
                device = model.inner_model.device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move tensors to appropriate device
        preview_noise = preview_noise.to(device)
        self.detail_mask = self.detail_mask.to(device)
        
        model.set_model_denoise_mask_function(self.forward)
        return (model, preview_noise, self.detail_mask)

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        
        if (self.noise_map is None or self.noise_map.shape != denoise_mask.shape):
            self.noise_map = self.generate_noise_map(denoise_mask.shape, 
                                                   self.detail_mask.cpu().numpy()[0, 0])
            self.noise_map = self.noise_map.to(device=denoise_mask.device)
        
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]
        
        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])
        
        base_threshold = (current_ts - ts_to) / (ts_from - ts_to)
        modified_threshold = base_threshold + self.noise_map * (1 - base_threshold)
        
        return (denoise_mask >= modified_threshold).to(denoise_mask.dtype)
        
NODE_CLASS_MAPPINGS = {
    "NoiseGuidedDiffusion": NoiseGuidedDiffusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseGuidedDiffusion": "Noise Guided Diffusion",
}
