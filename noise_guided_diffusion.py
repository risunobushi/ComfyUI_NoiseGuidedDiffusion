import torch
import numpy as np
import cv2

class NoiseGuidedDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "image": ("IMAGE",),
            "noise_scale": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1
            }),
            "detail_sensitivity": ("FLOAT", {
                "default": 1.0,
                "min": 0.1,
                "max": 5.0,
                "step": 0.1
            }),
            "smoothing": ("FLOAT", {
                "default": 1.0,
                "min": 0.1,
                "max": 5.0,
                "step": 0.1
            }),
            "seed": ("INT", {
                "default": 0,
                "min": 0,
                "max": 2**32 - 1
            })
        }}
    
    RETURN_TYPES = ("MODEL", "MASK", "MASK",)
    RETURN_NAMES = ("model", "noise_map", "detail_mask",)
    FUNCTION = "apply"
    CATEGORY = "sampling"
    
    def __init__(self):
        self.noise_map = None
        self.detail_mask = None

    def detect_details(self, image, sensitivity, smoothing):
        # Convert from torch tensor to numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            image = (image[0] * 255).astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))

        # Convert to grayscale
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0]

        # Detect edges
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.absolute(edges)
        
        # Normalize edge detection
        edges = edges / edges.max()
        
        # Apply sensitivity
        edges = np.power(edges, sensitivity)
        
        # Smooth the result
        sigma = smoothing * 2
        detail_mask = cv2.GaussianBlur(edges, (0, 0), sigma)
        
        # Normalize and invert (high values = less detail)
        detail_mask = 1.0 - cv2.normalize(detail_mask, None, 0, 1, cv2.NORM_MINMAX)
        
        return detail_mask

    def generate_noise(self, shape, seed):
        np.random.seed(seed)
        return np.random.rand(*shape)

    def apply(self, model, image, noise_scale, detail_sensitivity, smoothing, seed):
        # Get image dimensions
        height, width = image.shape[2], image.shape[3]
        
        # Generate detail mask
        detail_mask = self.detect_details(image, detail_sensitivity, smoothing)
        
        # Generate base noise
        noise = self.generate_noise((height, width), seed)
        
        # Apply detail mask to noise
        noise_map = noise * detail_mask
        
        # Scale noise
        noise_map = noise_map * noise_scale
        
        # Convert masks to torch tensors
        detail_mask_tensor = torch.from_numpy(detail_mask).float().unsqueeze(0).unsqueeze(0)
        noise_map_tensor = torch.from_numpy(noise_map).float().unsqueeze(0).unsqueeze(0)
        
        # Store for later use
        self.noise_map = noise_map_tensor
        self.detail_mask = detail_mask_tensor
        
        # Set up model
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        
        # Move to appropriate device
        device = next(model.parameters()).device
        self.noise_map = self.noise_map.to(device)
        self.detail_mask = self.detail_mask.to(device)
        
        return (model, noise_map_tensor, detail_mask_tensor)

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        
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
