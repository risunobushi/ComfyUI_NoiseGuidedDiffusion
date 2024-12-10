import torch
import numpy as np
import cv2

class NoiseGuidedDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "image": ("IMAGE",),
            "noise_type": (["perlin", "voronoi", "simplex"],),
            "noise_scale": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1
            }),
            "noise_size": ("FLOAT", {
                "default": 1.0,
                "min": 0.1,
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
            "black_level": ("INT", {
                "default": 0,
                "min": 0,
                "max": 255,
                "step": 1
            }),
            "white_level": ("INT", {
                "default": 255,
                "min": 0,
                "max": 255,
                "step": 1
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
        self.black_level = None
        self.white_level = None

    def detect_details(self, image, sensitivity, smoothing):
        # Convert from torch tensor to numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            # Get first image from batch
            image = image[0]
            # Image is already in HWC format, just need to scale
            image = (image * 255).astype(np.uint8)

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

    def generate_perlin_noise(self, height, width, noise_size):
        # Scale dimensions by noise size
        scaled_h = int(height / noise_size)
        scaled_w = int(width / noise_size)
        
        # Generate base noise at smaller scale
        y = np.linspace(0, scaled_h, scaled_h)
        x = np.linspace(0, scaled_w, scaled_w)
        x_idx, y_idx = np.meshgrid(x, y)
        
        # Add octaves for more natural look
        noise = np.zeros((scaled_h, scaled_w))
        octaves = 4
        persistence = 0.5
        amplitude = 1.0
        
        for _ in range(octaves):
            phase_x = np.random.rand() * 100
            phase_y = np.random.rand() * 100
            noise += amplitude * np.sin(2 * np.pi * (x_idx + phase_x) / 10) * \
                    np.sin(2 * np.pi * (y_idx + phase_y) / 10)
            amplitude *= persistence
        
        # Resize back to original dimensions
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        return noise

    def generate_voronoi_noise(self, height, width, noise_size):
        # Generate fewer points for larger noise
        num_points = int(max(height, width) / noise_size)
        points = np.random.rand(num_points, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        
        # Create coordinate grids
        y, x = np.mgrid[0:height, 0:width]
        noise = np.zeros((height, width))
        
        # Calculate distances to nearest points
        for px, py in points:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            noise = np.maximum(noise, 1 / (1 + dist * (0.1 / noise_size)))
        
        return noise

    def generate_simplex_noise(self, height, width, noise_size):
        # Scale dimensions by noise size
        freq = 1.0 / noise_size
        y = np.linspace(0, height * freq, height)
        x = np.linspace(0, width * freq, width)
        x_idx, y_idx = np.meshgrid(x, y)
        
        noise = np.zeros((height, width))
        octaves = 4
        persistence = 0.5
        amplitude = 1.0
        
        for i in range(octaves):
            freq = (2 ** i)
            current_noise = np.zeros((height, width))
            
            # Add multiple rotated sine waves
            angles = np.linspace(0, np.pi, 3)
            for angle in angles:
                rotated_x = x_idx * np.cos(angle) - y_idx * np.sin(angle)
                rotated_y = x_idx * np.sin(angle) + y_idx * np.cos(angle)
                current_noise += np.sin(2 * np.pi * freq * rotated_x) * \
                               np.sin(2 * np.pi * freq * rotated_y)
            
            noise += amplitude * current_noise
            amplitude *= persistence
        
        return noise

        def apply_levels(self, noise):
        # Convert levels from 0-255 to 0-1 range
        black = self.black_level / 255.0
        white = self.white_level / 255.0
        
        # Ensure white level is higher than black level
        if white <= black:
            white = black + 0.01
        
        # Scale the noise to fit between black and white levels
        noise = np.clip(noise, 0, 1)  # Ensure noise is in 0-1 range
        noise = black + noise * (white - black)
        return noise

    def apply(self, model, image, noise_type, noise_scale, noise_size, 
             detail_sensitivity, smoothing, black_level, white_level, seed):
        # Store levels for use in noise generation
        self.black_level = black_level
        self.white_level = white_level
        
        # Get image dimensions (BHWC format)
        height, width = image.shape[1], image.shape[2]
        
        # Generate detail mask
        detail_mask = self.detect_details(image, detail_sensitivity, smoothing)
        
        # Generate base noise based on selected type
        np.random.seed(seed)
        if noise_type == "perlin":
            noise = self.generate_perlin_noise(height, width, noise_size)
        elif noise_type == "voronoi":
            noise = self.generate_voronoi_noise(height, width, noise_size)
        else:  # simplex
            noise = self.generate_simplex_noise(height, width, noise_size)
        
        # Normalize noise to 0-1 range
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Apply levels adjustment
        noise = self.apply_levels(noise)
        
        # Apply detail mask and scale
        noise_map = noise * detail_mask * noise_scale
        
        # Convert masks to torch tensors with correct dimensions
        detail_mask_tensor = torch.from_numpy(detail_mask).float().unsqueeze(0)  # Add batch dimension
        noise_map_tensor = torch.from_numpy(noise_map).float().unsqueeze(0)      # Add batch dimension
        
        # Store for later use
        self.noise_map = noise_map_tensor
        self.detail_mask = detail_mask_tensor
        
        # Set up model
        model = model.clone()
        
        # Determine device
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
        self.noise_map = self.noise_map.to(device)
        self.detail_mask = self.detail_mask.to(device)
        noise_map_tensor = noise_map_tensor.to(device)
        detail_mask_tensor = detail_mask_tensor.to(device)
        
        model.set_model_denoise_mask_function(self.forward)
        
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
