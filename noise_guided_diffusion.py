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
            "white_level": ("INT", {
                "default": 0,
                "min": 0,
                "max": 255,
                "step": 1
            }),
            "black_level": ("INT", {
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
            image = image[0]
            image = (image * 255).astype(np.uint8)

        # Convert to grayscale
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0]

        # Detect texture details using multiple methods
        edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_detail = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
        
        # Local variance for detail detection
        kernel_size = 5
        local_mean = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
        local_sqr_mean = cv2.blur(np.square(gray.astype(float)), (kernel_size, kernel_size))
        local_variance = local_sqr_mean - np.square(local_mean)
        
        # Combine detail detection methods
        detail_mask = texture_detail + local_variance
        
        # Normalize
        detail_mask = detail_mask / detail_mask.max()
        
        # Apply sensitivity
        detail_mask = np.power(detail_mask, sensitivity)
        
        # Smooth the result
        sigma = smoothing * 2
        detail_mask = cv2.GaussianBlur(detail_mask, (0, 0), sigma)
        
        # Normalize and invert (high values = less detail)
        detail_mask = 1.0 - cv2.normalize(detail_mask, None, 0, 1, cv2.NORM_MINMAX)
        
        return detail_mask
    
    def generate_perlin_noise(self, height, width, noise_size):
        # Adjust the scale based on noise_size
        scale = noise_size * 0.1
        
        # Generate multiple frequencies
        noise = np.zeros((height, width))
        octaves = 4
        persistence = 0.5
        amplitude = 1.0
        frequency = 1.0
        
        for _ in range(octaves):
            h_period = int(width * scale / frequency)
            v_period = int(height * scale / frequency)
            if h_period < 1: h_period = 1
            if v_period < 1: v_period = 1
            
            # Generate base grids
            x = np.linspace(0, h_period, width)
            y = np.linspace(0, v_period, height)
            x_idx, y_idx = np.meshgrid(x, y)
            
            # Add random phase shifts
            phase_x = np.random.rand() * 2 * np.pi
            phase_y = np.random.rand() * 2 * np.pi
            
            # Generate noise layer
            noise += amplitude * np.sin(x_idx + phase_x) * np.sin(y_idx + phase_y)
            
            frequency *= 2
            amplitude *= persistence
        
        return noise

    def generate_voronoi_noise(self, height, width, noise_size):
        # Generate fewer points for larger noise
        num_points = int(max(height, width) / (noise_size * 2))
        points = np.random.rand(num_points, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        
        # Create coordinate grids
        y, x = np.mgrid[0:height, 0:width]
        noise = np.zeros((height, width))
        
        # Calculate distances to nearest points with adaptive scaling
        scale_factor = noise_size * 0.05  # Adjust this factor to control pattern size
        for px, py in points:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            noise = np.maximum(noise, 1 / (1 + dist * scale_factor))
        
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
    
    def apply(self, model, image, noise_type, noise_scale, noise_size, 
             detail_sensitivity, smoothing, white_level, black_level, seed,
             detail_attraction=2.0, min_noise=0.1):  # Added new parameters with defaults
        # Get image dimensions
        height = image.shape[1]
        width = image.shape[2]
    
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
    
        # Normalize base noise to 0-1
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Create attraction to plain areas
        detail_influence = np.power(detail_mask, detail_attraction)
        
        # Blend noise based on detail levels
        weighted_noise = noise * detail_influence
        
        # Add minimum noise level in detailed areas
        weighted_noise = min_noise + (weighted_noise * (1 - min_noise))
        
        # Scale the noise
        weighted_noise = weighted_noise * noise_scale
        
        # Apply black/white levels at the final stage
        w_level = self.white_level / 255.0
        b_level = self.black_level / 255.0
        
        # Ensure black level is higher than white level (since we're inverting)
        if b_level <= w_level:
            b_level = w_level + 0.01
        
        # Remap the values to the black/white range
        noise_map = w_level + (weighted_noise * (b_level - w_level))
        
        # Create inverted noise map
        inverted_noise_map = 1.0 - noise_map
        
        # Convert masks to torch tensors
        detail_mask_tensor = torch.from_numpy(detail_mask).float().unsqueeze(0)
        noise_map_tensor = torch.from_numpy(inverted_noise_map).float().unsqueeze(0)
        
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
        
        # Move all tensors to the correct device
        detail_mask_tensor = detail_mask_tensor.to(device)
        noise_map_tensor = noise_map_tensor.to(device)
        
        # Store tensors (already on correct device)
        self.noise_map = noise_map_tensor
        self.detail_mask = detail_mask_tensor
        
        # Set up model
        model = model.clone()
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
