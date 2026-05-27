"""
GradCAM implementation for CNN visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM
        
        Args:
            model: CNN model
            target_layer: Layer to visualize (e.g., model.resnet.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image):
        """
        Generate GradCAM heatmap
        
        Args:
            input_image: Input tensor (1, C, H, W)
        
        Returns:
            heatmap: Numpy array (H, W) with values in [0, 1]
            predicted_age: Predicted age value
        """
        self.model.eval()
        
        # Forward pass
        input_image.requires_grad = True
        output = self.model(input_image)
        
        # Backward pass
        self.model.zero_grad()
        output.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # (H, W)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Resize to input size
        cam = cv2.resize(cam, (224, 224))
        
        return cam, output.item()
    
    def remove_hooks(self):
        """Remove registered hooks"""
        self.forward_hook.remove()
        self.backward_hook.remove()


def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on grayscale image
    
    Args:
        image: Grayscale image (H, W) in [0, 1]
        heatmap: Heatmap (H, W) in [0, 1]
        alpha: Heatmap transparency
        colormap: OpenCV colormap
    
    Returns:
        RGB image with heatmap overlay
    """
    # Convert grayscale to RGB
    image_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    
    # Overlay
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay