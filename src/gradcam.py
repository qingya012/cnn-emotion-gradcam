import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activation = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradients)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output.squeeze(), dim=0)

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activation[0]

        weights = gradients.mean(dim=[1, 2])

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam.detach().cpu().numpy(), class_idx

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def overleaf_heatmap(image, heatmap, alpha=0.4):
        heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).float()
        heatmap.resized = F.interpolate(
            heatmap_tensor, 
            size=image.shape, 
            mode="bilinear", 
            align_corners=False
        ).squeeze().numpy()

        plt.imshow(image, cmap="gray")
        plt.imshow(heatmap.resized, cmap="jet", alpha=alpha)
        plt.axis("off")
