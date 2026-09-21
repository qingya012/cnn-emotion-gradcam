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
        self.diagnostics = {}

        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradients)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradients(self, module, grad_input, grad_output):
        # Full backward hooks receive tuples. The first item is the gradient
        # with respect to this layer's output.
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None, print_diagnostics=False):
        self.model.eval()
        self.activation = None
        self.gradients = None

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output[0].argmax().item()
        elif isinstance(class_idx, torch.Tensor):
            class_idx = class_idx.item()
        class_idx = int(class_idx)

        self.model.zero_grad(set_to_none=True)
        score = output[0, class_idx]
        score.backward()

        if self.activation is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations and gradients. "
                "Check that target_layer participates in the forward pass."
            )
        if self.activation.ndim != 4 or self.gradients.ndim != 4:
            raise ValueError(
                "Grad-CAM expects activation and gradient tensors with shape "
                f"[B, C, H, W], got {tuple(self.activation.shape)} and "
                f"{tuple(self.gradients.shape)}."
            )
        if self.activation.shape != self.gradients.shape:
            raise ValueError(
                "Activation and gradient shapes must match, got "
                f"{tuple(self.activation.shape)} and {tuple(self.gradients.shape)}."
            )

        activations = self.activation[0]
        gradients = self.gradients[0]

        # Average each channel's gradients spatially, then combine the
        # corresponding feature maps into a single H x W map.
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min).item() > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        self.diagnostics = {
            "model_output_shape": tuple(output.shape),
            "activation_shape": tuple(self.activation.shape),
            "gradient_shape": tuple(self.gradients.shape),
            "raw_cam_shape": tuple(cam.shape),
            "resized_cam_shape": None,
        }

        if print_diagnostics:
            self.print_diagnostics()

        return cam.detach().cpu().numpy(), class_idx

    def overlay_heatmap(self, image, heatmap, alpha=0.4, ax=None, plot=True):
        image_2d = self._as_2d_array(image, "image")
        heatmap_2d = self._as_2d_array(heatmap, "heatmap")

        # F.interpolate with bilinear mode expects [N, C, H, W].
        heatmap_tensor = (
            torch.as_tensor(heatmap_2d, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        resized_heatmap = F.interpolate(
            heatmap_tensor,
            size=image_2d.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        self.diagnostics["resized_cam_shape"] = tuple(resized_heatmap.shape)

        if plot:
            if ax is None:
                _, ax = plt.subplots()
            ax.imshow(image_2d, cmap="gray")
            ax.imshow(resized_heatmap, cmap="jet", alpha=alpha)
            ax.axis("off")

        return resized_heatmap

    def visualize(self, image, heatmap, alpha=0.4, print_diagnostics=True):
        image_2d = self._as_2d_array(image, "image")
        heatmap_2d = self._as_2d_array(heatmap, "heatmap")
        resized_heatmap = self.overlay_heatmap(
            image_2d, heatmap_2d, alpha=alpha, plot=False
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image_2d, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(heatmap_2d, cmap="jet")
        axes[1].set_title("Raw Grad-CAM")

        axes[2].imshow(image_2d, cmap="gray")
        axes[2].imshow(resized_heatmap, cmap="jet", alpha=alpha)
        axes[2].set_title("Grad-CAM overlay")

        for axis in axes:
            axis.axis("off")

        fig.tight_layout()

        if print_diagnostics:
            self.print_diagnostics()

        return fig, axes, resized_heatmap

    def print_diagnostics(self):
        labels = (
            ("model_output_shape", "Model output shape"),
            ("activation_shape", "Saved activation shape"),
            ("gradient_shape", "Saved gradient shape"),
            ("raw_cam_shape", "Raw CAM shape"),
            ("resized_cam_shape", "Resized CAM shape"),
        )
        for key, label in labels:
            print(f"{label}: {self.diagnostics.get(key)}")

    @staticmethod
    def _as_2d_array(value, name):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value).squeeze()
        if array.ndim != 2:
            raise ValueError(
                f"{name} must represent one 2D map, got shape {array.shape}."
            )
        return array

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
