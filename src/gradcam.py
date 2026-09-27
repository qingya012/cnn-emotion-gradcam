import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def find_last_conv2d(model):
    """Return the name and module of the model's final registered Conv2d."""
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = (name, module)
    if last_conv is None:
        raise ValueError("The model does not contain a torch.nn.Conv2d layer.")
    return last_conv


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.target_layer_name = self._find_module_name(target_layer)

        self.activation = None
        self.gradients = None
        self.raw_cam = None
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
        self.model.zero_grad(set_to_none=True)
        self.activation = None
        self.gradients = None
        self.raw_cam = None

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output[0].argmax().item()
        elif isinstance(class_idx, torch.Tensor):
            class_idx = class_idx.item()
        class_idx = int(class_idx)

        if not 0 <= class_idx < output.shape[1]:
            raise ValueError(
                f"class_idx must be in [0, {output.shape[1] - 1}], got {class_idx}."
            )

        # Backpropagate the pre-softmax class score (logit), not a probability.
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
        raw_cam = F.relu((weights * activations).sum(dim=0))
        self.raw_cam = raw_cam.detach().cpu()

        cam = F.interpolate(
            raw_cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
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
            "raw_cam_shape": tuple(raw_cam.shape),
            "resized_cam_shape": tuple(cam.shape),
            "target_class": class_idx,
            "target_layer": self.target_layer_name,
        }

        self.model.zero_grad(set_to_none=True)

        if print_diagnostics:
            self.print_diagnostics()

        return cam.detach().cpu().numpy(), class_idx

    def overlay_heatmap(self, image, heatmap, alpha=0.4, ax=None, plot=True):
        image_2d = self._as_2d_array(image, "image")
        heatmap_2d = self._as_2d_array(heatmap, "heatmap")

        if heatmap_2d.shape != image_2d.shape:
            raise ValueError(
                "Heatmap and image must have the same spatial shape. "
                "Use generate() to create an input-sized Grad-CAM, got "
                f"{heatmap_2d.shape} and {image_2d.shape}."
            )

        if plot:
            if ax is None:
                _, ax = plt.subplots()
            ax.imshow(image_2d, cmap="gray")
            ax.imshow(heatmap_2d, cmap="jet", alpha=alpha, vmin=0, vmax=1)
            ax.axis("off")

        return heatmap_2d

    def visualize(
        self,
        image,
        heatmap,
        target_label=None,
        alpha=0.4,
        print_diagnostics=False,
    ):
        image_2d = self._as_2d_array(image, "image")
        heatmap_2d = self._as_2d_array(heatmap, "heatmap")
        self.overlay_heatmap(image_2d, heatmap_2d, alpha=alpha, plot=False)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image_2d, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(heatmap_2d, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title(self._gradcam_title(target_label))

        axes[2].imshow(image_2d, cmap="gray")
        axes[2].imshow(heatmap_2d, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        axes[2].set_title("Grad-CAM overlay")

        for axis in axes:
            axis.axis("off")

        fig.tight_layout()

        if print_diagnostics:
            self.print_diagnostics()

        return fig, axes

    def visualize_comparison(
        self,
        image,
        predicted_heatmap,
        true_heatmap,
        predicted_label,
        true_label,
        alpha=0.4,
    ):
        """Show predicted-class and ground-truth-class CAM overlays."""
        image_2d = self._as_2d_array(image, "image")
        predicted_cam = self._as_2d_array(predicted_heatmap, "predicted_heatmap")
        true_cam = self._as_2d_array(true_heatmap, "true_heatmap")
        self.overlay_heatmap(image_2d, predicted_cam, plot=False)
        self.overlay_heatmap(image_2d, true_cam, plot=False)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_2d, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(image_2d, cmap="gray")
        axes[1].imshow(predicted_cam, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        axes[1].set_title(f"Grad-CAM: predicted {predicted_label}")

        axes[2].imshow(image_2d, cmap="gray")
        axes[2].imshow(true_cam, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        axes[2].set_title(f"Grad-CAM: true {true_label}")

        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        return fig, axes

    def print_diagnostics(self):
        labels = (
            ("model_output_shape", "Model output shape"),
            ("activation_shape", "Saved activation shape"),
            ("gradient_shape", "Saved gradient shape"),
            ("raw_cam_shape", "Raw CAM shape"),
            ("resized_cam_shape", "Resized CAM shape"),
            ("target_class", "Target class index"),
            ("target_layer", "Selected target layer"),
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

    def _find_module_name(self, target_layer):
        for name, module in self.model.named_modules():
            if module is target_layer:
                return name
        return f"<explicit {target_layer.__class__.__name__}>"

    @staticmethod
    def _gradcam_title(target_label):
        if target_label is None:
            return "Grad-CAM heatmap"
        return f"Grad-CAM: {target_label}"

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
