import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid
from ..event import (
    Event,
    Start,
    Stop,
    EnvStep,
    EnvReset,
    ModuleForwardStart,
    ModuleForwardEnd,
)


class TensorboardWriter:
    def __init__(self) -> None:
        self.writer = SummaryWriter()
        self.step_counter: dict[str, int] = {}

    def __call__(self, event: Event) -> None:
        """
        Callback for the event bus. Takes an event and adds it to the tensorboard.

        Parameters
        ----------
        event : Event
            The event to add to the tensorboard
        """
        if isinstance(event, Start):
            self.add_start(event)
        elif isinstance(event, Stop):
            self.add_stop(event)
        elif isinstance(event, EnvStep):
            self.add_env_step(event)
        elif isinstance(event, EnvReset):
            self.add_env_reset(event)
        elif isinstance(event, ModuleForwardStart):
            self.add_module_forward_start(event)
        elif isinstance(event, ModuleForwardEnd):
            self.add_module_forward_end(event)

    def add_env_step(self, event: EnvStep) -> None:
        # Add scalar for reward
        self.writer.add_scalar("EnvStep/reward", event.reward, global_step=None)

        # Add images for observation and next_observation
        if event.observation is not None:
            self.writer.add_image(
                "EnvStep/observation", event.observation.squeeze(0), dataformats="CHW"
            )

        if event.next_observation is not None:
            self.writer.add_image(
                "EnvStep/next_observation",
                event.next_observation.squeeze(0),
                dataformats="CHW",
            )

        # Add histogram for action
        if event.action is not None:
            self.writer.add_histogram("EnvStep/action", event.action, global_step=None)

    def add_env_reset(self, event: EnvReset) -> None:
        # Add image for observation
        if event.observation is not None:
            self.writer.add_image(
                "EnvReset/observation", event.observation.squeeze(0), dataformats="CHW"
            )

    def add_module_forward_start(self, event: ModuleForwardStart) -> None:
        """Handle module forward start events by logging inputs to TensorBoard."""
        module_name = event.name
        if module_name not in self.step_counter:
            self.step_counter[module_name] = 0

        step = self.step_counter[module_name]

        for key, value in event.inputs.items():
            if isinstance(value, torch.Tensor):
                # Log tensor statistics
                self._log_tensor_stats(f"{module_name}/input/{key}", value, step)

                # If tensor is 2D-4D, try to visualize it as an image
                if 2 <= len(value.shape) <= 4:
                    self._try_log_as_image(
                        f"{module_name}/input_viz/{key}", value, step
                    )

    def add_module_forward_end(self, event: ModuleForwardEnd) -> None:
        """Handle module forward end events by logging outputs to TensorBoard."""
        module_name = event.name
        if module_name not in self.step_counter:
            self.step_counter[module_name] = 0

        step = self.step_counter[module_name]

        for key, value in event.outputs.items():
            if isinstance(value, torch.Tensor):
                # Log tensor statistics
                self._log_tensor_stats(f"{module_name}/output/{key}", value, step)

                # If tensor is 2D-4D, try to visualize it as an image
                if 2 <= len(value.shape) <= 4:
                    self._try_log_as_image(
                        f"{module_name}/output_viz/{key}", value, step
                    )

        # Increment step counter for this module
        self.step_counter[module_name] += 1

    def add_start(self, event: Start) -> None:
        # Start event only has timestamp which is handled by tensorboard automatically
        self.writer.add_text("Start/event", "Simulation started", global_step=None)

    def add_stop(self, event: Stop) -> None:
        # Stop event only has timestamp which is handled by tensorboard automatically
        self.writer.add_text("Stop/event", "Simulation stopped", global_step=None)

    def close(self) -> None:
        self.writer.close()

    def _log_tensor_stats(self, name: str, tensor: torch.Tensor, step: int) -> None:
        """Log tensor statistics to TensorBoard."""
        if not tensor.numel():
            return  # Skip empty tensors

        # Ensure tensor is detached and on CPU
        tensor = tensor.detach().cpu().squeeze()

        # Basic statistics
        self.writer.add_histogram(f"{name}/hist", tensor, step)
        self.writer.add_scalar(f"{name}/mean", tensor.float().mean(), step)
        self.writer.add_scalar(f"{name}/std", tensor.float().std(), step)
        self.writer.add_scalar(f"{name}/min", tensor.float().min(), step)
        self.writer.add_scalar(f"{name}/max", tensor.float().max(), step)

    def _try_log_as_image(self, name: str, tensor: torch.Tensor, step: int) -> None:
        """
        Try to log a tensor as an image if possible.
        Handles different tensor shapes appropriately.
        """
        tensor = tensor.detach().cpu().squeeze()

        # Handle different shapes
        if len(tensor.shape) == 2:  # Single grayscale image
            self.writer.add_image(name, tensor.unsqueeze(0), step, dataformats="CHW")

        elif len(tensor.shape) == 3:
            if tensor.shape[0] <= 3:  # Assume CHW format (channels, height, width)
                self.writer.add_image(name, tensor, step, dataformats="CHW")
            else:  # Assume batch of grayscale images
                grid = self._make_grid(tensor.unsqueeze(1))
                self.writer.add_image(f"{name}/batch", grid, step, dataformats="CHW")

        elif len(tensor.shape) == 4:  # Batch of images
            if tensor.shape[1] <= 3:  # Channels in dimension 1 (BCHW format)
                grid = self._make_grid(tensor)
                self.writer.add_image(f"{name}/batch", grid, step, dataformats="CHW")

    def _make_grid(self, tensor: torch.Tensor, max_images: int = 16) -> torch.Tensor:
        """Create a grid of images for visualization."""
        # Limit number of images to avoid large grids
        tensor = tensor[:max_images]
        # Normalize for better visualization
        tensor = self._normalize_for_visualization(tensor)
        return make_grid(tensor)

    def _normalize_for_visualization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor values to [0, 1] range for better visualization."""
        tensor = tensor.float()  # Convert to float
        if tensor.numel() > 0:
            min_val = tensor.min()
            max_val = tensor.max()
            if min_val != max_val:  # Avoid division by zero
                tensor = (tensor - min_val) / (max_val - min_val)
        return tensor
