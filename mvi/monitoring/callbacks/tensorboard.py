from torch.utils.tensorboard.writer import SummaryWriter

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
        # Add module name
        self.writer.add_text("ModuleForwardStart/name", event.name, global_step=None)

        # Add histogram for inputs
        if event.inputs is not None:
            self.writer.add_histogram(
                f"ModuleForwardStart/{event.name}/inputs",
                event.inputs,
                global_step=None,
            )

    def add_module_forward_end(self, event: ModuleForwardEnd) -> None:
        # Add module name
        self.writer.add_text("ModuleForwardEnd/name", event.name, global_step=None)

        # Add histogram for outputs
        if event.outputs is not None:
            self.writer.add_histogram(
                f"ModuleForwardEnd/{event.name}/outputs",
                event.outputs,
                global_step=None,
            )

    def add_start(self, event: Start) -> None:
        # Start event only has timestamp which is handled by tensorboard automatically
        self.writer.add_text("Start/event", "Simulation started", global_step=None)

    def add_stop(self, event: Stop) -> None:
        # Stop event only has timestamp which is handled by tensorboard automatically
        self.writer.add_text("Stop/event", "Simulation stopped", global_step=None)

    def close(self) -> None:
        self.writer.close()
