import torch
import torch.optim as optim
from torch.optim.optimizer import required
import numpy as np


class LARS(optim.Optimizer):
    """LARS optimizer implementation."""

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False, eta=1e-3, eps=1e-8, clip_lr=False, exclude_bias_n_norm=False):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eta=eta,
            eps=eps,
            clip_lr=clip_lr,
            exclude_bias_n_norm=exclude_bias_n_norm,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if p.ndim != 1 or not group["exclude_bias_n_norm"]:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (
                            g_norm + p_norm * weight_decay + group["eps"]
                        )
                        lars_lr *= group["eta"]

                        # clip lr
                        if group["clip_lr"]:
                            lars_lr = min(lars_lr / group["lr"], 1)

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss


class WarmupCosineScheduler:

    """Warmup + cosine learning rate scheduler"""

    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int, start_lr: float, base_lr: float, final_lr: float):
        
        if total_steps <= 0:
            raise ValueError(
                f"total_steps must be positive, received {total_steps}"
            )

        if not 0 <= warmup_steps < total_steps:
            raise ValueError(
                "warmup_steps must satisfy "
                f"0 <= warmup_steps < total_steps, received "
                f"{warmup_steps} and {total_steps}"
            )

        if start_lr < 0 or base_lr < 0 or final_lr < 0:
            raise ValueError("Learning rates must be non-negative")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_lr = start_lr
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.current_step = 0

        self._set_lr(self._lr_at_step(0))

    def _lr_at_step(self, step: int) -> float:
        
        """Compute the learning rate at a given step."""

        # Warmup phase
        if self.warmup_steps > 0 and step < self.warmup_steps:
            warmup_progress = step / self.warmup_steps

            return self.start_lr + warmup_progress * (
                self.base_lr - self.start_lr
            )

        cosine_steps = self.total_steps - self.warmup_steps
        cosine_progress = (
            step - self.warmup_steps
        ) / cosine_steps

        cosine_progress = min(max(cosine_progress, 0.0), 1.0)

        return self.final_lr + 0.5 * (
            self.base_lr - self.final_lr
        ) * (
            1.0 + np.cos(np.pi * cosine_progress)
        )

    def _set_lr(self, learning_rate: float) -> None:

        """Set the learning rate for all parameter groups in the optimizer."""

        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def step(self) -> float:

        """Update the learning rate based on the current step and return it."""

        self.current_step = min(
            self.current_step + 1,
            self.total_steps,
        )

        learning_rate = self._lr_at_step(self.current_step)
        self._set_lr(learning_rate)

        return learning_rate

    def state_dict(self) -> dict[str, object]:

        """Return the state of the scheduler as a dict."""

        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "start_lr": self.start_lr,
            "base_lr": self.base_lr,
            "final_lr": self.final_lr,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state: dict[str, object]):

        """ Load the scheduler state from a dict. Validates the state to ensure it matches the expected configuration """

        required_keys = {
            "warmup_steps",
            "total_steps",
            "start_lr",
            "base_lr",
            "final_lr",
            "current_step",
        }

        missing_keys = required_keys.difference(state)

        if missing_keys:
            raise ValueError(
                "Scheduler state is missing keys: "
                + ", ".join(sorted(missing_keys))
            )

        expected_values = {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "start_lr": self.start_lr,
            "base_lr": self.base_lr,
            "final_lr": self.final_lr,
        }

        for key, expected in expected_values.items():
            received = state[key]

            if received != expected:
                raise ValueError(
                    f"Scheduler mismatch for {key}: "
                    f"expected {expected}, received {received}"
                )

        current_step = int(state["current_step"])

        if not 0 <= current_step <= self.total_steps:
            raise ValueError(
                f"Invalid scheduler current_step: {current_step}"
            )

        self.current_step = current_step

        learning_rate = self._lr_at_step(self.current_step)
        self._set_lr(learning_rate)