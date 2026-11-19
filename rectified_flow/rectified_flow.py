"""
Rectified Flow in PyTorch with diagnostic visualizations.

Implements:
  - 2D eight-mode Gaussian mixture target
  - Conditional flow-matching / Rectified Flow objective
  - Time-conditioned MLP velocity field
  - Euler ODE sampling
  - Optional reflow training
  - Diagnostic plots:
      1. Target vs generated distributions
      2. Sample trajectories colored by time
      3. Velocity fields at multiple times
      4. Velocity magnitude heatmaps
      5. Training loss curves
      6. Trajectory straightness distribution
      7. Sampling-step comparison
      8. Interpolation training examples

Run:
    pip install torch numpy matplotlib
    python rectified_flow.py

For a quick smoke test:
    python rectified_flow.py --iterations 3000 --reflow-iterations 2000
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GaussianMixture2D:
    """Eight isotropic Gaussian modes arranged on a circle."""

    def __init__(self, radius: float = 2.0, std: float = 0.15, modes: int = 8):
        self.radius = radius
        self.std = std
        self.modes = modes

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        angles = torch.arange(self.modes, device=device) * (
            2.0 * math.pi / self.modes
        )
        centers = self.radius * torch.stack(
            [torch.cos(angles), torch.sin(angles)], dim=1
        )
        indices = torch.randint(self.modes, (batch_size,), device=device)
        return centers[indices] + self.std * torch.randn(
            batch_size, 2, device=device
        )


def sample_source(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, 2, device=device)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        if embedding_dim % 2:
            raise ValueError("embedding_dim must be even")
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), embedding_dim // 2)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * t * self.frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class VelocityNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 256, time_dim: int = 32):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.network = nn.Sequential(
            nn.Linear(2 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([x, self.time_embedding(t)], dim=-1))


def rectified_flow_loss(
    model: nn.Module, x0: torch.Tensor, x1: torch.Tensor
) -> torch.Tensor:
    """MSE between predicted velocity and paired endpoint displacement."""
    batch_size = x0.shape[0]
    t = torch.rand(batch_size, 1, device=x0.device)
    xt = (1.0 - t) * x0 + t * x1
    target_velocity = x1 - x0
    predicted_velocity = model(xt, t)
    return F.mse_loss(predicted_velocity, target_velocity)


def train_flow(
    model: nn.Module,
    target: GaussianMixture2D,
    iterations: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    endpoint_pairs=None,
    log_every: int = 500,
):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    losses = []

    for iteration in range(1, iterations + 1):
        if endpoint_pairs is None:
            x0 = sample_source(batch_size, device)
            x1 = target.sample(batch_size, device)
        else:
            pair_x0, pair_x1 = endpoint_pairs
            indices = torch.randint(
                len(pair_x0), (batch_size,), device=device
            )
            x0 = pair_x0[indices]
            x1 = pair_x1[indices]

        loss = rectified_flow_loss(model, x0, x1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        if iteration % log_every == 0 or iteration == 1:
            print(
                f"Iteration {iteration:>6}/{iterations} | "
                f"loss = {losses[-1]:.6f}"
            )

    return losses


@torch.no_grad()
def sample_flow(
    model: nn.Module,
    num_samples: int,
    steps: int,
    device: torch.device,
    initial_noise: torch.Tensor = None,
    return_trajectory: bool = False,
):
    """Integrate dx/dt = v_theta(x,t) with explicit Euler."""
    if steps < 1:
        raise ValueError("steps must be at least 1")

    model.eval()
    x = (
        sample_source(num_samples, device)
        if initial_noise is None
        else initial_noise.to(device).clone()
    )
    dt = 1.0 / steps
    trajectory = [x.clone()] if return_trajectory else None

    for step in range(steps):
        t = torch.full((len(x), 1), step / steps, device=device)
        x = x + dt * model(x, t)
        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, torch.stack(trajectory, dim=0)
    return x


@torch.no_grad()
def collect_reflow_pairs(
    model: nn.Module, num_pairs: int, steps: int, device: torch.device
):
    """Pair fresh source points with their endpoints under the learned ODE."""
    x0 = sample_source(num_pairs, device)
    x1 = sample_flow(
        model, num_pairs, steps, device, initial_noise=x0
    )
    return x0, x1


def trajectory_straightness(trajectory: torch.Tensor, eps: float = 1e-8):
    """
    Per-path chord length / traveled path length.
    Values approach 1 for straight paths; lower values indicate curvature.
    """
    increments = trajectory[1:] - trajectory[:-1]
    path_length = increments.norm(dim=-1).sum(dim=0)
    chord_length = (trajectory[-1] - trajectory[0]).norm(dim=-1)
    return chord_length / (path_length + eps)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def plot_training_losses(initial_losses, reflow_losses, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(initial_losses, label="Initial Rectified Flow", alpha=0.85)
    if reflow_losses:
        ax.plot(
            np.arange(len(reflow_losses)),
            reflow_losses,
            label="Reflow",
            alpha=0.85,
        )
    ax.set(
        title="Training objective",
        xlabel="Optimization iteration",
        ylabel="Velocity MSE",
    )
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_training_losses.png", dpi=160)
    plt.close(fig)


def plot_distributions(target_samples, initial_samples, reflow_samples, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [
        (target_samples, "Target distribution"),
        (initial_samples, "Initial Rectified Flow"),
        (reflow_samples, "After reflow"),
    ]
    for ax, (samples, title) in zip(axes, panels):
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.45)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.grid(alpha=0.2)
    fig.suptitle("Generated samples compared with the target")
    fig.tight_layout()
    fig.savefig(output_dir / "02_distributions.png", dpi=160)
    plt.close(fig)


def plot_trajectories(trajectory, output_dir, max_paths=100):
    traj = to_numpy(trajectory)
    count = min(max_paths, traj.shape[1])
    indices = np.random.choice(traj.shape[1], count, replace=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("viridis")
    times = np.linspace(0.0, 1.0, traj.shape[0])

    for idx in indices:
        path = traj[:, idx, :]
        ax.plot(path[:, 0], path[:, 1], color="gray", alpha=0.25, linewidth=0.8)
        ax.scatter(
            path[:, 0], path[:, 1], c=times, cmap=cmap,
            s=8, alpha=0.75, vmin=0, vmax=1
        )

    ax.scatter(
        traj[0, indices, 0], traj[0, indices, 1],
        marker="x", s=24, label="t = 0"
    )
    ax.scatter(
        traj[-1, indices, 0], traj[-1, indices, 1],
        marker="o", facecolors="none", s=25, label="t = 1"
    )
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(scalar, ax=ax, label="Time")
    ax.set(
        title="Learned ODE trajectories",
        xlabel="x₁",
        ylabel="x₂",
        xlim=(-3.5, 3.5),
        ylim=(-3.5, 3.5),
    )
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "03_trajectories.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def plot_velocity_fields(model, output_dir, device, grid_lim=3.5, grid_size=25):
    model.eval()
    axis = torch.linspace(-grid_lim, grid_lim, grid_size, device=device)
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    times_to_plot = [0.0, 0.25, 0.5, 0.75]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, time_value in zip(axes.flat, times_to_plot):
        t = torch.full((len(points), 1), time_value, device=device)
        velocity = model(points, t)
        speed = velocity.norm(dim=1)

        p = to_numpy(points)
        v = to_numpy(velocity)
        s = to_numpy(speed)

        ax.quiver(
            p[:, 0], p[:, 1], v[:, 0], v[:, 1],
            s, cmap="plasma", angles="xy"
        )
        ax.set_title(f"Velocity field at t = {time_value:.2f}")
        ax.set_aspect("equal")
        ax.set_xlim(-grid_lim, grid_lim)
        ax.set_ylim(-grid_lim, grid_lim)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "04_velocity_fields.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def plot_velocity_magnitude(model, output_dir, device, grid_lim=3.5, grid_size=100):
    model.eval()
    axis = torch.linspace(-grid_lim, grid_lim, grid_size, device=device)
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, time_value in zip(axes, [0.0, 0.5, 1.0]):
        t = torch.full((len(points), 1), time_value, device=device)
        speed = model(points, t).norm(dim=1).reshape(grid_size, grid_size)
        image = ax.imshow(
            to_numpy(speed),
            extent=[-grid_lim, grid_lim, -grid_lim, grid_lim],
            origin="lower",
            cmap="magma",
            aspect="equal",
        )
        ax.set_title(f"Speed ||v(x,t)|| at t={time_value:.1f}")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "05_velocity_magnitude.png", dpi=160)
    plt.close(fig)


def plot_straightness(initial_trajectory, reflow_trajectory, output_dir):
    initial_scores = to_numpy(trajectory_straightness(initial_trajectory))
    reflow_scores = to_numpy(trajectory_straightness(reflow_trajectory))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0.0, 1.01, 35)
    ax.hist(initial_scores, bins=bins, alpha=0.55, label="Initial RF", density=True)
    ax.hist(reflow_scores, bins=bins, alpha=0.55, label="Reflow", density=True)
    ax.set(
        title="Trajectory straightness",
        xlabel="Chord length / path length (closer to 1 is straighter)",
        ylabel="Density",
        xlim=(0, 1.02),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "06_trajectory_straightness.png", dpi=160)
    plt.close(fig)

    print(
        "Mean trajectory straightness | "
        f"initial={initial_scores.mean():.4f}, "
        f"reflow={reflow_scores.mean():.4f}"
    )


def plot_sampling_steps(
    initial_model, reflow_model, output_dir, device, step_counts
):
    # Reuse identical initial noise across step counts for a fair visual comparison.
    fixed_noise = sample_source(1500, device)
    fig, axes = plt.subplots(2, len(step_counts), figsize=(4 * len(step_counts), 8))

    for row, (model, label) in enumerate(
        [(initial_model, "Initial RF"), (reflow_model, "Reflow")]
    ):
        for col, steps in enumerate(step_counts):
            samples = sample_flow(
                model, len(fixed_noise), steps, device,
                initial_noise=fixed_noise
            )
            samples = to_numpy(samples)
            ax = axes[row, col]
            ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.4)
            ax.set_title(f"{label}\n{steps} Euler step(s)")
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)

    fig.suptitle("Sampling quality as a function of ODE function evaluations")
    fig.tight_layout()
    fig.savefig(output_dir / "07_sampling_steps.png", dpi=160)
    plt.close(fig)


def plot_training_interpolations(target, output_dir, device):
    """Illustrate paired endpoints and their linear interpolation paths."""
    x0 = sample_source(12, device)
    x1 = target.sample(12, device)
    times = torch.linspace(0, 1, 6, device=device)

    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(len(x0)):
        path = (
            (1 - times[:, None]) * x0[i][None, :]
            + times[:, None] * x1[i][None, :]
        )
        path_np = to_numpy(path)
        ax.plot(path_np[:, 0], path_np[:, 1], alpha=0.6, linewidth=1)
        ax.scatter(path_np[:, 0], path_np[:, 1], c=to_numpy(times), cmap="viridis", s=22)

    ax.set_title("Training construction: straight interpolation between paired samples")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "08_training_interpolations.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--reflow-iterations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--reflow-pairs", type=int, default=30000)
    parser.add_argument("--reflow-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./figures")
    parser.add_argument("--no-reflow", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Saving plots to: {output_dir.resolve()}")

    target = GaussianMixture2D()
    plot_training_interpolations(target, output_dir, device)

    initial_model = VelocityNetwork().to(device)
    print("\nTraining initial Rectified Flow...")
    initial_losses = train_flow(
        initial_model,
        target,
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )

    initial_samples, initial_trajectory = sample_flow(
        initial_model,
        num_samples=2000,
        steps=args.eval_steps,
        device=device,
        return_trajectory=True,
    )

    reflow_losses = []
    reflow_model = initial_model
    reflow_samples = initial_samples
    reflow_trajectory = initial_trajectory

    if not args.no_reflow:
        print("\nCollecting endpoint pairs for reflow...")
        endpoint_pairs = collect_reflow_pairs(
            initial_model,
            num_pairs=args.reflow_pairs,
            steps=args.reflow_steps,
            device=device,
        )

        reflow_model = VelocityNetwork().to(device)
        print("\nTraining reflow model...")
        reflow_losses = train_flow(
            reflow_model,
            target,
            iterations=args.reflow_iterations,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            endpoint_pairs=endpoint_pairs,
        )

        reflow_samples, reflow_trajectory = sample_flow(
            reflow_model,
            num_samples=2000,
            steps=args.eval_steps,
            device=device,
            return_trajectory=True,
        )

    target_samples = target.sample(2000, device)

    plot_training_losses(initial_losses, reflow_losses, output_dir)
    plot_distributions(
        to_numpy(target_samples),
        to_numpy(initial_samples),
        to_numpy(reflow_samples),
        output_dir,
    )
    plot_trajectories(initial_trajectory, output_dir)
    plot_velocity_fields(initial_model, output_dir, device)
    plot_velocity_magnitude(initial_model, output_dir, device)
    plot_straightness(initial_trajectory, reflow_trajectory, output_dir)
    plot_sampling_steps(
        initial_model,
        reflow_model,
        output_dir,
        device,
        step_counts=[1, 2, 5, 10, 50],
    )

    print("\nFinished. Diagnostic plots:")
    for path in sorted(output_dir.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
