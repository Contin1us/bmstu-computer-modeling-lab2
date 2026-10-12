"""Замер точности: ошибка Барнса-Хата в зависимости от θ."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import Particle, compute_forces_direct, uniform_cluster
from barnes_hut import compute_forces_barnes_hut


def compute_errors(
    particles_ref: list[Particle], particles_bh: list[Particle]
) -> tuple[float, float]:
    """Считает среднюю и максимальную абсолютные ошибки по силам."""
    errors = []
    for pr, pb in zip(particles_ref, particles_bh):
        ex = abs(pr.ax - pb.ax)
        ey = abs(pr.ay - pb.ay)
        err = (ex**2 + ey**2) ** 0.5
        errors.append(err)

    mean_err = sum(errors) / len(errors)
    max_err = max(errors)
    return mean_err, max_err


def run_accuracy_experiment(
    n: int = 1000, thetas: list[float] = None, seed: int = 42
) -> tuple[list[float], list[float], list[float], float]:
    """Замеряет ошибку для разных θ.

    Returns:
        (thetas, mean_errors, max_errors, mean_force_magnitude)
    """
    if thetas is None:
        thetas = [0.1, 0.3, 0.5, 0.7, 1.0]

    # Эталон: прямой метод
    particles_ref = uniform_cluster(n=n, radius=1.0, total_mass=1.0, seed=seed)
    compute_forces_direct(particles_ref)

    # Средний модуль силы (для относительной ошибки)
    magnitudes = [(p.ax**2 + p.ay**2) ** 0.5 for p in particles_ref]
    mean_magnitude = sum(magnitudes) / len(magnitudes)

    mean_errors = []
    max_errors = []

    for theta in thetas:
        # Те же частицы, но BH
        particles_bh = uniform_cluster(n=n, radius=1.0, total_mass=1.0, seed=seed)
        compute_forces_barnes_hut(particles_bh, theta=theta)

        mean_err, max_err = compute_errors(particles_ref, particles_bh)
        mean_errors.append(mean_err)
        max_errors.append(max_err)

        rel_err = mean_err / mean_magnitude * 100
        print(
            f"θ = {theta:.1f}: mean_err = {mean_err:.6f}, max_err = {max_err:.6f}, rel = {rel_err:.4f}%"
        )

    return thetas, mean_errors, max_errors, mean_magnitude


def plot_accuracy(
    thetas: list[float],
    mean_errors: list[float],
    max_errors: list[float],
    save_path: str = "results/accuracy.png",
) -> None:
    """Рисует график ошибка vs θ."""
    plt.figure(figsize=(8, 6))
    plt.loglog(thetas, mean_errors, "o-", label="Средняя ошибка", lw=2)
    plt.loglog(thetas, max_errors, "s-", label="Максимальная ошибка", lw=2)

    plt.xlabel("Параметр θ")
    plt.ylabel("Абсолютная ошибка силы")
    plt.title("Точность Барнса-Хата vs θ")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён в {save_path}")
    plt.show()
    plt.close()


def save_csv(
    thetas,
    mean_errors,
    max_errors,
    mean_magnitude,
    save_path: str = "results/accuracy.csv",
) -> None:
    """Сохраняет данные в CSV."""
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta", "mean_error", "max_error", "relative_error_percent"])
        for th, me, mx in zip(thetas, mean_errors, max_errors):
            rel = me / mean_magnitude * 100
            writer.writerow([th, f"{me:.8f}", f"{mx:.8f}", f"{rel:.4f}"])
    print(f"CSV сохранён в {save_path}")


if __name__ == "__main__":
    thetas, mean_errs, max_errs, mean_mag = run_accuracy_experiment(
        n=1000,
        thetas=[0.1, 0.3, 0.5, 0.7, 1.0],
    )
    plot_accuracy(thetas, mean_errs, max_errs, save_path="results/accuracy.png")
    save_csv(thetas, mean_errs, max_errs, mean_mag, save_path="results/accuracy.csv")
