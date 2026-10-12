"""Замер времени: прямой метод vs Барнс-Хат."""

import time
import csv
from pathlib import Path

import matplotlib.pyplot as plt

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import Particle, compute_forces_direct, uniform_cluster
from barnes_hut import compute_forces_barnes_hut


def time_direct(particles: list[Particle], G: float = 1.0) -> float:
    """Замеряет время работы прямого метода."""
    start = time.perf_counter()
    compute_forces_direct(particles, G)
    return time.perf_counter() - start


def time_barnes_hut(particles: list[Particle], theta: float = 0.5) -> float:
    """Замеряет время работы Барнса-Хата."""
    start = time.perf_counter()
    compute_forces_barnes_hut(particles, theta=theta)
    return time.perf_counter() - start


def run_timing_experiment(
    n_values: list[int], repeat: int = 3, seed: int = 42
) -> tuple[list[int], list[float], list[float]]:
    """Замеряет время для разных N.

    Args:
        n_values: список N.
        repeat: сколько раз повторить каждый замер (для усреднения).
        seed: зерно генератора.

    Returns:
        (n_values, direct_times, bh_times) — списки.
    """
    direct_times = []
    bh_times = []

    for n in n_values:
        # Генерируем N частиц
        particles_d = uniform_cluster(n=n, radius=1.0, total_mass=1.0, seed=seed)
        particles_bh = uniform_cluster(n=n, radius=1.0, total_mass=1.0, seed=seed)

        # Замеряем несколько раз и берём среднее
        d_time = min(time_direct(particles_d) for _ in range(repeat))
        b_time = min(time_barnes_hut(particles_bh) for _ in range(repeat))

        direct_times.append(d_time)
        bh_times.append(b_time)

        print(f"N = {n:>6}: direct = {d_time:.6f} s, barnes_hut = {b_time:.6f} s")

    return n_values, direct_times, bh_times


def plot_timing(
    n_values: list[int],
    direct_times: list[float],
    bh_times: list[float],
    save_path: str = "results/timing.png",
) -> None:
    """Рисует график время vs N (log-log)."""
    plt.figure(figsize=(8, 6))
    plt.loglog(n_values, direct_times, "o-", label="Прямой метод O(N²)", lw=2)
    plt.loglog(n_values, bh_times, "s-", label="Барнс-Хат O(N log N)", lw=2)

    plt.xlabel("Число частиц N")
    plt.ylabel("Время (сек)")
    plt.title("Сравнение времени работы методов")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён в {save_path}")
    plt.show()
    plt.close()


def save_csv(
    n_values: list[int],
    direct_times: list[float],
    bh_times: list[float],
    save_path: str = "results/timing.csv",
) -> None:
    """Сохраняет данные в CSV."""
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "direct_time", "barnes_hut_time", "speedup"])
        for n, dt, bt in zip(n_values, direct_times, bh_times):
            writer.writerow([n, f"{dt:.8f}", f"{bt:.8f}", f"{dt/bt:.2f}"])
    print(f"CSV сохранён в {save_path}")


if __name__ == "__main__":
    n_values = [100, 300, 500, 1000, 2000]

    n, d_times, b_times = run_timing_experiment(n_values, repeat=3)
    plot_timing(n, d_times, b_times, save_path="results/timing.png")
    save_csv(n, d_times, b_times, save_path="results/timing.csv")
