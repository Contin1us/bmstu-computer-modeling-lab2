"""Задача N тел: прямой метод и метод Барнса-Хата."""


import math
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class Particle:
    """Частица с массой, положением и скоростью в 2D."""
    mass: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0

    def distance_to(self, other: "Particle") -> float:
        """Расстояние до другой частицы."""
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5

def two_body_system(m1: float, m2: float, separation: float,
                    G: float = 1.0) -> list[Particle]:
    """Два тела на круговой орбите вокруг общего центра масс."""
    total = m1 + m2
    r1 = separation * m2 / total
    r2 = separation * m1 / total

    v1 = math.sqrt(G * m2 * m2 / (total * separation))
    v2 = math.sqrt(G * m1 * m1 / (total * separation))

    p1 = Particle(m1, -r1, 0.0, 0.0,  v1)
    p2 = Particle(m2,  r2, 0.0, 0.0, -v2)
    return [p1, p2]

def uniform_cluster(n: int,
                    radius: float,
                    total_mass: float,
                    seed: int | None = None) -> list[Particle]:
    """Однородное скопление из n частиц в круге радиуса radius."""
    if seed is not None:
        random.seed(seed)

    particles = []
    mass_per_particle = total_mass / n

    for _ in range(n):
        # rejection sampling: генерируем точку в квадрате,
        # пока она не попадёт в круг
        while True:
            x = random.uniform(-radius, radius)
            y = random.uniform(-radius, radius)
            if x * x + y * y <= radius * radius:
                break

        particles.append(Particle(mass_per_particle, x, y))

    return particles

def rotating_disk(n: int,
                  radius: float,
                  total_mass: float,
                  seed: int | None = None) -> list[Particle]:
    """Вращающийся диск: частицы с тангенциальными скоростями."""
    if seed is not None:
        random.seed(seed)

    particles = []
    mass_per_particle = total_mass / n
    G = 1.0

    for _ in range(n):
        # Случайная точка в круге
        while True:
            x = random.uniform(-radius, radius)
            y = random.uniform(-radius, radius)
            if x * x + y * y <= radius * radius:
                break

        r = math.sqrt(x * x + y * y)
        if r == 0:
            vx, vy = 0.0, 0.0
        else:
            # Круговая скорость внутри диска: v = sqrt(G * M(r) / r)
            # M(r) — масса внутри радиуса r
            mass_inside = total_mass * (r / radius) ** 2
            v = 0.5 * math.sqrt(G * mass_inside / r)
            # Тангенциальное направление: перпендикулярно (x, y)
            vx = -v * y / r
            vy =  v * x / r

        particles.append(Particle(mass_per_particle, x, y, vx, vy))

    return particles


def compute_forces_direct(particles: list[Particle],
                          G: float = 1.0,
                          softening: float = 0.01) -> None:
    """Считает ускорения прямым методом O(N²)."""
    n = len(particles)
    for p in particles:
        p.ax = 0.0
        p.ay = 0.0

    eps2 = softening * softening

    for i in range(n):
        pi = particles[i]
        for j in range(i + 1, n):
            pj = particles[j]
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            r2 = dx * dx + dy * dy + eps2
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))

            pi.ax += G * pj.mass * dx * inv_r3
            pi.ay += G * pj.mass * dy * inv_r3
            pj.ax -= G * pi.mass * dx * inv_r3
            pj.ay -= G * pi.mass * dy * inv_r3


def velocity_verlet_step(particles: list[Particle],
                         dt: float,
                         G: float = 1.0) -> None:
    """Один шаг интегратора velocity Verlet."""
    # Шаг 1: половина обновления скорости
    for p in particles:
        p.vx += 0.5 * p.ax * dt
        p.vy += 0.5 * p.ay * dt

    # Шаг 2: полное обновление позиции
    for p in particles:
        p.x += p.vx * dt
        p.y += p.vy * dt

    # Шаг 3: пересчёт сил по новым позициям
    compute_forces_direct(particles, G)

    # Шаг 4: вторая половина обновления скорости
    for p in particles:
        p.vx += 0.5 * p.ax * dt
        p.vy += 0.5 * p.ay * dt


def simulate(particles: list[Particle],
             dt: float,
             steps: int,
             G: float = 1.0,
             record_every: int = 10) -> list[list[Particle]]:
    """Полная симуляция. Возвращает историю состояний."""
    history: list[list[Particle]] = []

    # Начальный расчёт сил
    compute_forces_direct(particles, G)

    for step in range(steps):
        velocity_verlet_step(particles, dt, G)

        # Записываем состояние каждые record_every шагов
        if step % record_every == 0:
            snapshot = [
                Particle(p.mass, p.x, p.y, p.vx, p.vy)
                for p in particles
            ]
            history.append(snapshot)

    return history

def plot_orbits(history: list[list[Particle]],
                title: str = "Орбиты N тел",
                save_path: str | None = None,
                show_legend: bool = True) -> None:
    """Рисует траектории всех частиц.

    Args:
        history: история состояний.
        title: заголовок.
        save_path: путь для сохранения.
        show_legend: показывать ли легенду (для N > 20 — обычно нет).
    """
    n = len(history[0])
    plt.figure(figsize=(8, 8))

    for i in range(n):
        xs = [state[i].x for state in history]
        ys = [state[i].y for state in history]
        label = f"Частица {i}" if show_legend else None
        plt.plot(xs, ys, lw=1.0, label=label, alpha=0.7)

    plt.gca().set_aspect("equal")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    if show_legend and n <= 20:
        plt.legend(fontsize=8, loc="upper right")
    plt.xlabel("x")
    plt.ylabel("y")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён в {save_path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    # Тест 1: два тела
    print("=== Тест 1: два тела ===")
    ps = two_body_system(1.0, 1.0, 1.0)
    history = simulate(ps, dt=0.01, steps=2000, record_every=10)
    plot_orbits(
        history,
        title="Орбиты двух тел",
        save_path="results/two_body.png",
        show_legend=True,   
    )

    # Тест 2: вращающийся диск
    print("\n=== Тест 2: вращающийся диск N=100 ===")
    disk = rotating_disk(n=100, radius=1.0, total_mass=10.0, seed=42)
    history_disk = simulate(disk, dt=0.0001, steps=20000, record_every=200)
    plot_orbits(
        history_disk,
        title="Вращающийся диск N=100",
        save_path="results/disk.png",
        show_legend=False,   
    )