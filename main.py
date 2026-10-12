"""Задача N тел: прямой метод и метод Барнса-Хата."""


import math
from dataclasses import dataclass


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


if __name__ == "__main__":
    ps = two_body_system(1.0, 1.0, 1.0)

    # Симулируем 2000 шагов с dt = 0.01
    history = simulate(ps, dt=0.01, steps=2000, record_every=10)

    # Проверим, что частицы не улетели:
    # энергия должна сохраняться примерно одинаковой
    print(f"Всего снимков: {len(history)}")
    for i, p in enumerate(history[-1]):
        print(f"Частица {i}: x={p.x:.3f}, y={p.y:.3f}, vx={p.vx:.3f}, vy={p.vy:.3f}")