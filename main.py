import math
from dataclasses import dataclass

"""Задача N тел: прямой метод и метод Барнса-Хата."""

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


if __name__ == "__main__":
    ps = two_body_system(1.0, 1.0, 1.0)
    compute_forces_direct(ps)
    for i, p in enumerate(ps):
        print(f"Частица {i}: ax={p.ax:.3f}, ay={p.ay:.3f}")