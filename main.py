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


if __name__ == "__main__":
    p = Particle(1.0, 0, 0)
    q = Particle(1.0, 3, 4)
    print("Расстояние:", q.distance_to(p))