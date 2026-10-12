"""Дерево Барнса-Хата для задачи N тел (2D)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from particle import Particle

# Гравитационная постоянная
G = 1.0


@dataclass
class QuadTree:
    """Узел квадродерева — квадратная ячейка в 2D.

    Атрибуты:
        x_min, x_max, y_min, y_max: границы ячейки.
        particles: список частиц в этом узле (только для листьев).
        children: 4 подузла (для внутренних узлов).
        center_mass: (x, y) центр масс всех частиц в поддереве.
        total_mass: суммарная масса всех частиц в поддереве.
        is_leaf: True, если узел — лист (без детей).
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    particles: list[Particle] = field(default_factory=list)
    children: list["QuadTree"] | None = None
    center_mass: tuple[float, float] = (0.0, 0.0)
    total_mass: float = 0.0
    is_leaf: bool = True

    def _subdivide(self) -> None:
        """Делит узел на 4 подузла.

        Вызывается, когда в лист добавляется 2-я частица.
        """
        x_mid = 0.5 * (self.x_min + self.x_max)
        y_mid = 0.5 * (self.y_min + self.y_max)

        self.children = [
            QuadTree(self.x_min, x_mid, self.y_min, y_mid),  # левый-нижний
            QuadTree(x_mid, self.x_max, self.y_min, y_mid),  # правый-нижний
            QuadTree(self.x_min, x_mid, y_mid, self.y_max),  # левый-верхний
            QuadTree(x_mid, self.x_max, y_mid, self.y_max),  # правый-верхний
        ]
        self.is_leaf = False

    def _get_quadrant(self, particle: Particle) -> int:
        """Определяет, в какой из 4 подузлов попадает частица.

        Возвращает индекс: 0 (ЛН), 1 (ПН), 2 (ЛВ), 3 (ПВ).
        """
        x_mid = 0.5 * (self.x_min + self.x_max)
        y_mid = 0.5 * (self.y_min + self.y_max)

        if particle.x < x_mid:
            if particle.y < y_mid:
                return 0  # левый-нижний
            return 2  # левый-верхний
        else:
            if particle.y < y_mid:
                return 1  # правый-нижний
            return 3  # правый-верхний

    def insert(self, particle: Particle) -> None:
        """Добавляет частицу в дерево.

        Args:
            particle: частица для добавления.
        """
        # Случай 1: лист пуст — просто добавляем частицу
        if self.is_leaf and len(self.particles) == 0:
            self.particles.append(particle)
            return

        # Случай 2: лист с 1 частицей — делим и распределяем ОБЕ
        if self.is_leaf and len(self.particles) == 1:
            old_particle = self.particles[0]
            self.particles = []  # очищаем старый список
            self._subdivide()  # создаём 4 подузла

            # Распределяем обе частицы по подузлам
            self.children[self._get_quadrant(old_particle)].insert(old_particle)
            self.children[self._get_quadrant(particle)].insert(particle)
            return

        # Случай 3: внутренний узел — рекурсивно идём в нужный подузел
        idx = self._get_quadrant(particle)
        self.children[idx].insert(particle)

    def compute_mass_center(self) -> None:
        """Считает total_mass и center_mass для узла и всех его детей.

        Вызывается на корне после построения дерева.
        Рекурсивно обходит дерево снизу вверх.
        """
        # Случай 1: лист
        if self.is_leaf:
            if len(self.particles) == 0:
                self.total_mass = 0.0
                self.center_mass = (0.0, 0.0)
                return

            total = 0.0
            cx = 0.0
            cy = 0.0
            for p in self.particles:
                total += p.mass
                cx += p.mass * p.x
                cy += p.mass * p.y

            self.total_mass = total
            self.center_mass = (cx / total, cy / total)
            return

        # Случай 2: внутренний узел — суммируем по детям
        total = 0.0
        cx = 0.0
        cy = 0.0
        for child in self.children:
            child.compute_mass_center()  # сначала дети
            total += child.total_mass
            cx += child.total_mass * child.center_mass[0]
            cy += child.total_mass * child.center_mass[1]

        self.total_mass = total
        if total > 0:
            self.center_mass = (cx / total, cy / total)
        else:
            self.center_mass = (0.0, 0.0)

    def compute_force(
        self, target: Particle, theta: float = 0.5, softening: float = 0.01
    ) -> tuple[float, float]:
        """Считает ускорение, действующее на target со стороны всех частиц дерева.

        Args:
            target: частица, для которой считаем силу.
            theta: параметр точности Барнса-Хата (0.5 — стандарт).
            softening: сглаживание для избежания деления на 0.

        Returns:
            (ax, ay) — компоненты ускорения.
        """
        ax = 0.0
        ay = 0.0

        # Если узел пустой — вклад нулевой
        if self.total_mass == 0.0:
            return 0.0, 0.0

        # Вектор от target до центра масс узла
        dx = self.center_mass[0] - target.x
        dy = self.center_mass[1] - target.y
        r2 = dx * dx + dy * dy + softening * softening
        r = math.sqrt(r2)

        # Размер ячейки
        s = self.x_max - self.x_min

        # Критерий Барнса-Хата: s/d < theta
        # ИЛИ узел — лист (дальше некуда)
        if self.is_leaf or (s / r) < theta:
            # Приближаем весь узел одной частицей (центром масс)
            inv_r3 = 1.0 / (r2 * r)
            ax += G * self.total_mass * dx * inv_r3
            ay += G * self.total_mass * dy * inv_r3
            return ax, ay

        # Иначе — спускаемся к детям
        for child in self.children:
            cax, cay = child.compute_force(target, theta, softening)
            ax += cax
            ay += cay

        return ax, ay


def compute_forces_barnes_hut(
    particles: list[Particle],
    theta: float = 0.5,
    softening: float = 0.01,
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    """Считает ускорения для всех частиц через дерево Барнса-Хата.

    Args:
        particles: список частиц.
        theta: параметр точности (0.5 — стандарт).
        softening: сглаживание.
        bounds: (x_min, x_max, y_min, y_max) корневого узла.
            Если None — вычисляется по частицам автоматически.
    """
    if len(particles) == 0:
        return

    # Определяем границы корневого узла
    if bounds is None:
        x_min = min(p.x for p in particles)
        x_max = max(p.x for p in particles)
        y_min = min(p.y for p in particles)
        y_max = max(p.y for p in particles)

        # Делаем квадрат (чтобы s = x_max - x_min = y_max - y_min)
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        size = max(x_max - x_min, y_max - y_min) * 1.01  # +1% запаса
        half = 0.5 * size

        x_min, x_max = cx - half, cx + half
        y_min, y_max = cy - half, cy + half

    # Строим дерево
    root = QuadTree(x_min, x_max, y_min, y_max)
    for p in particles:
        root.insert(p)

    # Считаем центры масс
    root.compute_mass_center()

    # Считаем силу для каждой частицы
    for p in particles:
        ax, ay = root.compute_force(p, theta, softening)
        p.ax = ax
        p.ay = ay
