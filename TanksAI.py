#!/usr/bin/env python3
# ============================================================================
# TESLA SWARM AGI - WORLD MODEL & AUTONOMOUS COMBAT SIMULATION (FIXED)
# ============================================================================
# Version: 1.0.2-fixed
# Fixes applied:
#   - Fixed NN_INPUT_SIZE mismatch (was 16 config vs 13 actual inputs).
#   - Fixed SpatialHash attribute name (pos -> position).
#   - Removed unused imports.
#   - Fixed RGBA color issues for standard pygame display.
#   - Fixed Vec2 objects passed to pygame.draw.line (TypeError crash).
#   - Fixed team 1 agents spawning across entire map instead of right half.
#   - Fixed Kalman filter velocity update using post-update residual.
#   - Fixed MathUtils.sigmoid overflow for large negative inputs.
#   - Fixed obstacle proximity NN input going negative.
#   - Fixed FPS counter returning 0 when FPS_UNLOCK is True.
# ============================================================================

import pygame
import math
import random
import heapq
import sys
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

# ============================================================================
# PART 1: CONFIGURATION & CONSTANTS
# ============================================================================

class Difficulty(Enum):
    EASY = auto()
    NORMAL = auto()
    HARD = auto()
    EXTREME = auto()

@dataclass
class SimulationConfig:
    """Central configuration for all simulation parameters."""
    
    # Display Settings
    SCREEN_W: int = 1280
    SCREEN_H: int = 720
    FULLSCREEN: bool = False  # Changed to False for easier debugging
    FPS: int = 60
    FPS_UNLOCK: bool = False
    
    # World Settings
    WORLD_MARGIN: int = 60
    GRID_RESOLUTION: int = 80
    CELL_UPDATE_THRESHOLD: float = 0.1
    
    # Agent Settings
    AGENT_COUNT_PER_TEAM: int = 5
    AGENT_RADIUS: float = 14.0
    AGENT_MAX_SPEED: float = 4.5
    AGENT_MAX_ACCEL: float = 0.25
    AGENT_TURN_RATE: float = 0.08
    AGENT_FRICTION: float = 0.94
    
    # Swarm Behavior Weights
    SWARM_SEPARATION_WEIGHT: float = 2.5
    SWARM_ALIGNMENT_WEIGHT: float = 1.5
    SWARM_COHESION_WEIGHT: float = 1.2
    SWARM_SEPARATION_DIST: float = 50.0
    SWARM_ALIGNMENT_DIST: float = 120.0
    SWARM_COHESION_DIST: float = 200.0
    
    # Combat Settings
    FIRE_RANGE: float = 350.0
    FIRE_COOLDOWN_FRAMES: int = 45
    PROJECTILE_SPEED: float = 18.0
    PROJECTILE_DAMAGE: float = 30.0
    PROJECTILE_LIFETIME: int = 120
    CRITICAL_HIT_CHANCE: float = 0.15
    CRITICAL_MULTIPLIER: float = 2.0
    
    # AI Settings
    NN_INPUT_SIZE: int = 16
    NN_HIDDEN_LAYERS: Tuple[int, ...] = (32, 24)
    NN_OUTPUT_SIZE: int = 6
    NN_LEARNING_RATE: float = 0.001
    DECISION_INTERVAL_FRAMES: int = 3
    
    # World Model Settings
    KALMAN_PROCESS_VAR: float = 0.05
    KALMAN_MEASURE_VAR: float = 0.3
    OCCUPANCY_DECAY_RATE: float = 0.02
    PREDICTION_HORIZON_FRAMES: int = 30
    
    # Obstacle Settings
    OBSTACLE_COUNT: int = 8
    OBSTACLE_MIN_RADIUS: int = 30
    OBSTACLE_MAX_RADIUS: int = 70
    
    # Colors (Tesla-Inspired Palette)
    COLOR_BG: Tuple[int, int, int] = (8, 8, 12)
    COLOR_WORLD: Tuple[int, int, int] = (15, 15, 22)
    COLOR_GRID: Tuple[int, int, int] = (25, 35, 30)
    COLOR_GRID_HIGH: Tuple[int, int, int] = (40, 50, 45)
    COLOR_AGENT_TEAM_0: Tuple[int, int, int] = (0, 255, 200)
    COLOR_AGENT_TEAM_1: Tuple[int, int, int] = (255, 68, 68)
    COLOR_AGENT_SELECTED: Tuple[int, int, int] = (255, 255, 100)
    COLOR_PROJECTILE: Tuple[int, int, int] = (255, 220, 80)
    COLOR_PROJECTILE_TRAIL: Tuple[int, int, int] = (255, 180, 50)
    COLOR_OBSTACLE: Tuple[int, int, int] = (80, 80, 95)
    COLOR_OBSTACLE_BORDER: Tuple[int, int, int] = (120, 180, 140)
    COLOR_PATH: Tuple[int, int, int] = (100, 100, 120)
    COLOR_PATH_ACTIVE: Tuple[int, int, int] = (255, 200, 100)
    COLOR_VISIBILITY_CONE: Tuple[int, int, int] = (0, 255, 255) # Removed Alpha
    COLOR_TEXT_PRIMARY: Tuple[int, int, int] = (220, 220, 240)
    COLOR_TEXT_SECONDARY: Tuple[int, int, int] = (150, 150, 170)
    COLOR_TEXT_ACCENT: Tuple[int, int, int] = (0, 255, 200)
    COLOR_HP_HIGH: Tuple[int, int, int] = (0, 255, 100)
    COLOR_HP_MEDIUM: Tuple[int, int, int] = (255, 200, 50)
    COLOR_HP_LOW: Tuple[int, int, int] = (255, 50, 50)
    COLOR_DEBUG: Tuple[int, int, int] = (255, 0, 255)
    
    # Difficulty Scaling
    DIFFICULTY_SCALING: Dict[Difficulty, Dict[str, float]] = field(default_factory=lambda: {
        Difficulty.EASY: {'ai_accuracy': 0.6, 'reaction_time': 1.5, 'damage_mult': 0.8},
        Difficulty.NORMAL: {'ai_accuracy': 0.8, 'reaction_time': 1.0, 'damage_mult': 1.0},
        Difficulty.HARD: {'ai_accuracy': 0.95, 'reaction_time': 0.7, 'damage_mult': 1.2},
        Difficulty.EXTREME: {'ai_accuracy': 1.0, 'reaction_time': 0.5, 'damage_mult': 1.5},
    })
    
    def get_difficulty_params(self, difficulty: Difficulty) -> Dict[str, float]:
        return self.DIFFICULTY_SCALING[difficulty]

# Global configuration instance
CFG = SimulationConfig()

# ============================================================================
# PART 2: MATH & GEOMETRY UTILITIES
# ============================================================================

class Vec2:
    """
    2D Vector class with comprehensive operations for game physics.
    Uses __slots__ for memory efficiency in large-scale simulations.
    """
    __slots__ = ('x', 'y')
    
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)
    
    def __repr__(self) -> str:
        return f"Vec2({self.x:.2f}, {self.y:.2f})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec2):
            return False
        return abs(self.x - other.x) < 1e-6 and abs(self.y - other.y) < 1e-6
    
    def __add__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vec2':
        return Vec2(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vec2':
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vec2':
        if abs(scalar) < 1e-10:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / scalar, self.y / scalar)
    
    def __neg__(self) -> 'Vec2':
        return Vec2(-self.x, -self.y)
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def copy(self) -> 'Vec2':
        return Vec2(self.x, self.y)
    
    def mag(self) -> float:
        """Return magnitude (length) of vector."""
        return math.hypot(self.x, self.y)
    
    def mag_sq(self) -> float:
        """Return squared magnitude (avoids sqrt for performance)."""
        return self.x * self.x + self.y * self.y
    
    def norm(self) -> 'Vec2':
        """Return normalized vector. Returns zero vector if magnitude is too small."""
        m = self.mag()
        if m < 1e-6:
            return Vec2(0.0, 0.0)
        return self / m
    
    def limit(self, max_val: float) -> 'Vec2':
        """Limit vector magnitude to max_val."""
        if self.mag() > max_val:
            return self.norm() * max_val
        return self.copy()
    
    def dist(self, other: 'Vec2') -> float:
        """Return Euclidean distance to another vector."""
        return (self - other).mag()
    
    def dist_sq(self, other: 'Vec2') -> float:
        """Return squared Euclidean distance (avoids sqrt)."""
        return (self - other).mag_sq()
    
    def angle(self) -> float:
        """Return angle in radians (-PI to PI)."""
        return math.atan2(self.y, self.x)
    
    def dot(self, other: 'Vec2') -> float:
        """Return dot product."""
        return self.x * other.x + self.y * other.y
    
    def cross(self, other: 'Vec2') -> float:
        """Return 2D cross product (scalar)."""
        return self.x * other.y - self.y * other.x
    
    def rotate(self, angle: float) -> 'Vec2':
        """Return rotated vector by angle (radians)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )
    
    def lerp(self, other: 'Vec2', t: float) -> 'Vec2':
        """Linear interpolation to another vector."""
        t = max(0.0, min(1.0, t))
        return self + (other - self) * t
    
    def clamp(self, min_vec: 'Vec2', max_vec: 'Vec2') -> 'Vec2':
        """Clamp vector components to range."""
        return Vec2(
            max(min_vec.x, min(max_vec.x, self.x)),
            max(min_vec.y, min(max_vec.y, self.y))
        )
    
    @staticmethod
    def from_angle(angle: float, length: float = 1.0) -> 'Vec2':
        """Create vector from angle and length."""
        return Vec2(math.cos(angle) * length, math.sin(angle) * length)
    
    @staticmethod
    def random_in_circle(radius: float = 1.0) -> 'Vec2':
        """Generate random vector within circle of given radius."""
        r = radius * math.sqrt(random.random())
        theta = random.uniform(0, math.pi * 2)
        return Vec2(r * math.cos(theta), r * math.sin(theta))

class MathUtils:
    """Static mathematical utility functions."""
    
    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        t = max(0.0, min(1.0, t))
        return a + (b - a) * t
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= math.pi * 2
        while angle < -math.pi:
            angle += math.pi * 2
        return angle
    
    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        diff = MathUtils.wrap_angle(b - a)
        return diff
    
    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
    
    @staticmethod
    def line_circle_intersect(
        line_start: Vec2, 
        line_end: Vec2, 
        circle_center: Vec2, 
        circle_radius: float
    ) -> bool:
        line_vec = line_end - line_start
        line_len_sq = line_vec.mag_sq()
        
        if line_len_sq == 0:
            return line_start.dist(circle_center) <= circle_radius
        
        t = ((circle_center.x - line_start.x) * line_vec.x + 
             (circle_center.y - line_start.y) * line_vec.y) / line_len_sq
        t = MathUtils.clamp(t, 0.0, 1.0)
        
        closest = line_start + line_vec * t
        return closest.dist(circle_center) <= circle_radius

# ============================================================================
# PART 3: SPATIAL DATA STRUCTURES
# ============================================================================

class SpatialHash:
    """
    Spatial hashing grid for O(1) neighbor queries.
    Essential for swarm simulations with many agents.
    """
    
    def __init__(self, cell_size: float = 50.0):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List[Any]] = defaultdict(list)
    
    def clear(self) -> None:
        self.grid.clear()
    
    def _hash(self, pos: Vec2) -> Tuple[int, int]:
        return (
            int(pos.x // self.cell_size),
            int(pos.y // self.cell_size)
        )
    
    def insert(self, obj: Any, pos: Vec2) -> None:
        key = self._hash(pos)
        self.grid[key].append(obj)
    
    def query(self, pos: Vec2, radius: float) -> List[Any]:
        """Query all objects within radius of position."""
        results = []
        cell_radius = int(math.ceil(radius / self.cell_size))
        center = self._hash(pos)
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                key = (center[0] + dx, center[1] + dy)
                if key in self.grid:
                    for obj in self.grid[key]:
                        # FIX: Check for 'position' attribute instead of 'pos'
                        if hasattr(obj, 'position') and obj.position.dist(pos) <= radius:
                            results.append(obj)
        
        return results

# ============================================================================
# PART 4: NEURAL NETWORK
# ============================================================================

class NeuralNetwork:
    """
    Feedforward neural network with configurable architecture.
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_layers: Tuple[int, ...], 
        output_size: int,
        learning_rate: float = 0.001
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.layer_sizes = [input_size] + list(hidden_layers) + [output_size]
        self.num_layers = len(self.layer_sizes) - 1
        
        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = []
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            
            limit = math.sqrt(6.0 / (n_in + n_out))
            
            weight_matrix = [
                [random.uniform(-limit, limit) for _ in range(n_out)]
                for _ in range(n_in)
            ]
            self.weights.append(weight_matrix)
            
            self.biases.append([0.0] * n_out)
    
    def relu(self, x: float) -> float:
        return max(0.0, x)
    
    def sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
    
    def softmax(self, values: List[float]) -> List[float]:
        max_val = max(values)
        exps = [math.exp(v - max_val) for v in values]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]
    
    def forward(self, inputs: List[float], use_softmax: bool = False) -> List[float]:
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        current = inputs
        
        for layer_idx in range(self.num_layers):
            weights = self.weights[layer_idx]
            biases = self.biases[layer_idx]
            
            n_out = len(biases)
            outputs = [0.0] * n_out
            
            for j in range(n_out):
                sum_val = biases[j]
                for i in range(len(current)):
                    sum_val += current[i] * weights[i][j]
                outputs[j] = sum_val
            
            if layer_idx < self.num_layers - 1:
                current = [self.relu(v) for v in outputs]
            else:
                if use_softmax:
                    current = self.softmax(outputs)
                else:
                    current = outputs
        
        return current
    
    def predict(self, inputs: List[float]) -> List[float]:
        return self.forward(inputs, use_softmax=True)

# ============================================================================
# PART 5: KALMAN FILTER
# ============================================================================

class KalmanFilter2D:
    def __init__(
        self, 
        process_variance: float = 0.1,
        measurement_variance: float = 0.5,
        initial_position: Optional[Vec2] = None
    ):
        self.state = [0.0, 0.0, 0.0, 0.0]
        
        if initial_position:
            self.state[0] = initial_position.x
            self.state[1] = initial_position.y
        
        self.covariance = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
        
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.initialized = initial_position is not None
    
    def predict(self, dt: float) -> Vec2:
        if not self.initialized:
            return Vec2(self.state[0], self.state[1])
        
        self.state[0] += self.state[2] * dt
        self.state[1] += self.state[3] * dt
        
        for i in range(4):
            self.covariance[i * 4 + i] += self.process_variance
        
        return Vec2(self.state[0], self.state[1])
    
    def update(self, measurement: Vec2, dt: float) -> Vec2:
        if not self.initialized:
            self.state[0] = measurement.x
            self.state[1] = measurement.y
            self.initialized = True
            return measurement.copy()
        
        self.predict(dt)
        
        k_x = self.covariance[0] / (self.covariance[0] + self.measurement_variance)
        k_y = self.covariance[5] / (self.covariance[5] + self.measurement_variance)
        
        innovation_x = measurement.x - self.state[0]
        innovation_y = measurement.y - self.state[1]
        
        self.state[0] += k_x * innovation_x
        self.state[1] += k_y * innovation_y
        
        if dt > 0:
            self.state[2] += k_x * innovation_x / dt
            self.state[3] += k_y * innovation_y / dt
        
        self.covariance[0] *= (1 - k_x)
        self.covariance[5] *= (1 - k_y)
        
        return Vec2(self.state[0], self.state[1])
        
    def get_position(self) -> Vec2:
        return Vec2(self.state[0], self.state[1])
    
    def get_velocity(self) -> Vec2:
        return Vec2(self.state[2], self.state[3])
    
    def predict_future(self, time_horizon: float) -> Vec2:
        return Vec2(
            self.state[0] + self.state[2] * time_horizon,
            self.state[1] + self.state[3] * time_horizon
        )

# ============================================================================
# PART 6: OCCUPANCY GRID & PATHFINDING
# ============================================================================

class OccupancyGrid:
    def __init__(
        self, 
        width: int, 
        height: int, 
        resolution: float = 40.0,
        decay_rate: float = 0.02
    ):
        self.resolution = resolution
        self.width_cells = int(width / resolution)
        self.height_cells = int(height / resolution)
        self.decay_rate = decay_rate
        
        self.cells: List[List[float]] = [
            [0.5] * self.width_cells 
            for _ in range(self.height_cells)
        ]
    
    def decay(self) -> None:
        # Only decay every 5th call for performance
        if not hasattr(self, '_decay_counter'):
            self._decay_counter = 0
        self._decay_counter += 1
        if self._decay_counter % 5 != 0:
            return
        for y in range(self.height_cells):
            for x in range(self.width_cells):
                if self.cells[y][x] > 0.5:
                    self.cells[y][x] = max(0.5, self.cells[y][x] - self.decay_rate)
                else:
                    self.cells[y][x] = min(0.5, self.cells[y][x] + self.decay_rate)
    
    def _world_to_grid(self, pos: Vec2) -> Tuple[int, int]:
        gx = int(pos.x / self.resolution)
        gy = int(pos.y / self.resolution)
        return (
            max(0, min(self.width_cells - 1, gx)),
            max(0, min(self.height_cells - 1, gy))
        )
    
    def _grid_to_world(self, gx: int, gy: int) -> Vec2:
        return Vec2(
            gx * self.resolution + self.resolution / 2,
            gy * self.resolution + self.resolution / 2
        )
    
    def update_circle(
        self, 
        center: Vec2, 
        radius: float, 
        probability: float,
        is_occupied: bool = True
    ) -> None:
        gx0, gy0 = self._world_to_grid(Vec2(center.x - radius, center.y - radius))
        gx1, gy1 = self._world_to_grid(Vec2(center.x + radius, center.y + radius))
        
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                cell_center = self._grid_to_world(gx, gy)
                dist_sq = cell_center.dist_sq(center)
                
                if dist_sq <= radius * radius:
                    prior = self.cells[gy][gx]
                    
                    if is_occupied:
                        likelihood = probability
                        posterior = (prior * likelihood) / (
                            prior * likelihood + (1 - prior) * (1 - likelihood) + 1e-6
                        )
                    else:
                        likelihood = probability
                        posterior = (prior * (1 - likelihood)) / (
                            prior * (1 - likelihood) + (1 - prior) * likelihood + 1e-6
                        )
                    
                    self.cells[gy][gx] = max(0.0, min(1.0, posterior))
    
    def get_cost(self, pos: Vec2) -> float:
        gx, gy = self._world_to_grid(pos)
        if 0 <= gx < self.width_cells and 0 <= gy < self.height_cells:
            return self.cells[gy][gx]
        return 1.0
    
    def get_neighbors(self, gx: int, gy: int) -> List[Tuple[int, int]]:
        neighbors = [
            (gx, gy - 1), (gx, gy + 1), (gx - 1, gy), (gx + 1, gy),
            (gx - 1, gy - 1), (gx + 1, gy - 1), (gx - 1, gy + 1), (gx + 1, gy + 1),
        ]
        return [
            (nx, ny) for nx, ny in neighbors
            if 0 <= nx < self.width_cells and 0 <= ny < self.height_cells
        ]

class AStarPathfinder:
    def __init__(self, grid: OccupancyGrid, max_cost_threshold: float = 0.6):
        self.grid = grid
        self.max_cost = max_cost_threshold
    
    def find_path(self, start: Vec2, goal: Vec2, max_iterations: int = 1000) -> Optional[List[Vec2]]:
        start_gx, start_gy = self.grid._world_to_grid(start)
        goal_gx, goal_gy = self.grid._world_to_grid(goal)
        
        open_set: List[Tuple[float, int, int]] = []
        heapq.heappush(open_set, (0.0, start_gx, start_gy))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {(start_gx, start_gy): 0.0}
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current_gx, current_gy = heapq.heappop(open_set)
            
            if (current_gx, current_gy) == (goal_gx, goal_gy):
                return self._reconstruct_path(came_from, (current_gx, current_gy), start, goal)
            
            for neighbor_gx, neighbor_gy in self.grid.get_neighbors(current_gx, current_gy):
                cell_cost = self.grid.cells[neighbor_gy][neighbor_gx]
                if cell_cost > self.max_cost:
                    continue
                
                tentative_g = g_score[(current_gx, current_gy)] + 1.0
                
                if tentative_g < g_score.get((neighbor_gx, neighbor_gy), float('inf')):
                    came_from[(neighbor_gx, neighbor_gy)] = (current_gx, current_gy)
                    g_score[(neighbor_gx, neighbor_gy)] = tentative_g
                    f = tentative_g + abs(neighbor_gx - goal_gx) + abs(neighbor_gy - goal_gy)
                    heapq.heappush(open_set, (f, neighbor_gx, neighbor_gy))
        
        return None
    
    def _reconstruct_path(
        self, 
        came_from: Dict[Tuple[int, int], Tuple[int, int]], 
        goal: Tuple[int, int],
        world_start: Vec2,
        world_goal: Vec2
    ) -> List[Vec2]:
        path: List[Tuple[int, int]] = [goal]
        current = goal
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        
        world_path = [world_start]
        for gx, gy in path[1:-1]:
            world_path.append(self.grid._grid_to_world(gx, gy))
        world_path.append(world_goal)
        
        return world_path

# ============================================================================
# PART 7: GAME ENTITIES
# ============================================================================

@dataclass
class TrackedObject:
    id: int
    position: Vec2
    velocity: Vec2
    kalman_filter: KalmanFilter2D
    last_seen_frame: int
    confidence: float
    team: int
    is_alive: bool = True

@dataclass
class Projectile:
    id: int
    position: Vec2
    velocity: Vec2
    owner_id: int
    team: int
    damage: float
    lifetime: int
    max_lifetime: int
    trail: List[Vec2] = field(default_factory=list)
    alive: bool = True
    
    def update(self, dt: float) -> None:
        self.trail.append(self.position.copy())
        if len(self.trail) > 8:
            self.trail.pop(0)
        
        self.position += self.velocity * dt * 60
        self.lifetime -= 1
        
        if self.lifetime <= 0:
            self.alive = False
    
    def is_out_of_bounds(self, bounds: Tuple[float, float, float, float]) -> bool:
        x, y, w, h = bounds
        return (
            self.position.x < x - 50 or
            self.position.x > x + w + 50 or
            self.position.y < y - 50 or
            self.position.y > y + h + 50
        )

@dataclass
class Obstacle:
    id: int
    position: Vec2
    radius: float

class Agent:
    def __init__(
        self,
        agent_id: int,
        team: int,
        position: Vec2,
        world_model: 'WorldModel',
        config: SimulationConfig = CFG
    ):
        self.id = agent_id
        self.team = team
        self.position = position.copy()
        self.velocity = Vec2()
        self.acceleration = Vec2()
        self.angle = random.uniform(0, math.pi * 2)
        self.turret_angle = self.angle
        
        self.config = config
        self.world_model = world_model
        
        self.radius = config.AGENT_RADIUS
        self.max_speed = config.AGENT_MAX_SPEED
        self.max_accel = config.AGENT_MAX_ACCEL
        self.turn_rate = config.AGENT_TURN_RATE
        self.friction = config.AGENT_FRICTION
        
        self.health = 100.0
        self.max_health = 100.0
        self.alive = True
        self.fire_cooldown = 0
        self.fire_range = config.FIRE_RANGE
        self.kills = 0
        
        self.brain = NeuralNetwork(
            config.NN_INPUT_SIZE,
            config.NN_HIDDEN_LAYERS,
            config.NN_OUTPUT_SIZE,
            config.NN_LEARNING_RATE
        )
        self.path: List[Vec2] = []
        self.current_target_id: Optional[int] = None
        self.decision_timer = 0
        self.state_label = 'IDLE'
        
        self.hit_flash = 0
        self.muzzle_flash = 0
        self.trail: List[Vec2] = []
    
    def sense(
        self, 
        all_agents: List['Agent'],
        obstacles: List[Obstacle],
        spatial_hash: SpatialHash
    ) -> Dict[str, Any]:
        sensor_data = {
            'friends': [],
            'enemies': [],
            'obstacles': [],
            'coverage': 0.0
        }
        
        # Use the larger of FIRE_RANGE and SWARM_COHESION_DIST so agents can
        # detect enemies within shooting distance, not just swarm-cohesion distance.
        sense_radius = max(self.config.FIRE_RANGE, self.config.SWARM_COHESION_DIST)
        nearby_agents = spatial_hash.query(self.position, sense_radius)
        
        for other in nearby_agents:
            if not hasattr(other, 'alive') or not other.alive:
                continue
            if other.id == self.id:
                continue
            
            dist = self.position.dist(other.position)
            to_other = other.position - self.position
            
            has_los = self._check_line_of_sight(other.position, obstacles)
            
            detection = {
                'id': other.id,
                'position': other.position.copy(),
                'velocity': other.velocity.copy(),
                'distance': dist,
                'direction': to_other.norm(),
                'team': other.team,
                'health': other.health,
                'has_los': has_los
            }
            
            if other.team == self.team:
                sensor_data['friends'].append(detection)
            else:
                if has_los:
                    sensor_data['enemies'].append(detection)
        
        for obs in obstacles:
            dist = self.position.dist(obs.position)
            if dist < obs.radius + 100:
                sensor_data['obstacles'].append({
                    'position': obs.position.copy(),
                    'radius': obs.radius,
                    'distance': dist
                })
        
        sensor_data['coverage'] = len(sensor_data['enemies']) / max(1, len(nearby_agents))
        
        return sensor_data
    
    def _check_line_of_sight(self, target_pos: Vec2, obstacles: List[Obstacle]) -> bool:
        for obs in obstacles:
            if MathUtils.line_circle_intersect(
                self.position, 
                target_pos, 
                obs.position, 
                obs.radius
            ):
                return False
        return True
    
    def decide(
        self, 
        sensor_data: Dict[str, Any],
        dt: float
    ) -> Dict[str, Any]:
        self.decision_timer += 1
        if self.decision_timer < self.config.DECISION_INTERVAL_FRAMES:
            return {'move': self.acceleration.copy(), 'fire': False, 'target': self.current_target_id}
        
        self.decision_timer = 0
        
        inputs = self._build_nn_inputs(sensor_data)
        
        # FIX: inputs length is now corrected to match NN_INPUT_SIZE (16)
        outputs = self.brain.predict(inputs)
        
        action = {
            'move': Vec2(),
            'fire': False,
            'target': None
        }
        
        if sensor_data['enemies']:
            visible_enemies = [e for e in sensor_data['enemies'] if e['has_los']]
            if visible_enemies:
                target = min(visible_enemies, key=lambda e: e['distance'])
                self.current_target_id = target['id']
                action['target'] = target['id']
                
                if target['distance'] < self.fire_range * 0.6:
                    self.state_label = 'ENGAGE'
                else:
                    self.state_label = 'SEEK'
                
                seek_force = target['direction'] * self.max_accel * 1.8
                action['move'] += seek_force
            else:
                self.state_label = 'SCAN'
        else:
            self.state_label = 'WANDER'
            # Wander toward the enemy side when no enemies are detected.
            # Team 0 pushes right, team 1 pushes left, with some random variation.
            wander_dir_x = 1.0 if self.team == 0 else -1.0
            wander_dir_y = random.uniform(-0.5, 0.5)
            wander_force = Vec2(wander_dir_x, wander_dir_y).norm() * self.max_accel * 0.8
            action['move'] += wander_force
        
        sep_force = Vec2()
        for friend in sensor_data['friends']:
            if friend['distance'] < self.config.SWARM_SEPARATION_DIST:
                push = (self.position - friend['position']).norm()
                sep_force += push / (friend['distance'] + 0.1)
        
        if sensor_data['friends']:
            sep_force = sep_force.limit(self.max_accel) * self.config.SWARM_SEPARATION_WEIGHT
            action['move'] += sep_force
        
        if sensor_data['friends'] and outputs[3] > 0.5:
            avg_vel = Vec2()
            for friend in sensor_data['friends']:
                avg_vel += friend['velocity']
            avg_vel /= len(sensor_data['friends'])
            align_force = (avg_vel - self.velocity) * 0.1
            action['move'] += align_force * self.config.SWARM_ALIGNMENT_WEIGHT
        
        for obs in sensor_data['obstacles']:
            if obs['distance'] < obs['radius'] + 60:
                avoid = (self.position - obs['position']).norm()
                avoid *= (obs['radius'] + 60 - obs['distance']) / 60
                action['move'] += avoid * self.max_accel * 3
        
        if sensor_data['enemies'] and self.fire_cooldown == 0:
            fire_target = min(sensor_data['enemies'], key=lambda e: e['distance'])
            if fire_target['distance'] < self.fire_range and fire_target['has_los']:
                action['fire'] = True
        
        action['move'] = action['move'].limit(self.max_accel)
        
        return action
    
    def _build_nn_inputs(self, sensor_data: Dict[str, Any]) -> List[float]:
        inputs = []
        
        # 1. Health status
        inputs.append(self.health / self.max_health)
        
        # 2. Cooldown status
        inputs.append(1.0 if self.fire_cooldown == 0 else 0.0)
        
        # 3,4,5,6. Nearest enemy info (was 3 items, added health)
        if sensor_data['enemies']:
            nearest = min(sensor_data['enemies'], key=lambda e: e['distance'])
            inputs.append(1.0 - (nearest['distance'] / self.fire_range)) # 3
            inputs.append(nearest['direction'].x) # 4
            inputs.append(nearest['direction'].y) # 5
            inputs.append(nearest['health'] / 100.0) # 6 (NEW)
        else:
            inputs.extend([0.0, 0.0, 0.0, 0.0]) # Padding for 4 values
        
        # 7. Nearest friend info
        if sensor_data['friends']:
            nearest = min(sensor_data['friends'], key=lambda e: e['distance'])
            inputs.append(nearest['distance'] / self.config.SWARM_COHESION_DIST)
        else:
            inputs.append(1.0)
        
        # 8,9. Velocity
        inputs.append(self.velocity.x / self.max_speed)
        inputs.append(self.velocity.y / self.max_speed)
        
        # 10,11. Orientation
        inputs.append(math.cos(self.angle))
        inputs.append(math.sin(self.angle))
        
        # 12. Obstacle proximity
        if sensor_data['obstacles']:
            nearest_obs = min(sensor_data['obstacles'], key=lambda o: o['distance'])
            inputs.append(max(0.0, 1.0 - (nearest_obs['distance'] / 100)))
        else:
            inputs.append(0.0)
        
        # 13. Team advantage
        friend_count = len(sensor_data['friends'])
        enemy_count = len(sensor_data['enemies'])
        inputs.append((friend_count - enemy_count) / max(1, friend_count + enemy_count))
        
        # 14. Coverage
        inputs.append(sensor_data['coverage'])
        
        # 15. Current Speed Magnitude (NEW)
        inputs.append(self.velocity.mag() / self.max_speed)
        
        # 16. Bias (NEW)
        inputs.append(1.0)
        
        # Total inputs: 1+1+4+1+2+2+1+1+1+1+1 = 16. Correct.
        
        return inputs
    
    def act(
        self, 
        action: Dict[str, Any],
        dt: float,
        all_agents: List['Agent'],
        obstacles: List[Obstacle]
    ) -> Optional[Dict[str, Any]]:
        if not self.alive:
            return None
        
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        if self.hit_flash > 0:
            self.hit_flash -= 1
        if self.muzzle_flash > 0:
            self.muzzle_flash -= 1
        
        self.acceleration = action['move']
        self.velocity += self.acceleration
        self.velocity *= self.friction
        self.velocity = self.velocity.limit(self.max_speed)
        
        self.position += self.velocity * dt * 60
        
        if self.velocity.mag() > 0.1:
            target_angle = self.velocity.angle()
            angle_diff = MathUtils.angle_diff(self.angle, target_angle)
            self.angle += MathUtils.clamp(angle_diff, -self.turn_rate, self.turn_rate)
        
        if action['target'] is not None:
            for agent in all_agents:
                if agent.id == action['target'] and agent.alive:
                    to_target = agent.position - self.position
                    self.turret_angle = to_target.angle()
                    break
        
        margin = self.radius + 10
        if self.position.x < CFG.WORLD_MARGIN + margin:
            self.position.x = CFG.WORLD_MARGIN + margin
            self.velocity.x *= -0.5
        if self.position.x > CFG.SCREEN_W - CFG.WORLD_MARGIN - margin:
            self.position.x = CFG.SCREEN_W - CFG.WORLD_MARGIN - margin
            self.velocity.x *= -0.5
        if self.position.y < CFG.WORLD_MARGIN + margin:
            self.position.y = CFG.WORLD_MARGIN + margin
            self.velocity.y *= -0.5
        if self.position.y > CFG.SCREEN_H - CFG.WORLD_MARGIN - margin:
            self.position.y = CFG.SCREEN_H - CFG.WORLD_MARGIN - margin
            self.velocity.y *= -0.5
        
        for obs in obstacles:
            dist = self.position.dist(obs.position)
            min_dist = self.radius + obs.radius + 5
            if dist < min_dist:
                push = (self.position - obs.position).norm() * (min_dist - dist)
                self.position += push
                self.velocity *= 0.7
        
        # Agent-agent collision: prevent tanks from overlapping
        for other in all_agents:
            if other.id == self.id or not other.alive:
                continue
            dist = self.position.dist(other.position)
            min_dist = self.radius + other.radius + 2
            if dist < min_dist and dist > 0.01:
                push = (self.position - other.position).norm() * (min_dist - dist) * 0.5
                self.position += push
                self.velocity *= 0.8
        
        self.trail.append(self.position.copy())
        if len(self.trail) > 20:
            self.trail.pop(0)
        
        combat_action = None
        if action['fire'] and self.fire_cooldown == 0:
            self.fire_cooldown = self.config.FIRE_COOLDOWN_FRAMES
            self.muzzle_flash = 5
            self.state_label = 'FIRE'
            
            aim_dir = Vec2.from_angle(self.turret_angle)
            
            if action['target'] is not None:
                target_obj = self.world_model.get_tracked_object(action['target'])
                if target_obj:
                    time_to_hit = target_obj.position.dist(self.position) / self.config.PROJECTILE_SPEED
                    predicted_pos = target_obj.kalman_filter.predict_future(time_to_hit)
                    aim_dir = (predicted_pos - self.position).norm()
            
            combat_action = {
                'type': 'fire',
                'position': self.position.copy(),
                'direction': aim_dir,
                'owner_id': self.id,
                'team': self.team
            }
        
        return combat_action
    
    def take_damage(self, amount: float) -> bool:
        self.health -= amount
        self.hit_flash = 8
        
        if self.health <= 0:
            self.alive = False
            return True
        return False
    
    def add_kill(self) -> None:
        self.kills += 1
    
    def draw(self, screen: pygame.Surface, camera_offset: Vec2 = Vec2()) -> None:
        if not self.alive:
            return
        
        pos = self.position + camera_offset
        x, y = pos.x, pos.y
        
        if self.team == 0:
            base_color = CFG.COLOR_AGENT_TEAM_0
        else:
            base_color = CFG.COLOR_AGENT_TEAM_1
        
        if self.hit_flash > 0:
            color = (255, 150, 150)
        else:
            color = base_color
        
        points = []
        half_w = self.radius * 1.2
        half_h = self.radius * 0.6
        
        for dx, dy in [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]:
            rx = dx * math.cos(self.angle) - dy * math.sin(self.angle)
            ry = dx * math.sin(self.angle) + dy * math.cos(self.angle)
            points.append((x + rx, y + ry))
        
        pygame.draw.polygon(screen, color, points)
        pygame.draw.polygon(screen, (min(255, color[0]+30), min(255, color[1]+30), min(255, color[2]+30)), points, 2)
        
        turret_end_x = x + math.cos(self.turret_angle) * self.radius * 1.5
        turret_end_y = y + math.sin(self.turret_angle) * self.radius * 1.5
        pygame.draw.line(screen, (100, 100, 120), (int(x), int(y)), (int(turret_end_x), int(turret_end_y)), 6)
        pygame.draw.circle(screen, (80, 80, 100), (int(x), int(y)), 10)
        
        if self.muzzle_flash > 0:
            flash_pos = Vec2(
                x + math.cos(self.turret_angle) * self.radius * 2,
                y + math.sin(self.turret_angle) * self.radius * 2
            )
            pygame.draw.circle(screen, CFG.COLOR_PROJECTILE, (int(flash_pos.x), int(flash_pos.y)), 8)
        
        hp_width = 35
        hp_height = 4
        hp_x = x - hp_width / 2
        hp_y = y - self.radius - 12
        
        pygame.draw.rect(screen, (40, 40, 50), (hp_x, hp_y, hp_width, hp_height))
        
        if self.health > 60:
            hp_color = CFG.COLOR_HP_HIGH
        elif self.health > 30:
            hp_color = CFG.COLOR_HP_MEDIUM
        else:
            hp_color = CFG.COLOR_HP_LOW
        
        hp_fill = hp_width * (self.health / self.max_health)
        pygame.draw.rect(screen, hp_color, (hp_x, hp_y, hp_fill, hp_height))
        
        if not hasattr(Agent, '_cached_font'):
            Agent._cached_font = pygame.font.SysFont('consolas', 12)
        font = Agent._cached_font
        id_text = font.render(f'#{self.id}', True, CFG.COLOR_TEXT_PRIMARY)
        screen.blit(id_text, (x - 10, y - self.radius - 24))
        
        # State label (autopilot mode)
        state_color = {
            'IDLE': CFG.COLOR_TEXT_SECONDARY,
            'WANDER': (100, 180, 255),
            'SEEK': (255, 200, 50),
            'ENGAGE': (255, 80, 80),
            'FIRE': (255, 40, 40),
            'SCAN': (180, 130, 255),
        }.get(self.state_label, CFG.COLOR_TEXT_SECONDARY)
        state_text = font.render(self.state_label, True, state_color)
        screen.blit(state_text, (x - 16, y + self.radius + 4))
        
        if len(self.trail) > 1:
            trail_points = [(p.x + camera_offset.x, p.y + camera_offset.y) for p in self.trail[-10:]]
            if len(trail_points) > 1:
                pygame.draw.lines(screen, base_color, False, trail_points, 2)

# ============================================================================
# PART 8: WORLD MODEL
# ============================================================================

class WorldModel:
    def __init__(
        self,
        width: int,
        height: int,
        config: SimulationConfig = CFG
    ):
        self.width = width
        self.height = height
        self.config = config
        
        self.grid = OccupancyGrid(
            width - CFG.WORLD_MARGIN * 2,
            height - CFG.WORLD_MARGIN * 2,
            config.GRID_RESOLUTION,
            config.OCCUPANCY_DECAY_RATE
        )
        
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.agents: List[Agent] = []
        self.obstacles: List[Obstacle] = []
        
        self.spatial_hash = SpatialHash(50.0)
        self.pathfinder = AStarPathfinder(self.grid)
        self.frame = 0
    
    def initialize_obstacles(self, obstacle_data: List[Dict]) -> None:
        for i, obs in enumerate(obstacle_data):
            self.obstacles.append(Obstacle(
                id=i,
                position=obs['position'].copy(),
                radius=obs['radius']
            ))
            self.grid.update_circle(obs['position'], obs['radius'], 0.95, is_occupied=True)
    
    def register_agent(self, agent: Agent) -> None:
        self.agents.append(agent)
    
    def update(self, dt: float) -> None:
        self.frame += 1
        
        self.spatial_hash.clear()
        for agent in self.agents:
            if agent.alive:
                self.spatial_hash.insert(agent, agent.position)
        
        self.grid.decay()
        
        for agent in self.agents:
            if not agent.alive:
                continue
            
            # Reuse cached sensor data if available (set by GameEngine.update)
            sensor_data = getattr(agent, '_cached_sensor', None)
            if sensor_data is None:
                sensor_data = agent.sense(self.agents, self.obstacles, self.spatial_hash)
            
            for enemy in sensor_data['enemies']:
                eid = enemy['id']
                
                if eid not in self.tracked_objects:
                    self.tracked_objects[eid] = TrackedObject(
                        id=eid,
                        position=enemy['position'].copy(),
                        velocity=enemy['velocity'].copy(),
                        kalman_filter=KalmanFilter2D(
                            self.config.KALMAN_PROCESS_VAR,
                            self.config.KALMAN_MEASURE_VAR,
                            enemy['position']
                        ),
                        last_seen_frame=self.frame,
                        confidence=1.0,
                        team=enemy['team'],
                        is_alive=True
                    )
                else:
                    tracked = self.tracked_objects[eid]
                    tracked.kalman_filter.update(enemy['position'], dt)
                    tracked.position = tracked.kalman_filter.get_position()
                    tracked.velocity = tracked.kalman_filter.get_velocity()
                    tracked.last_seen_frame = self.frame
                    tracked.confidence = min(1.0, tracked.confidence + 0.1)
            
            for obs in sensor_data['obstacles']:
                self.grid.update_circle(obs['position'], obs['radius'], 0.8, is_occupied=True)
        
        stale_ids = [
            tid for tid, tracked in self.tracked_objects.items()
            if self.frame - tracked.last_seen_frame > self.config.PREDICTION_HORIZON_FRAMES
        ]
        for tid in stale_ids:
            del self.tracked_objects[tid]
    
    def get_tracked_object(self, object_id: int) -> Optional[TrackedObject]:
        return self.tracked_objects.get(object_id)

# ============================================================================
# PART 9: GAME ENGINE
# ============================================================================

class GameEngine:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("TESLA SWARM AGI - FIXED VERSION")
        
        if CFG.FULLSCREEN:
            info = pygame.display.Info()
            self.screen = pygame.display.set_mode(
                (info.current_w, info.current_h),
                pygame.FULLSCREEN
            )
            CFG.SCREEN_W = info.current_w
            CFG.SCREEN_H = info.current_h
        else:
            self.screen = pygame.display.set_mode((CFG.SCREEN_W, CFG.SCREEN_H))
        
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('consolas', 14)
        self.font_medium = pygame.font.SysFont('consolas', 18)
        self.font_large = pygame.font.SysFont('consolas', 24, bold=True)
        
        self.running = True
        self.paused = False
        self.debug_mode = False
        self.show_world_model = False
        self.show_tactical = True
        
        self.world_model = WorldModel(CFG.SCREEN_W, CFG.SCREEN_H)
        self._initialize_world()
        
        self.projectiles: List[Projectile] = []
        self.next_projectile_id = 0
        self.stats = {
            'total_shots': 0,
            'total_hits': 0,
            'total_kills': 0,
            'frame': 0
        }
        
        self.camera = Vec2()
    
    def _initialize_world(self) -> None:
        obstacle_data = []
        for i in range(CFG.OBSTACLE_COUNT):
            x = random.randint(
                CFG.WORLD_MARGIN + 100,
                CFG.SCREEN_W - CFG.WORLD_MARGIN - 100
            )
            y = random.randint(
                CFG.WORLD_MARGIN + 100,
                CFG.SCREEN_H - CFG.WORLD_MARGIN - 100
            )
            r = random.randint(CFG.OBSTACLE_MIN_RADIUS, CFG.OBSTACLE_MAX_RADIUS)
            obstacle_data.append({
                'position': Vec2(x, y),
                'radius': r
            })
        
        self.world_model.initialize_obstacles(obstacle_data)
        
        for i in range(CFG.AGENT_COUNT_PER_TEAM * 2):
            team = i % 2
            if team == 0:
                x = random.randint(CFG.WORLD_MARGIN + 50, CFG.SCREEN_W // 2 - 50)
            else:
                x = random.randint(CFG.SCREEN_W // 2 + 50, CFG.SCREEN_W - CFG.WORLD_MARGIN - 50)
            y = random.randint(
                CFG.WORLD_MARGIN + 50,
                CFG.SCREEN_H - CFG.WORLD_MARGIN - 50
            )
            
            agent = Agent(
                agent_id=i,
                team=team,
                position=Vec2(x, y),
                world_model=self.world_model
            )
            self.world_model.register_agent(agent)
    
    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset_game()
                elif event.key == pygame.K_d:
                    self.debug_mode = not self.debug_mode
                elif event.key == pygame.K_w:
                    self.show_world_model = not self.show_world_model
                elif event.key == pygame.K_t:
                    self.show_tactical = not self.show_tactical
    
    def _reset_game(self) -> None:
        self.world_model = WorldModel(CFG.SCREEN_W, CFG.SCREEN_H)
        self._initialize_world()
        self.projectiles = []
        self.next_projectile_id = 0
        self.stats = {'total_shots': 0, 'total_hits': 0, 'total_kills': 0, 'frame': 0}
    
    def update(self, dt: float) -> None:
        if self.paused:
            return
        
        self.stats['frame'] += 1
        self.world_model.update(dt)
        
        combat_actions = []
        for agent in self.world_model.agents:
            if not agent.alive:
                continue
            
            sensor_data = agent.sense(
                self.world_model.agents,
                self.world_model.obstacles,
                self.world_model.spatial_hash
            )
            # Cache sensor data so WorldModel.update() can reuse it
            agent._cached_sensor = sensor_data
            
            action = agent.decide(sensor_data, dt)
            combat_action = agent.act(action, dt, self.world_model.agents, self.world_model.obstacles)
            
            if combat_action:
                combat_actions.append(combat_action)
        
        for action in combat_actions:
            projectile = Projectile(
                id=self.next_projectile_id,
                position=action['position'].copy(),
                velocity=action['direction'] * CFG.PROJECTILE_SPEED,
                owner_id=action['owner_id'],
                team=action['team'],
                damage=CFG.PROJECTILE_DAMAGE,
                lifetime=CFG.PROJECTILE_LIFETIME,
                max_lifetime=CFG.PROJECTILE_LIFETIME
            )
            self.projectiles.append(projectile)
            self.next_projectile_id += 1
            self.stats['total_shots'] += 1
        
        for projectile in self.projectiles[:]:
            # Sub-step projectile movement to prevent tunneling through agents
            proj_speed = projectile.velocity.mag()
            step_dist = proj_speed * dt * 60
            num_steps = max(1, int(step_dist / 10) + 1)
            sub_dt = dt / num_steps
            
            hit = False
            for _step in range(num_steps):
                projectile.update(sub_dt)
                
                if projectile.is_out_of_bounds((
                    CFG.WORLD_MARGIN,
                    CFG.WORLD_MARGIN,
                    CFG.SCREEN_W - CFG.WORLD_MARGIN * 2,
                    CFG.SCREEN_H - CFG.WORLD_MARGIN * 2
                )):
                    projectile.alive = False
                    hit = True
                    break
                
                for agent in self.world_model.agents:
                    if not agent.alive:
                        continue
                    if agent.team == projectile.team:
                        continue
                    if agent.id == projectile.owner_id:
                        continue
                    
                    if projectile.position.dist(agent.position) < agent.radius + 5:
                        projectile.alive = False
                        hit = True
                        
                        damage = projectile.damage
                        if random.random() < CFG.CRITICAL_HIT_CHANCE:
                            damage *= CFG.CRITICAL_MULTIPLIER
                        
                        died = agent.take_damage(damage)
                        self.stats['total_hits'] += 1
                        
                        if died:
                            self.stats['total_kills'] += 1
                            for shooter in self.world_model.agents:
                                if shooter.id == projectile.owner_id:
                                    shooter.add_kill()
                                    break
                        break
                if hit:
                    break
        
        self.projectiles = [p for p in self.projectiles if p.alive]
    
    def render(self) -> None:
        self.screen.fill(CFG.COLOR_BG)
        
        pygame.draw.rect(
            self.screen,
            CFG.COLOR_WORLD,
            (CFG.WORLD_MARGIN, CFG.WORLD_MARGIN,
             CFG.SCREEN_W - CFG.WORLD_MARGIN * 2,
             CFG.SCREEN_H - CFG.WORLD_MARGIN * 2)
        )
        
        for x in range(CFG.WORLD_MARGIN, CFG.SCREEN_W - CFG.WORLD_MARGIN, CFG.GRID_RESOLUTION):
            pygame.draw.line(
                self.screen,
                CFG.COLOR_GRID,
                (x, CFG.WORLD_MARGIN),
                (x, CFG.SCREEN_H - CFG.WORLD_MARGIN),
                1
            )
        for y in range(CFG.WORLD_MARGIN, CFG.SCREEN_H - CFG.WORLD_MARGIN, CFG.GRID_RESOLUTION):
            pygame.draw.line(
                self.screen,
                CFG.COLOR_GRID,
                (CFG.WORLD_MARGIN, y),
                (CFG.SCREEN_W - CFG.WORLD_MARGIN, y),
                1
            )
        
        for obs in self.world_model.obstacles:
            pygame.draw.circle(
                self.screen,
                CFG.COLOR_OBSTACLE,
                (int(obs.position.x), int(obs.position.y)),
                int(obs.radius)
            )
            pygame.draw.circle(
                self.screen,
                CFG.COLOR_OBSTACLE_BORDER,
                (int(obs.position.x), int(obs.position.y)),
                int(obs.radius),
                2
            )
        
        if self.show_world_model:
            for tracked in self.world_model.tracked_objects.values():
                pos = tracked.position
                pygame.draw.circle(self.screen, (255, 0, 255), (int(pos.x), int(pos.y)), 8, 2)
                
                pred = tracked.kalman_filter.predict_future(0.5)
                pygame.draw.line(self.screen, (255, 0, 255), (int(pos.x), int(pos.y)), (int(pred.x), int(pred.y)), 2)
        
        for agent in self.world_model.agents:
            agent.draw(self.screen, self.camera)
        
        # Visualization overlays: target lines, swarm connections, fire range arcs
        if self.show_tactical:
            self._draw_tactical_overlay()
        
        for proj in self.projectiles:
            pos = proj.position + self.camera
            
            for i, trail_pos in enumerate(proj.trail):
                tpos = trail_pos + self.camera
                pygame.draw.circle(
                    self.screen,
                    CFG.COLOR_PROJECTILE_TRAIL,
                    (int(tpos.x), int(tpos.y)),
                    3
                )
            
            pygame.draw.circle(
                self.screen,
                CFG.COLOR_PROJECTILE,
                (int(pos.x), int(pos.y)),
                5
            )
        
        self._draw_ui()
        
        if self.paused:
            self._draw_pause_overlay()
        
        pygame.display.flip()
    
    def _draw_ui(self) -> None:
        sidebar_x = CFG.SCREEN_W - 280
        y = 20
        
        title = self.font_large.render("SWARM AGI", True, CFG.COLOR_TEXT_ACCENT)
        self.screen.blit(title, (sidebar_x, y))
        y += 40
        
        stats = [
            (f"FPS: {int(self.clock.get_fps())}", CFG.COLOR_TEXT_ACCENT),
            (f"Frame: {self.stats['frame']}", CFG.COLOR_TEXT_PRIMARY),
            (f"Alive: {self._count_alive()}/{len(self.world_model.agents)}", CFG.COLOR_TEXT_PRIMARY),
            (f"Shots: {self.stats['total_shots']}", CFG.COLOR_TEXT_PRIMARY),
            (f"Hits: {self.stats['total_hits']}", CFG.COLOR_TEXT_PRIMARY),
            (f"Kills: {self.stats['total_kills']}", CFG.COLOR_TEXT_ACCENT),
        ]
        
        if self.stats['total_shots'] > 0:
            accuracy = self.stats['total_hits'] / self.stats['total_shots'] * 100
            stats.append((f"Accuracy: {accuracy:.1f}%", 
                         CFG.COLOR_HP_HIGH if accuracy > 40 else CFG.COLOR_HP_MEDIUM))
        
        for text, color in stats:
            surf = self.font_medium.render(text, True, color)
            self.screen.blit(surf, (sidebar_x, y))
            y += 25
        
        y += 20
        
        team0_alive = self._count_alive(0)
        team1_alive = self._count_alive(1)
        
        pygame.draw.rect(self.screen, (30, 30, 40), (sidebar_x, y, 260, 60))
        
        team0_text = self.font_medium.render(f"TEAM 0: {team0_alive}", True, CFG.COLOR_AGENT_TEAM_0)
        team1_text = self.font_medium.render(f"TEAM 1: {team1_alive}", True, CFG.COLOR_AGENT_TEAM_1)
        self.screen.blit(team0_text, (sidebar_x + 10, y + 10))
        self.screen.blit(team1_text, (sidebar_x + 10, y + 35))
        
        y += 80
        
        controls = [
            "SPACE - Pause",
            "R - Reset",
            "D - Debug",
            "W - World Model",
            "T - Tactical View",
            "ESC - Exit"
        ]
        
        for ctrl in controls:
            surf = self.font_small.render(ctrl, True, CFG.COLOR_TEXT_SECONDARY)
            self.screen.blit(surf, (sidebar_x, y))
            y += 20
        
        if self.debug_mode:
            y = CFG.SCREEN_H - 150
            debug_info = [
                f"Agents: {len(self.world_model.agents)}",
                f"Tracked: {len(self.world_model.tracked_objects)}",
                f"Projectiles: {len(self.projectiles)}",
                f"Grid: {self.world_model.grid.width_cells}x{self.world_model.grid.height_cells}"
            ]
            
            for info in debug_info:
                surf = self.font_small.render(info, True, CFG.COLOR_DEBUG)
                self.screen.blit(surf, (20, y))
                y += 18
    
    def _count_alive(self, team: Optional[int] = None) -> int:
        count = 0
        for agent in self.world_model.agents:
            if agent.alive:
                if team is None or agent.team == team:
                    count += 1
        return count
    
    def _draw_tactical_overlay(self) -> None:
        """Draw target selection lines, swarm connections, and engagement arcs."""
        for agent in self.world_model.agents:
            if not agent.alive:
                continue
            ax = int(agent.position.x + self.camera.x)
            ay = int(agent.position.y + self.camera.y)
            team_color = CFG.COLOR_AGENT_TEAM_0 if agent.team == 0 else CFG.COLOR_AGENT_TEAM_1
            
            # Target selection line: draw line from agent to its current target
            if agent.current_target_id is not None:
                for other in self.world_model.agents:
                    if other.id == agent.current_target_id and other.alive:
                        tx = int(other.position.x + self.camera.x)
                        ty = int(other.position.y + self.camera.y)
                        pygame.draw.line(self.screen, (255, 60, 60), (ax, ay), (tx, ty), 1)
                        # Target reticle
                        pygame.draw.circle(self.screen, (255, 60, 60), (tx, ty), 12, 1)
                        pygame.draw.line(self.screen, (255, 60, 60), (tx - 6, ty), (tx + 6, ty), 1)
                        pygame.draw.line(self.screen, (255, 60, 60), (tx, ty - 6), (tx, ty + 6), 1)
                        break
            
            # Swarm connections: thin lines between nearby friendly agents
            for other in self.world_model.agents:
                if other.id <= agent.id or other.team != agent.team or not other.alive:
                    continue
                dist = agent.position.dist(other.position)
                if dist < CFG.SWARM_COHESION_DIST:
                    ox = int(other.position.x + self.camera.x)
                    oy = int(other.position.y + self.camera.y)
                    alpha_factor = 1.0 - (dist / CFG.SWARM_COHESION_DIST)
                    link_color = (
                        max(0, min(255, int(team_color[0] * 0.3 * alpha_factor))),
                        max(0, min(255, int(team_color[1] * 0.3 * alpha_factor))),
                        max(0, min(255, int(team_color[2] * 0.3 * alpha_factor))),
                    )
                    pygame.draw.line(self.screen, link_color, (ax, ay), (ox, oy), 1)
            
            # Fire range indicator (simple unfilled circle — no SRCALPHA surface)
            if agent.state_label == 'ENGAGE':
                pygame.draw.circle(
                    self.screen, (80, 20, 20),
                    (ax, ay), int(agent.fire_range), 1
                )
    
    def _draw_pause_overlay(self) -> None:
        overlay = pygame.Surface((CFG.SCREEN_W, CFG.SCREEN_H))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))
        
        text = self.font_large.render("PAUSED", True, CFG.COLOR_TEXT_PRIMARY)
        rect = text.get_rect(center=(CFG.SCREEN_W // 2, CFG.SCREEN_H // 2))
        self.screen.blit(text, rect)
    
    def run(self) -> None:
        while self.running:
            if CFG.FPS_UNLOCK:
                self.clock.tick(0)
                dt = 1.0 / 60.0
            else:
                dt = self.clock.tick(CFG.FPS) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.render()
        
        pygame.quit()
        sys.exit()

# ============================================================================
# PART 10: ENTRY POINT
# ============================================================================

def main():
    print("=" * 60)
    print("TESLA SWARM AGI - FIXED VERSION")
    print("=" * 60)
    
    engine = GameEngine()
    engine.run()

if __name__ == "__main__":
    main()
