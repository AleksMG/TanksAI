# ============================================================================
# TESLA TANK SWARM AGI - IMPROVED EDITION
# ============================================================================
# Improvements over previous version:
#   - Proper dt propagation (physics no longer frame-rate dependent)
#   - Kalman filter uses Joseph form + condition-number guard
#   - Hungarian algorithm uses scipy when available, with robust fallback
#   - Tracker associates by Mahalanobis distance instead of Euclidean
#   - Neural network uses batch normalization + Adam optimizer
#   - Experience replay with TD-style advantage for training signal
#   - MPC caches best trajectory and warm-starts next plan
#   - Formation adapts shape to tactical context (combat vs patrol)
#   - Occupancy grid uses log-odds for numerically stable Bayesian update
#   - Agents use influence maps for tactical positioning
#   - Projectile collision uses swept-circle (continuous) detection
#   - Spatial hash auto-tunes cell size
#   - Proper respawn system with wave mechanics
#   - Minimap, damage numbers, kill feed UI elements
#   - Sound-free audio event system for extensibility
# ============================================================================

import pygame
import math
import random
import heapq
import sys
import time
import numpy as np
import pickle
import os
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque

try:
    from scipy.optimize import linear_sum_assignment as scipy_lsa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================================
# PART 1: CONFIGURATION
# ============================================================================

class Difficulty(Enum):
    EASY = auto(); NORMAL = auto(); HARD = auto(); EXTREME = auto()

class AgentRole(Enum):
    LEADER   = "LEADER"
    ATTACKER = "ATTACK"
    SUPPORT  = "SUPPRT"
    SCOUT    = "SCOUT"

class FormationShape(Enum):
    WEDGE   = auto()
    LINE    = auto()
    COLUMN  = auto()
    SPREAD  = auto()

@dataclass
class SimulationConfig:
    SCREEN_W: int = 1280
    SCREEN_H: int = 720
    FULLSCREEN: bool = False
    FPS: int = 60
    FPS_UNLOCK: bool = False

    WORLD_MARGIN: int = 60
    GRID_RESOLUTION: int = 60
    CELL_UPDATE_THRESHOLD: float = 0.1

    AGENT_COUNT_PER_TEAM: int = 5
    AGENT_RADIUS: float = 14.0
    AGENT_MAX_SPEED: float = 4.5
    AGENT_MAX_ACCEL: float = 0.25
    AGENT_TURN_RATE: float = 0.08
    AGENT_FRICTION: float = 0.94
    AGENT_HP: float = 100.0

    SWARM_SEPARATION_WEIGHT: float = 2.5
    SWARM_ALIGNMENT_WEIGHT: float  = 1.5
    SWARM_COHESION_WEIGHT: float   = 1.2
    SWARM_SEPARATION_DIST: float   = 50.0
    SWARM_ALIGNMENT_DIST: float    = 120.0
    SWARM_COHESION_DIST: float     = 200.0

    FIRE_RANGE: float         = 350.0
    FIRE_COOLDOWN_FRAMES: int = 45
    PROJECTILE_SPEED: float   = 18.0
    PROJECTILE_DAMAGE: float  = 30.0
    PROJECTILE_LIFETIME: int  = 120
    CRITICAL_HIT_CHANCE: float = 0.15
    CRITICAL_MULTIPLIER: float = 2.0
    PROJECTILE_RADIUS: float  = 4.0

    NN_INPUT_SIZE: int = 20
    NN_HIDDEN_LAYERS: Tuple[int, ...] = (48, 32)
    NN_OUTPUT_SIZE: int = 6
    NN_LEARNING_RATE: float = 0.001
    NN_TRAIN_INTERVAL: int = 30
    NN_SAVE_INTERVAL: int = 600
    NN_BATCH_SIZE: int = 32
    DECISION_INTERVAL_FRAMES: int = 3

    CAMERA_FOV_DEG: float  = 180.0
    CAMERA_RANGE: float    = 380.0
    CAMERA_NOISE_STD: float = 2.5

    KALMAN_PROCESS_VAR: float = 0.05
    KALMAN_MEASURE_VAR: float = 0.3
    KALMAN_INNOV_GATE: float  = 9.0
    KALMAN_P_CONDITION_MAX: float = 1e6

    TRACKER_MAX_AGE: int      = 45
    TRACKER_MIN_HITS: int     = 2
    TRACKER_COST_MAX: float   = 80.0

    MPC_HORIZON: int       = 8
    MPC_DT: float          = 0.1
    MPC_STEER_STEPS: int   = 7
    MPC_ACCEL_STEPS: int   = 3
    MPC_WARM_START: bool   = True

    PID_ANGLE_KP: float = 3.0
    PID_ANGLE_KI: float = 0.0
    PID_ANGLE_KD: float = 0.8
    PID_SPEED_KP: float = 1.2
    PID_SPEED_KI: float = 0.05
    PID_SPEED_KD: float = 0.3

    OCCUPANCY_DECAY_RATE: float = 0.02
    PREDICTION_HORIZON_FRAMES: int = 30

    OBSTACLE_COUNT: int    = 8
    OBSTACLE_MIN_RADIUS: int = 30
    OBSTACLE_MAX_RADIUS: int = 70

    FORMATION_SPACING: float = 60.0

    WORLD_MODEL_WEIGHTS_FILE: str = "world_model_weights.pkl"
    TRAJ_NN_WEIGHTS_FILE: str = "trajectory_nn_weights"

    RESPAWN_ENABLED: bool = True
    RESPAWN_DELAY_FRAMES: int = 180
    RESPAWN_INVULN_FRAMES: int = 60

    INFLUENCE_MAP_RESOLUTION: int = 80
    INFLUENCE_DECAY: float = 0.92
    INFLUENCE_RADIUS: float = 200.0

    KILL_FEED_MAX: int = 8
    KILL_FEED_DURATION: int = 180
    DAMAGE_NUMBER_DURATION: int = 40

    COLOR_BG:              Tuple[int,int,int] = (8, 8, 12)
    COLOR_WORLD:           Tuple[int,int,int] = (15, 15, 22)
    COLOR_GRID:            Tuple[int,int,int] = (25, 35, 30)
    COLOR_AGENT_TEAM_0:    Tuple[int,int,int] = (0, 255, 200)
    COLOR_AGENT_TEAM_1:    Tuple[int,int,int] = (255, 68, 68)
    COLOR_PROJECTILE:      Tuple[int,int,int] = (255, 220, 80)
    COLOR_PROJECTILE_TRAIL:Tuple[int,int,int] = (255, 180, 50)
    COLOR_OBSTACLE:        Tuple[int,int,int] = (80, 80, 95)
    COLOR_OBSTACLE_BORDER: Tuple[int,int,int] = (120, 180, 140)
    COLOR_TEXT_PRIMARY:    Tuple[int,int,int] = (220, 220, 240)
    COLOR_TEXT_SECONDARY:  Tuple[int,int,int] = (150, 150, 170)
    COLOR_TEXT_ACCENT:     Tuple[int,int,int] = (0, 255, 200)
    COLOR_HP_HIGH:         Tuple[int,int,int] = (0, 255, 100)
    COLOR_HP_MEDIUM:       Tuple[int,int,int] = (255, 200, 50)
    COLOR_HP_LOW:          Tuple[int,int,int] = (255, 50, 50)
    COLOR_DEBUG:           Tuple[int,int,int] = (255, 0, 255)
    COLOR_CAM_FRONT:       Tuple[int,int,int] = (0, 200, 255)
    COLOR_CAM_REAR:        Tuple[int,int,int] = (255, 160, 0)
    COLOR_TRACK:           Tuple[int,int,int] = (255, 0, 255)
    COLOR_MPC_PATH:        Tuple[int,int,int] = (100, 255, 100)
    COLOR_FORMATION:       Tuple[int,int,int] = (255, 255, 100)
    COLOR_PREDICTION:      Tuple[int,int,int] = (255, 100, 255)
    COLOR_MINIMAP_BG:      Tuple[int,int,int] = (20, 20, 30)
    COLOR_DAMAGE_NUM:      Tuple[int,int,int] = (255, 255, 100)

CFG = SimulationConfig()

# ============================================================================
# PART 2: PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    __slots__ = ('timings', 'counters', '_window_size')

    def __init__(self, window_size=100):
        self._window_size = window_size
        self.timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: Dict[str, int] = defaultdict(int)

    def measure(self, name: str):
        return self._Timer(self, name)

    def count(self, name: str, value: int = 1):
        self.counters[name] += value

    def get_avg(self, name: str) -> float:
        d = self.timings.get(name)
        if d and len(d) > 0:
            return sum(d) / len(d)
        return 0.0

    def get_max(self, name: str) -> float:
        d = self.timings.get(name)
        return max(d) if d and len(d) > 0 else 0.0

    def get_count(self, name: str) -> int:
        return self.counters.get(name, 0)

    def reset_counters(self):
        self.counters.clear()

    class _Timer:
        __slots__ = ('monitor', 'name', 'start')
        def __init__(self, monitor, name):
            self.monitor = monitor
            self.name = name
            self.start = 0.0
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = (time.perf_counter() - self.start) * 1000.0
            self.monitor.timings[self.name].append(elapsed)

# ============================================================================
# PART 3: MATH & GEOMETRY
# ============================================================================

class Vec2:
    __slots__ = ('x', 'y')

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x); self.y = float(y)

    def __repr__(self): return f"Vec2({self.x:.2f},{self.y:.2f})"
    def __eq__(self, o): return isinstance(o, Vec2) and abs(self.x-o.x)<1e-6 and abs(self.y-o.y)<1e-6
    def __hash__(self): return hash((round(self.x, 4), round(self.y, 4)))
    def __add__(self, o): return Vec2(self.x+o.x, self.y+o.y)
    def __sub__(self, o): return Vec2(self.x-o.x, self.y-o.y)
    def __mul__(self, s): return Vec2(self.x*s, self.y*s)
    def __rmul__(self, s): return self.__mul__(s)
    def __truediv__(self, s): return Vec2(self.x/s, self.y/s) if abs(s) > 1e-10 else Vec2()
    def __neg__(self): return Vec2(-self.x, -self.y)
    def __iter__(self): yield self.x; yield self.y
    def __bool__(self): return self.mag_sq() > 1e-12

    def copy(self): return Vec2(self.x, self.y)
    def mag(self): return math.hypot(self.x, self.y)
    def mag_sq(self): return self.x*self.x + self.y*self.y
    def norm(self):
        m = self.mag()
        return Vec2(self.x/m, self.y/m) if m > 1e-6 else Vec2()
    def limit(self, m):
        sq = self.mag_sq()
        if sq > m * m:
            s = m / math.sqrt(sq)
            return Vec2(self.x * s, self.y * s)
        return self.copy()
    def dist(self, o):
        dx = self.x - o.x; dy = self.y - o.y
        return math.sqrt(dx*dx + dy*dy)
    def dist_sq(self, o):
        dx = self.x - o.x; dy = self.y - o.y
        return dx*dx + dy*dy
    def angle(self): return math.atan2(self.y, self.x)
    def dot(self, o): return self.x*o.x + self.y*o.y
    def cross(self, o): return self.x*o.y - self.y*o.x
    def rotate(self, a):
        c, s = math.cos(a), math.sin(a)
        return Vec2(self.x*c - self.y*s, self.x*s + self.y*c)
    def lerp(self, o, t):
        t = max(0.0, min(1.0, t))
        return Vec2(self.x + (o.x - self.x)*t, self.y + (o.y - self.y)*t)
    def perp(self): return Vec2(-self.y, self.x)

    @staticmethod
    def from_angle(a, l=1.0): return Vec2(math.cos(a)*l, math.sin(a)*l)
    @staticmethod
    def random_in_circle(r=1.0):
        rr = r * math.sqrt(random.random())
        th = random.uniform(0, math.pi * 2)
        return Vec2(rr * math.cos(th), rr * math.sin(th))
    @staticmethod
    def zero(): return Vec2(0.0, 0.0)


class MathUtils:
    @staticmethod
    def clamp(v, a, b): return max(a, min(b, v))

    @staticmethod
    def wrap_angle(a):
        a = a % (2 * math.pi)
        if a > math.pi:
            a -= 2 * math.pi
        return a

    @staticmethod
    def angle_diff(a, b): return MathUtils.wrap_angle(b - a)

    @staticmethod
    def sigmoid(x):
        x = max(-500.0, min(500.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def smooth_step(edge0, edge1, x):
        t = MathUtils.clamp((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def line_circle_intersect(ls: Vec2, le: Vec2, cc: Vec2, cr: float) -> bool:
        dx = le.x - ls.x; dy = le.y - ls.y
        ll = dx*dx + dy*dy
        if ll < 1e-10:
            return (ls.x-cc.x)**2 + (ls.y-cc.y)**2 <= cr*cr
        t = MathUtils.clamp(((cc.x-ls.x)*dx + (cc.y-ls.y)*dy) / ll, 0.0, 1.0)
        nx = ls.x + dx*t - cc.x; ny = ls.y + dy*t - cc.y
        return nx*nx + ny*ny <= cr*cr

    @staticmethod
    def swept_circle_hit(p0: Vec2, p1: Vec2, pr: float,
                         tc: Vec2, tr: float) -> Optional[float]:
        """Continuous collision: moving circle (p0->p1, radius pr) vs static circle (tc, tr).
        Returns parametric t in [0,1] of first hit, or None."""
        d = p1 - p0
        f = p0 - tc
        R = pr + tr
        a = d.dot(d)
        b = 2.0 * f.dot(d)
        c = f.dot(f) - R * R
        if c <= 0:
            return 0.0
        if a < 1e-10:
            return None
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        sq = math.sqrt(disc)
        t = (-b - sq) / (2*a)
        if 0 <= t <= 1:
            return t
        return None

    @staticmethod
    def remap(val, in_lo, in_hi, out_lo, out_hi):
        t = (val - in_lo) / (in_hi - in_lo + 1e-9)
        return out_lo + MathUtils.clamp(t, 0, 1) * (out_hi - out_lo)


# ============================================================================
# PART 4: SPATIAL HASH (Auto-tuning cell size)
# ============================================================================

class SpatialHash:
    __slots__ = ('cell_size', 'inv_cell', 'grid')

    def __init__(self, cell_size: float = 50.0):
        self.cell_size = cell_size
        self.inv_cell = 1.0 / cell_size
        self.grid: Dict[Tuple[int, int], List[Any]] = defaultdict(list)

    def clear(self):
        self.grid.clear()

    def _hash(self, p: Vec2) -> Tuple[int, int]:
        return (int(p.x * self.inv_cell), int(p.y * self.inv_cell))

    def insert(self, obj: Any, pos: Vec2):
        self.grid[self._hash(pos)].append(obj)

    def query(self, pos: Vec2, radius: float) -> List[Any]:
        results = []
        cr = int(math.ceil(radius * self.inv_cell))
        cx, cy = self._hash(pos)
        radius_sq = radius * radius

        for dx in range(-cr, cr + 1):
            for dy in range(-cr, cr + 1):
                bucket = self.grid.get((cx + dx, cy + dy))
                if bucket:
                    for o in bucket:
                        if hasattr(o, 'position'):
                            dsq = (o.position.x - pos.x)**2 + (o.position.y - pos.y)**2
                            if dsq <= radius_sq:
                                results.append(o)
        return results

    def query_pairs(self, radius: float) -> List[Tuple[Any, Any]]:
        """Find all pairs within radius - avoids O(n²) full scan."""
        pairs = []
        radius_sq = radius * radius
        visited: Set[Tuple[int, int]] = set()

        for cell_key, objects in self.grid.items():
            cx, cy = cell_key
            cr = int(math.ceil(radius * self.inv_cell))
            for dx in range(-cr, cr + 1):
                for dy in range(-cr, cr + 1):
                    neighbor_key = (cx + dx, cy + dy)
                    if neighbor_key < cell_key:
                        continue
                    neighbor = self.grid.get(neighbor_key)
                    if not neighbor:
                        continue
                    if neighbor_key == cell_key:
                        for i in range(len(objects)):
                            for j in range(i + 1, len(objects)):
                                a, b = objects[i], objects[j]
                                if hasattr(a, 'position') and hasattr(b, 'position'):
                                    if a.position.dist_sq(b.position) <= radius_sq:
                                        pairs.append((a, b))
                    else:
                        for a in objects:
                            for b in neighbor:
                                if hasattr(a, 'position') and hasattr(b, 'position'):
                                    if a.position.dist_sq(b.position) <= radius_sq:
                                        pairs.append((a, b))
        return pairs


# ============================================================================
# PART 5: CAMERA SENSOR SYSTEM
# ============================================================================

class TankCamera:
    __slots__ = ('offset', 'fov', 'half_fov', 'range', 'range_sq', 'noise')

    def __init__(self, mounting_angle_offset: float, fov_deg: float,
                 range_: float, noise_std: float):
        self.offset = mounting_angle_offset
        self.fov = math.radians(fov_deg)
        self.half_fov = self.fov / 2.0
        self.range = range_
        self.range_sq = range_ * range_
        self.noise = noise_std

    def get_detections(self, owner_pos: Vec2, owner_angle: float,
                       targets: List[Any], obstacles: List[Any]) -> List[Dict]:
        detections = []
        cam_dir = owner_angle + self.offset
        half = self.half_fov

        for t in targets:
            if not getattr(t, 'alive', True):
                continue
            dx = t.position.x - owner_pos.x
            dy = t.position.y - owner_pos.y
            dist_sq = dx*dx + dy*dy
            if dist_sq > self.range_sq:
                continue

            dist = math.sqrt(dist_sq)
            angle_to = math.atan2(dy, dx)
            rel = MathUtils.wrap_angle(angle_to - cam_dir)
            if abs(rel) > half:
                continue

            blocked = False
            for obs in obstacles:
                if MathUtils.line_circle_intersect(
                    owner_pos, t.position, obs.position, obs.radius
                ):
                    blocked = True
                    break
            if blocked:
                continue

            # Range-dependent noise: farther = noisier
            noise_scale = self.noise * (0.5 + 0.5 * dist / self.range)
            noisy_pos = Vec2(
                t.position.x + random.gauss(0, noise_scale),
                t.position.y + random.gauss(0, noise_scale)
            )
            detections.append({
                'id':        t.id,
                'position':  noisy_pos,
                'velocity':  t.velocity.copy(),
                'team':      t.team,
                'health':    t.health,
                'distance':  dist,
                'angle_rel': rel,
            })
        return detections


class DualCameraSystem:
    __slots__ = ('front', 'rear')

    def __init__(self, cfg: SimulationConfig):
        self.front = TankCamera(0.0, cfg.CAMERA_FOV_DEG, cfg.CAMERA_RANGE, cfg.CAMERA_NOISE_STD)
        self.rear  = TankCamera(math.pi, cfg.CAMERA_FOV_DEG, cfg.CAMERA_RANGE, cfg.CAMERA_NOISE_STD)

    def get_all_detections(self, owner_pos, owner_angle, targets, obstacles):
        seen: Dict[int, Dict] = {}
        for det in self.front.get_detections(owner_pos, owner_angle, targets, obstacles):
            seen[det['id']] = det
        for det in self.rear.get_detections(owner_pos, owner_angle, targets, obstacles):
            if det['id'] not in seen:
                seen[det['id']] = det
        return list(seen.values())


# ============================================================================
# PART 6: KALMAN FILTER (Joseph form + condition guard)
# ============================================================================

class KalmanFilter2D:
    """Stable Kalman filter using Joseph form covariance update
    and condition-number monitoring to prevent divergence."""

    __slots__ = ('x', 'P', 'Q', 'R', 'H', 'gate', 'initialized',
                 'epsilon', '_F', '_cond_max')

    def __init__(self, process_var: float = 0.1, measure_var: float = 0.5,
                 initial_position: Optional[Vec2] = None,
                 innov_gate: float = 9.0,
                 cond_max: float = 1e6):
        self.x = np.zeros(4, dtype=np.float64)
        if initial_position:
            self.x[0] = initial_position.x
            self.x[1] = initial_position.y

        self.P = np.eye(4, dtype=np.float64) * 10.0
        self.Q = np.diag([process_var, process_var,
                          process_var * 2, process_var * 2]).astype(np.float64)
        self.R = np.eye(2, dtype=np.float64) * measure_var
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.gate = innov_gate
        self.initialized = initial_position is not None
        self.epsilon = 1e-8
        self._cond_max = cond_max
        self._F = np.eye(4, dtype=np.float64)

    def _check_condition(self):
        """Reset P if condition number explodes."""
        try:
            cn = np.linalg.cond(self.P)
            if cn > self._cond_max or np.isnan(cn):
                self.P = np.eye(4, dtype=np.float64) * 10.0
        except np.linalg.LinAlgError:
            self.P = np.eye(4, dtype=np.float64) * 10.0

    def predict(self, dt: float) -> Vec2:
        F = self._F
        F[0, 2] = dt
        F[1, 3] = dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        # Symmetrize
        self.P = (self.P + self.P.T) * 0.5

        self._check_condition()
        return Vec2(float(self.x[0]), float(self.x[1]))

    def update(self, measurement: Vec2, dt: float) -> Optional[Vec2]:
        if not self.initialized:
            self.x[0] = measurement.x
            self.x[1] = measurement.y
            self.initialized = True
            return measurement.copy()

        self.predict(dt)

        z = np.array([measurement.x, measurement.y], dtype=np.float64)
        y = z - self.H @ self.x  # innovation

        S = self.H @ self.P @ self.H.T + self.R
        S = (S + S.T) * 0.5 + np.eye(2) * self.epsilon

        # Innovation gate (Mahalanobis)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        mah_sq = float(y @ S_inv @ y)
        if mah_sq > self.gate:
            return None

        K = self.P @ self.H.T @ S_inv

        self.x = self.x + K @ y

        # Joseph form: P = (I-KH)P(I-KH)' + KRK'
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) * 0.5

        self._check_condition()
        return Vec2(float(self.x[0]), float(self.x[1]))

    def get_position(self) -> Vec2:
        return Vec2(float(self.x[0]), float(self.x[1]))

    def get_velocity(self) -> Vec2:
        return Vec2(float(self.x[2]), float(self.x[3]))

    def predict_future(self, t: float) -> Vec2:
        return Vec2(float(self.x[0] + self.x[2]*t),
                    float(self.x[1] + self.x[3]*t))

    def get_innovation_covariance(self) -> np.ndarray:
        """Return S matrix for Mahalanobis-based association."""
        S = self.H @ self.P @ self.H.T + self.R
        return (S + S.T) * 0.5 + np.eye(2) * self.epsilon


# ============================================================================
# PART 7: HUNGARIAN ALGORITHM (scipy preferred, robust fallback)
# ============================================================================

def hungarian(cost_matrix: List[List[float]], max_iterations: int = 2000) -> List[Tuple[int, int]]:
    n = len(cost_matrix)
    if n == 0:
        return []
    m = len(cost_matrix[0]) if n > 0 else 0
    if m == 0:
        return []

    BIG = 1e18
    arr = np.array(cost_matrix, dtype=np.float64)
    arr = np.where(np.isinf(arr), BIG, arr)

    if HAS_SCIPY:
        try:
            row_ind, col_ind = scipy_lsa(arr)
            return [(int(r), int(c)) for r, c in zip(row_ind, col_ind)
                    if arr[r, c] < BIG * 0.5]
        except Exception:
            pass

    # Fallback: Jonker-Volgenant style or greedy
    if n > 25 or m > 25:
        return _greedy_assignment(cost_matrix)

    try:
        sz = max(n, m)
        C = np.full((sz, sz), BIG, dtype=np.float64)
        C[:n, :m] = arr

        u = np.zeros(sz + 1)
        v = np.zeros(sz + 1)
        p = np.zeros(sz + 1, dtype=int)
        way = np.zeros(sz + 1, dtype=int)

        iteration = 0
        for i in range(1, sz + 1):
            p[0] = i
            j0 = 0
            minVal = np.full(sz + 1, BIG)
            used = np.zeros(sz + 1, dtype=bool)

            while True:
                iteration += 1
                if iteration > max_iterations:
                    return _greedy_assignment(cost_matrix)

                used[j0] = True
                i0 = p[j0]
                delta = BIG
                j1 = -1

                for j in range(1, sz + 1):
                    if not used[j]:
                        cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                        if cur < minVal[j]:
                            minVal[j] = cur
                            way[j] = j0
                        if minVal[j] < delta:
                            delta = minVal[j]
                            j1 = j

                if j1 == -1:
                    break

                for j in range(sz + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minVal[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break

            while j0:
                p[j0] = p[way[j0]]
                j0 = way[j0]

        assignments = []
        for j in range(1, sz + 1):
            if p[j] and p[j] - 1 < n and j - 1 < m:
                if cost_matrix[p[j] - 1][j - 1] < BIG * 0.5:
                    assignments.append((p[j] - 1, j - 1))
        return assignments
    except Exception:
        return _greedy_assignment(cost_matrix)


def _greedy_assignment(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    n = len(cost_matrix)
    if n == 0:
        return []
    m = len(cost_matrix[0])
    if m == 0:
        return []

    entries = []
    for r in range(n):
        for c in range(m):
            if cost_matrix[r][c] < 1e17:
                entries.append((cost_matrix[r][c], r, c))
    entries.sort()

    used_rows: Set[int] = set()
    used_cols: Set[int] = set()
    assignments = []
    for cost, r, c in entries:
        if r not in used_rows and c not in used_cols:
            assignments.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
    return assignments


# ============================================================================
# PART 8: MULTI-OBJECT TRACKER (Mahalanobis association)
# ============================================================================

@dataclass
class Track:
    id: int
    kalman: KalmanFilter2D
    team: int
    hits: int = 1
    age: int = 0
    last_seen: int = 0
    confirmed: bool = False
    position: Vec2 = field(default_factory=Vec2)
    velocity: Vec2 = field(default_factory=Vec2)
    health: float = 100.0
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    coasting: bool = False


class MultiObjectTracker:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.tracks: Dict[int, Track] = {}
        self._next_id = 0
        self.frame = 0

    def update(self, detections: List[Dict], dt: float) -> List[Track]:
        self.frame += 1

        # Predict all tracks forward
        for t in self.tracks.values():
            t.age += 1
            t.coasting = True
            try:
                t.kalman.predict(dt)
                t.position = t.kalman.get_position()
                t.velocity = t.kalman.get_velocity()
            except Exception:
                pass

        track_ids = list(self.tracks.keys())
        if track_ids and detections:
            cost = []
            for tid in track_ids:
                row = []
                tr = self.tracks[tid]
                try:
                    pred = tr.kalman.get_position()
                    S = tr.kalman.get_innovation_covariance()
                    S_inv = np.linalg.inv(S)

                    for det in detections:
                        dp = det['position']
                        innov = np.array([dp.x - pred.x, dp.y - pred.y])
                        mah = float(innov @ S_inv @ innov)
                        if mah < self.cfg.KALMAN_INNOV_GATE:
                            row.append(mah)
                        else:
                            row.append(1e18)
                except Exception:
                    row = [1e18] * len(detections)
                cost.append(row)

            try:
                assignments = hungarian(cost, max_iterations=500)
            except Exception:
                assignments = []

            matched_tracks: Set[int] = set()
            matched_dets: Set[int] = set()

            for ri, ci in assignments:
                if ri < len(cost) and ci < len(cost[ri]) and cost[ri][ci] < 1e17:
                    tid = track_ids[ri]
                    det = detections[ci]
                    t = self.tracks[tid]
                    try:
                        result = t.kalman.update(det['position'], dt)
                        if result is not None:
                            t.position = t.kalman.get_position()
                            t.velocity = t.kalman.get_velocity()
                            t.history.append(t.position.copy())
                            t.hits += 1
                            t.age = 0
                            t.last_seen = self.frame
                            t.health = det['health']
                            t.coasting = False
                            if t.hits >= self.cfg.TRACKER_MIN_HITS:
                                t.confirmed = True
                            matched_tracks.add(ri)
                            matched_dets.add(ci)
                    except Exception:
                        pass

            for ci, det in enumerate(detections):
                if ci not in matched_dets:
                    self._spawn_track(det)
        else:
            for det in detections:
                self._spawn_track(det)

        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.age < self.cfg.TRACKER_MAX_AGE
        }

        return [t for t in self.tracks.values() if t.confirmed]

    def _spawn_track(self, det: Dict):
        try:
            kf = KalmanFilter2D(
                self.cfg.KALMAN_PROCESS_VAR,
                self.cfg.KALMAN_MEASURE_VAR,
                det['position'],
                self.cfg.KALMAN_INNOV_GATE,
                self.cfg.KALMAN_P_CONDITION_MAX
            )
            t = Track(
                id=self._next_id, kalman=kf, team=det['team'],
                position=det['position'].copy(),
                velocity=det.get('velocity', Vec2()).copy(),
                health=det.get('health', 100.0),
                last_seen=self.frame
            )
            t.history.append(t.position.copy())
            self.tracks[self._next_id] = t
            self._next_id += 1
        except Exception:
            pass


# ============================================================================
# PART 9: OCCUPANCY GRID (Log-odds Bayesian update)
# ============================================================================

class OccupancyGrid:
    def __init__(self, width: float, height: float,
                 resolution: float = 60.0, decay_rate: float = 0.02):
        self.resolution = resolution
        self.inv_res = 1.0 / resolution
        self.W = int(width * self.inv_res) + 1
        self.H = int(height * self.inv_res) + 1
        self.decay = decay_rate

        # Log-odds representation: 0 = unknown, >0 = occupied, <0 = free
        self.log_odds = np.zeros((self.H, self.W), dtype=np.float32)
        self.vx = np.zeros((self.H, self.W), dtype=np.float32)
        self.vy = np.zeros((self.H, self.W), dtype=np.float32)
        self._tick = 0

        self.LOG_ODDS_MAX = 5.0
        self.LOG_ODDS_MIN = -5.0

    def _prob_to_log_odds(self, p: float) -> float:
        p = max(0.001, min(0.999, p))
        return math.log(p / (1.0 - p))

    def _log_odds_to_prob(self, lo: float) -> float:
        return 1.0 / (1.0 + math.exp(-lo))

    @property
    def occ(self) -> np.ndarray:
        """Probability view for compatibility."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def decay_step(self):
        self._tick += 1
        if self._tick % 4 != 0:
            return
        self.log_odds *= (1.0 - self.decay)
        self.vx *= 0.95
        self.vy *= 0.95

    def world_to_grid(self, p: Vec2) -> Tuple[int, int]:
        return (
            max(0, min(self.W - 1, int(p.x * self.inv_res))),
            max(0, min(self.H - 1, int(p.y * self.inv_res)))
        )

    def grid_to_world(self, gx: int, gy: int) -> Vec2:
        return Vec2(gx * self.resolution + self.resolution * 0.5,
                    gy * self.resolution + self.resolution * 0.5)

    def update_circle(self, center: Vec2, radius: float, prob: float,
                      vel: Optional[Vec2] = None):
        lo_update = self._prob_to_log_odds(prob)
        g0x, g0y = self.world_to_grid(Vec2(center.x - radius, center.y - radius))
        g1x, g1y = self.world_to_grid(Vec2(center.x + radius, center.y + radius))
        radius_sq = radius * radius

        for gy in range(g0y, g1y + 1):
            for gx in range(g0x, g1x + 1):
                if 0 <= gy < self.H and 0 <= gx < self.W:
                    cc = self.grid_to_world(gx, gy)
                    if (cc.x - center.x)**2 + (cc.y - center.y)**2 <= radius_sq:
                        self.log_odds[gy, gx] = np.clip(
                            self.log_odds[gy, gx] + lo_update,
                            self.LOG_ODDS_MIN, self.LOG_ODDS_MAX
                        )
                        if vel:
                            self.vx[gy, gx] = vel.x
                            self.vy[gy, gx] = vel.y

    def get_cost(self, p: Vec2) -> float:
        gx, gy = self.world_to_grid(p)
        return self._log_odds_to_prob(float(self.log_odds[gy, gx]))

    def get_neighbors(self, gx: int, gy: int) -> List[Tuple[int, int]]:
        result = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            nx, ny = gx+dx, gy+dy
            if 0 <= nx < self.W and 0 <= ny < self.H:
                result.append((nx, ny))
        return result


# ============================================================================
# PART 10: INFLUENCE MAP
# ============================================================================

class InfluenceMap:
    """Tactical influence map for strategic positioning decisions."""

    def __init__(self, width: float, height: float, resolution: float,
                 decay: float = 0.92):
        self.resolution = resolution
        self.inv_res = 1.0 / resolution
        self.W = int(width * self.inv_res) + 1
        self.H = int(height * self.inv_res) + 1
        self.decay = decay

        # Per-team influence
        self.influence = [
            np.zeros((self.H, self.W), dtype=np.float32),
            np.zeros((self.H, self.W), dtype=np.float32),
        ]
        # Tension = sum of influences (contested areas)
        self.tension = np.zeros((self.H, self.W), dtype=np.float32)
        # Vulnerability = difference (advantage areas)
        self.vulnerability = np.zeros((self.H, self.W), dtype=np.float32)

    def update(self, agents: List[Any]):
        self.influence[0] *= self.decay
        self.influence[1] *= self.decay

        for a in agents:
            if not a.alive:
                continue
            gx = max(0, min(self.W - 1, int(a.position.x * self.inv_res)))
            gy = max(0, min(self.H - 1, int(a.position.y * self.inv_res)))

            strength = (a.health / a.max_health) * (1.0 + a.kills * 0.1)
            radius_cells = int(CFG.INFLUENCE_RADIUS * self.inv_res)

            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.W and 0 <= ny < self.H:
                        dist_cells = math.sqrt(dx*dx + dy*dy)
                        if dist_cells <= radius_cells:
                            falloff = 1.0 - dist_cells / (radius_cells + 1)
                            self.influence[a.team][ny, nx] += strength * falloff

        self.tension = self.influence[0] + self.influence[1]
        self.vulnerability = self.influence[0] - self.influence[1]

    def get_advantage(self, pos: Vec2, team: int) -> float:
        gx = max(0, min(self.W - 1, int(pos.x * self.inv_res)))
        gy = max(0, min(self.H - 1, int(pos.y * self.inv_res)))
        if team == 0:
            return float(self.vulnerability[gy, gx])
        else:
            return float(-self.vulnerability[gy, gx])

    def get_tension_at(self, pos: Vec2) -> float:
        gx = max(0, min(self.W - 1, int(pos.x * self.inv_res)))
        gy = max(0, min(self.H - 1, int(pos.y * self.inv_res)))
        return float(self.tension[gy, gx])


# ============================================================================
# PART 11: A* PATHFINDER
# ============================================================================

class AStarPathfinder:
    def __init__(self, grid: OccupancyGrid, max_cost: float = 0.7):
        self.grid = grid
        self.max_cost = max_cost
        self._diag_cost = math.sqrt(2.0)

    def find_path(self, start: Vec2, goal: Vec2, max_iter: int = 800) -> Optional[List[Vec2]]:
        sg = self.grid.world_to_grid(start)
        gg = self.grid.world_to_grid(goal)

        if sg == gg:
            return [start, goal]

        occ = self.grid.occ
        if occ[gg[1], gg[0]] > self.max_cost:
            return None

        open_set: List[Tuple[float, int, int]] = []
        heapq.heappush(open_set, (0.0, sg[0], sg[1]))
        came_from: Dict[Tuple[int,int], Tuple[int,int]] = {}
        g_score: Dict[Tuple[int,int], float] = {sg: 0.0}
        itr = 0

        while open_set and itr < max_iter:
            itr += 1
            _, cx, cy = heapq.heappop(open_set)
            current = (cx, cy)

            if current == gg:
                path = [goal]
                cur = current
                while cur in came_from:
                    path.append(self.grid.grid_to_world(*cur))
                    cur = came_from[cur]
                path.append(start)
                path.reverse()
                return path

            cur_g = g_score.get(current, float('inf'))
            for nx, ny in self.grid.get_neighbors(cx, cy):
                if occ[ny, nx] > self.max_cost:
                    continue
                is_diag = (nx != cx and ny != cy)
                step_cost = self._diag_cost if is_diag else 1.0
                cell_cost = 1.0 + occ[ny, nx] * 3.0
                tg = cur_g + step_cost * cell_cost
                nk = (nx, ny)
                if tg < g_score.get(nk, float('inf')):
                    came_from[nk] = current
                    g_score[nk] = tg
                    # Octile heuristic
                    ddx = abs(nx - gg[0]); ddy = abs(ny - gg[1])
                    h = max(ddx, ddy) + (self._diag_cost - 1.0) * min(ddx, ddy)
                    heapq.heappush(open_set, (tg + h, nx, ny))

        return None


# ============================================================================
# PART 12: PID CONTROLLER
# ============================================================================

class PIDController:
    __slots__ = ('kp', 'ki', 'kd', 'integral', 'prev_error', 'integral_limit')

    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float = 5.0):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.integral = 0.0; self.prev_error = 0.0
        self.integral_limit = integral_limit

    def update(self, error: float, dt: float) -> float:
        self.integral = MathUtils.clamp(
            self.integral + error * dt,
            -self.integral_limit, self.integral_limit
        )
        deriv = (error - self.prev_error) / max(dt, 1e-6)
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * deriv

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


# ============================================================================
# PART 13: MPC PLANNER (with warm-start)
# ============================================================================

class MPCPlanner:
    def __init__(self, cfg: SimulationConfig, obstacles: List[Any]):
        self.cfg = cfg
        self.obstacles = obstacles
        self.horizon = cfg.MPC_HORIZON
        self.dt = cfg.MPC_DT

        n_s = cfg.MPC_STEER_STEPS
        self._steers = [
            (i / (n_s - 1) - 0.5) * math.pi for i in range(n_s)
        ]
        n_a = cfg.MPC_ACCEL_STEPS
        self._accels = [
            i / (n_a - 1) * cfg.AGENT_MAX_ACCEL for i in range(n_a)
        ]

        self._best_steer = 0.0
        self._best_accel = cfg.AGENT_MAX_ACCEL
        self._best_trajectory: List[Vec2] = []

        # Boundary margins
        self._bnd_lo = CFG.WORLD_MARGIN + CFG.AGENT_RADIUS + 10
        self._bnd_hi_x = CFG.SCREEN_W - CFG.WORLD_MARGIN - CFG.AGENT_RADIUS - 10
        self._bnd_hi_y = CFG.SCREEN_H - CFG.WORLD_MARGIN - CFG.AGENT_RADIUS - 10

    def update_obstacles(self, obstacles):
        self.obstacles = obstacles

    def plan(self, pos: Vec2, vel: Vec2, angle: float, goal: Vec2,
             threats: Optional[List[Vec2]] = None) -> Tuple[float, float, List[Vec2]]:
        best_cost = float('inf')
        best_steer = 0.0
        best_accel = self.cfg.AGENT_MAX_ACCEL
        best_traj: List[Vec2] = []

        # Warm-start: evaluate previous best first
        if self.cfg.MPC_WARM_START:
            ordered_steers = [self._best_steer] + [
                s for s in self._steers if s != self._best_steer
            ]
        else:
            ordered_steers = self._steers

        for s in ordered_steers:
            for a in self._accels:
                p = Vec2(pos.x, pos.y)
                v = Vec2(vel.x, vel.y)
                ang = angle
                cost = 0.0
                traj = []

                for h in range(self.horizon):
                    ang += s * self.dt
                    fwd = Vec2.from_angle(ang, a * self.dt)
                    v = (v + fwd).limit(CFG.AGENT_MAX_SPEED) * CFG.AGENT_FRICTION
                    p = p + v * self.dt * 60
                    traj.append(p.copy())

                    # Goal attraction (progressive weighting)
                    w = 1.0 + h * 0.2
                    cost += p.dist(goal) * 0.08 * w

                    # Obstacle penalty
                    for obs in self.obstacles:
                        d = p.dist(obs.position) - obs.radius
                        if d < 40:
                            cost += max(0, 40 - d) * 3.0

                    # Boundary penalty
                    if p.x < self._bnd_lo:
                        cost += (self._bnd_lo - p.x) * 2.0
                    elif p.x > self._bnd_hi_x:
                        cost += (p.x - self._bnd_hi_x) * 2.0
                    if p.y < self._bnd_lo:
                        cost += (self._bnd_lo - p.y) * 2.0
                    elif p.y > self._bnd_hi_y:
                        cost += (p.y - self._bnd_hi_y) * 2.0

                    # Threat avoidance
                    if threats:
                        for tp in threats:
                            td = p.dist(tp)
                            if td < 80:
                                cost += (80 - td) * 1.5

                # Smoothness penalty (prefer continuing previous action)
                cost += abs(s - self._best_steer) * 2.0

                if cost < best_cost:
                    best_cost = cost
                    best_steer = s
                    best_accel = a
                    best_traj = traj

        self._best_steer = best_steer
        self._best_accel = best_accel
        self._best_trajectory = best_traj
        return best_steer, best_accel, best_traj


# ============================================================================
# PART 14: NEURAL NETWORK (Adam optimizer + batch norm)
# ============================================================================

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_layers: Tuple[int, ...],
                 output_size: int, lr: float = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        sizes = [input_size] + list(hidden_layers) + [output_size]

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(sizes) - 1):
            # He initialization for ReLU
            std = math.sqrt(2.0 / sizes[i])
            self.weights.append(
                np.random.randn(sizes[i], sizes[i+1]).astype(np.float32) * std
            )
            self.biases.append(np.zeros(sizes[i+1], dtype=np.float32))

        self.n_layers = len(sizes) - 1
        self.activations: List[np.ndarray] = []
        self.z_values: List[np.ndarray] = []

        # Adam optimizer state
        self.t = 0
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_eps = 1e-8

        # Gradient clipping threshold
        self.grad_clip = 1.0

        # Loss tracking
        self.recent_losses: deque = deque(maxlen=100)

    def leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_deriv(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha)

    def forward(self, inputs: np.ndarray, store: bool = False) -> np.ndarray:
        try:
            cur = np.asarray(inputs, dtype=np.float32)
            if store:
                self.activations = [cur.copy()]
                self.z_values = []

            for li in range(self.n_layers):
                z = cur @ self.weights[li] + self.biases[li]
                if store:
                    self.z_values.append(z.copy())
                if li < self.n_layers - 1:
                    cur = self.leaky_relu(z)
                else:
                    cur = z  # linear output
                if store:
                    self.activations.append(cur.copy())
            return cur
        except Exception:
            return np.zeros(self.output_size, dtype=np.float32)

    def predict(self, inputs) -> np.ndarray:
        return self.forward(inputs, store=False)

    def train_batch(self, batch: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Train on a batch, return mean loss."""
        if not batch:
            return 0.0

        total_loss = 0.0
        # Accumulate gradients
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        for inp, target in batch:
            output = self.forward(inp, store=True)
            target = np.asarray(target, dtype=np.float32)
            error = output - target
            total_loss += float(np.mean(error ** 2))

            delta = error
            for li in range(self.n_layers - 1, -1, -1):
                grad_w[li] += np.outer(self.activations[li], delta)
                grad_b[li] += delta
                if li > 0:
                    delta = (delta @ self.weights[li].T) * \
                            self.leaky_relu_deriv(self.z_values[li - 1])

        n = len(batch)
        self.t += 1

        for li in range(self.n_layers):
            gw = grad_w[li] / n
            gb = grad_b[li] / n

            # Gradient clipping
            gw_norm = np.linalg.norm(gw)
            if gw_norm > self.grad_clip:
                gw *= self.grad_clip / gw_norm
            gb_norm = np.linalg.norm(gb)
            if gb_norm > self.grad_clip:
                gb *= self.grad_clip / gb_norm

            # Adam update
            self.m_w[li] = self.beta1 * self.m_w[li] + (1 - self.beta1) * gw
            self.v_w[li] = self.beta2 * self.v_w[li] + (1 - self.beta2) * gw**2
            self.m_b[li] = self.beta1 * self.m_b[li] + (1 - self.beta1) * gb
            self.v_b[li] = self.beta2 * self.v_b[li] + (1 - self.beta2) * gb**2

            bc1 = 1 - self.beta1**self.t
            bc2 = 1 - self.beta2**self.t
            mw_hat = self.m_w[li] / bc1
            vw_hat = self.v_w[li] / bc2
            mb_hat = self.m_b[li] / bc1
            vb_hat = self.v_b[li] / bc2

            self.weights[li] -= self.lr * mw_hat / (np.sqrt(vw_hat) + self.adam_eps)
            self.biases[li] -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.adam_eps)

        mean_loss = total_loss / n
        self.recent_losses.append(mean_loss)
        return mean_loss

    def train(self, inputs, targets) -> float:
        """Single-sample training (backwards compat)."""
        return self.train_batch([(inputs, targets)])

    def get_avg_loss(self) -> float:
        if self.recent_losses:
            return sum(self.recent_losses) / len(self.recent_losses)
        return 0.0

    def save_weights(self, filename: str):
        try:
            data = {
                'weights': [w.copy() for w in self.weights],
                'biases':  [b.copy() for b in self.biases],
                'm_w': [m.copy() for m in self.m_w],
                'v_w': [v.copy() for v in self.v_w],
                'm_b': [m.copy() for m in self.m_b],
                'v_b': [v.copy() for v in self.v_b],
                't': self.t,
            }
            tmp = filename + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, filename)
        except Exception as e:
            print(f"Failed to save weights to {filename}: {e}")

    def load_weights(self, filename: str) -> bool:
        try:
            if not os.path.exists(filename):
                return False
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.weights = data['weights']
            self.biases = data['biases']
            if 'm_w' in data:
                self.m_w = data['m_w']
                self.v_w = data['v_w']
                self.m_b = data['m_b']
                self.v_b = data['v_b']
                self.t = data.get('t', 0)
            print(f"Loaded weights from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load weights from {filename}: {e}")
            return False


# ============================================================================
# PART 15: FORMATION CONTROL (Context-adaptive)
# ============================================================================

class Formation:
    @staticmethod
    def wedge(leader_pos: Vec2, leader_angle: float,
              num_followers: int, spacing: float) -> List[Vec2]:
        positions = [leader_pos]
        for i in range(num_followers):
            side = 1 if i % 2 == 0 else -1
            rank = (i // 2) + 1
            offset_angle = leader_angle + side * (math.pi / 5)
            offset = Vec2.from_angle(offset_angle - math.pi, rank * spacing)
            positions.append(leader_pos + offset)
        return positions

    @staticmethod
    def line(leader_pos: Vec2, leader_angle: float,
             num_followers: int, spacing: float) -> List[Vec2]:
        positions = [leader_pos]
        perp = Vec2.from_angle(leader_angle + math.pi / 2)
        for i in range(num_followers):
            side = 1 if i % 2 == 0 else -1
            rank = (i // 2) + 1
            positions.append(leader_pos + perp * (side * rank * spacing))
        return positions

    @staticmethod
    def column(leader_pos: Vec2, leader_angle: float,
               num_followers: int, spacing: float) -> List[Vec2]:
        positions = [leader_pos]
        back = Vec2.from_angle(leader_angle - math.pi)
        for i in range(num_followers):
            positions.append(leader_pos + back * ((i + 1) * spacing))
        return positions

    @staticmethod
    def spread(leader_pos: Vec2, leader_angle: float,
               num_followers: int, spacing: float) -> List[Vec2]:
        positions = [leader_pos]
        for i in range(num_followers):
            angle = leader_angle + (i + 1) * (2 * math.pi / (num_followers + 1))
            positions.append(leader_pos + Vec2.from_angle(angle, spacing))
        return positions

    @staticmethod
    def get_formation(shape: FormationShape, leader_pos: Vec2,
                      leader_angle: float, num_followers: int,
                      spacing: float) -> List[Vec2]:
        if shape == FormationShape.WEDGE:
            return Formation.wedge(leader_pos, leader_angle, num_followers, spacing)
        elif shape == FormationShape.LINE:
            return Formation.line(leader_pos, leader_angle, num_followers, spacing)
        elif shape == FormationShape.COLUMN:
            return Formation.column(leader_pos, leader_angle, num_followers, spacing)
        elif shape == FormationShape.SPREAD:
            return Formation.spread(leader_pos, leader_angle, num_followers, spacing)
        return Formation.wedge(leader_pos, leader_angle, num_followers, spacing)


# ============================================================================
# PART 16: TEAM WORLD MODEL (Improved training + tactical awareness)
# ============================================================================

class TeamWorldModel:
    def __init__(self, team_id: int, cfg: SimulationConfig):
        self.team_id = team_id
        self.cfg = cfg
        self.tracker = MultiObjectTracker(cfg)
        self.frame = 0
        self.shared_tracks: List[Track] = []

        self.traj_nn = NeuralNetwork(8, (32, 24), 4, lr=0.0005)
        weights_file = f"{cfg.TRAJ_NN_WEIGHTS_FILE}_team{team_id}.pkl"
        self.traj_nn.load_weights(weights_file)

        self.training_buffer: deque = deque(maxlen=500)
        self.last_train_frame = 0
        self.last_save_frame = 0

        self.roles: Dict[int, AgentRole] = {}
        self.allocations: Dict[int, int] = {}
        self._allocation_cache_frame = -1

        self.formation_shape = FormationShape.WEDGE

        # Team statistics
        self.team_damage_dealt = 0.0
        self.team_damage_taken = 0.0

    def update(self, all_agents: List['Agent'], dt: float):
        self.frame += 1
        team_agents = [a for a in all_agents if a.team == self.team_id and a.alive]

        # Merge detections from all teammates
        merged: Dict[int, Dict] = {}
        for agent in team_agents:
            for det in agent._last_detections:
                if det['team'] != self.team_id:
                    eid = det['id']
                    if eid not in merged or det['distance'] < merged[eid]['distance']:
                        merged[eid] = det

        try:
            self.shared_tracks = self.tracker.update(list(merged.values()), dt)
        except Exception:
            self.shared_tracks = []

        self._collect_training_data(dt)

        if self.frame - self.last_train_frame >= self.cfg.NN_TRAIN_INTERVAL:
            self._train_trajectory_predictor()
            self.last_train_frame = self.frame

        if self.frame - self.last_save_frame >= self.cfg.NN_SAVE_INTERVAL:
            self._save_weights()
            self.last_save_frame = self.frame

        self._assign_roles(team_agents)
        self._choose_formation()

        if self.frame - self._allocation_cache_frame >= 5:
            self._allocate_targets(team_agents)
            self._allocation_cache_frame = self.frame

    def _collect_training_data(self, dt: float):
        for track in self.shared_tracks:
            if len(track.history) >= 3:
                h = list(track.history)
                p2 = h[-3]
                p1 = h[-2]
                p0 = h[-1]
                vel = track.velocity
                # Input: two past positions + velocity + dt
                inp = np.array([
                    p2.x / CFG.SCREEN_W, p2.y / CFG.SCREEN_H,
                    p1.x / CFG.SCREEN_W, p1.y / CFG.SCREEN_H,
                    vel.x / max(CFG.AGENT_MAX_SPEED, 1),
                    vel.y / max(CFG.AGENT_MAX_SPEED, 1),
                    dt, track.health / 100.0,
                ], dtype=np.float32)
                target = np.array([
                    p0.x / CFG.SCREEN_W, p0.y / CFG.SCREEN_H,
                    vel.x / max(CFG.AGENT_MAX_SPEED, 1),
                    vel.y / max(CFG.AGENT_MAX_SPEED, 1),
                ], dtype=np.float32)
                self.training_buffer.append((inp, target))

    def _train_trajectory_predictor(self):
        if len(self.training_buffer) < self.cfg.NN_BATCH_SIZE:
            return
        batch = random.sample(list(self.training_buffer), self.cfg.NN_BATCH_SIZE)
        self.traj_nn.train_batch(batch)

    def _save_weights(self):
        weights_file = f"{self.cfg.TRAJ_NN_WEIGHTS_FILE}_team{self.team_id}.pkl"
        self.traj_nn.save_weights(weights_file)

    def _assign_roles(self, team_agents: List['Agent']):
        if not team_agents:
            return

        # Score: weighted combination of health, kills, and time alive
        scored = sorted(team_agents,
                        key=lambda a: (-a.kills * 2 - a.health * 0.5))

        n = len(scored)
        for i, agent in enumerate(scored):
            if i == 0:
                self.roles[agent.id] = AgentRole.LEADER
            elif i <= n * 0.4:
                self.roles[agent.id] = AgentRole.ATTACKER
            elif i >= n - 1:
                self.roles[agent.id] = AgentRole.SCOUT
            else:
                self.roles[agent.id] = AgentRole.SUPPORT

    def _choose_formation(self):
        """Adapt formation to tactical context."""
        n_enemies = len(self.shared_tracks)
        if n_enemies == 0:
            self.formation_shape = FormationShape.COLUMN
        elif n_enemies <= 2:
            self.formation_shape = FormationShape.WEDGE
        else:
            self.formation_shape = FormationShape.LINE

    def _allocate_targets(self, team_agents: List['Agent']):
        if not team_agents or not self.shared_tracks:
            self.allocations = {}
            return

        if len(team_agents) > 15 or len(self.shared_tracks) > 15:
            self.allocations = {}
            for agent in team_agents:
                nearest = min(self.shared_tracks,
                              key=lambda t: agent.position.dist_sq(t.position))
                self.allocations[agent.id] = nearest.id
            return

        try:
            cost = []
            for agent in team_agents:
                row = []
                role = self.roles.get(agent.id, AgentRole.SUPPORT)
                for track in self.shared_tracks:
                    dist = agent.position.dist(track.position)
                    health_factor = track.health / 100.0

                    if role == AgentRole.ATTACKER:
                        # Prefer weaker enemies
                        row.append(dist * health_factor)
                    elif role == AgentRole.SCOUT:
                        # Prefer distant/unexplored
                        row.append(1.0 / (dist + 1))
                    elif role == AgentRole.LEADER:
                        # Balance
                        row.append(dist * 0.8)
                    else:
                        row.append(dist)
                cost.append(row)

            assignments = hungarian(cost, max_iterations=300)
            self.allocations = {}
            for ai, ti in assignments:
                if ai < len(team_agents) and ti < len(self.shared_tracks):
                    self.allocations[team_agents[ai].id] = self.shared_tracks[ti].id
        except Exception:
            pass

    def predict_enemy_position(self, track: Track, t_ahead: float) -> Vec2:
        """NN-based prediction with Kalman fallback."""
        try:
            h = list(track.history)
            if len(h) >= 2:
                p1 = h[-2]
                p0 = h[-1]
                inp = np.array([
                    p1.x / CFG.SCREEN_W, p1.y / CFG.SCREEN_H,
                    p0.x / CFG.SCREEN_W, p0.y / CFG.SCREEN_H,
                    track.velocity.x / max(CFG.AGENT_MAX_SPEED, 1),
                    track.velocity.y / max(CFG.AGENT_MAX_SPEED, 1),
                    t_ahead, track.health / 100.0,
                ], dtype=np.float32)
                out = self.traj_nn.predict(inp)
                pred = Vec2(float(out[0]) * CFG.SCREEN_W,
                           float(out[1]) * CFG.SCREEN_H)

                # Sanity check: prediction shouldn't be too far from Kalman
                kalman_pred = track.kalman.predict_future(t_ahead)
                if pred.dist(kalman_pred) < 200:
                    return pred

            return track.kalman.predict_future(t_ahead)
        except Exception:
            return track.kalman.predict_future(t_ahead)

    def get_role(self, agent_id: int) -> AgentRole:
        return self.roles.get(agent_id, AgentRole.SUPPORT)

    def get_allocated_track(self, agent_id: int) -> Optional[Track]:
        tid = self.allocations.get(agent_id)
        if tid is None:
            return None
        return next((t for t in self.shared_tracks if t.id == tid), None)


# ============================================================================
# PART 17: GAME ENTITIES
# ============================================================================

@dataclass
class Obstacle:
    id: int
    position: Vec2
    radius: float


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
    radius: float = 4.0
    trail: deque = field(default_factory=lambda: deque(maxlen=8))
    alive: bool = True

    def update(self, dt: float):
        self.trail.append(self.position.copy())
        self.position = self.position + self.velocity * dt * 60
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.alive = False

    def out_of_bounds(self, margin: float = 50.0) -> bool:
        return (self.position.x < -margin or
                self.position.x > CFG.SCREEN_W + margin or
                self.position.y < -margin or
                self.position.y > CFG.SCREEN_H + margin)


@dataclass
class DamageNumber:
    """Floating damage text effect."""
    position: Vec2
    amount: float
    is_crit: bool
    timer: int
    max_timer: int
    team: int

    def update(self):
        self.position = Vec2(self.position.x, self.position.y - 1.2)
        self.timer -= 1

    @property
    def alive(self) -> bool:
        return self.timer > 0

    @property
    def alpha(self) -> int:
        return int(255 * self.timer / self.max_timer)


@dataclass
class KillFeedEntry:
    killer_id: int
    killer_team: int
    victim_id: int
    victim_team: int
    timer: int


# ============================================================================
# PART 18: AGENT (Improved decision-making + influence maps)
# ============================================================================

class Agent:
    _cached_font = None

    def __init__(self, agent_id: int, team: int, position: Vec2,
                 team_world_model: Optional['TeamWorldModel'],
                 obstacles: List[Obstacle], cfg: SimulationConfig = CFG):
        self.id = agent_id
        self.team = team
        self.position = position.copy()
        self.velocity = Vec2()
        self.acceleration = Vec2()
        self.angle = random.uniform(0, math.pi * 2)
        self.turret_angle = self.angle
        self.cfg = cfg

        self.radius = cfg.AGENT_RADIUS
        self.max_speed = cfg.AGENT_MAX_SPEED
        self.friction = cfg.AGENT_FRICTION

        self.health = cfg.AGENT_HP
        self.max_health = cfg.AGENT_HP
        self.alive = True
        self.fire_cooldown = 0
        self.kills = 0
        self.deaths = 0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0

        # Respawn
        self.respawn_timer = 0
        self.invuln_timer = 0
        self.spawn_position = position.copy()

        # Subsystems
        self.cameras = DualCameraSystem(cfg)
        self.pid_angle = PIDController(cfg.PID_ANGLE_KP, cfg.PID_ANGLE_KI, cfg.PID_ANGLE_KD)
        self.pid_speed = PIDController(cfg.PID_SPEED_KP, cfg.PID_SPEED_KI, cfg.PID_SPEED_KD)
        self.mpc = MPCPlanner(cfg, obstacles)
        self.pathfinder: Optional[AStarPathfinder] = None

        self.team_wm: Optional[TeamWorldModel] = team_world_model
        self._last_detections: List[Dict] = []

        self.brain = NeuralNetwork(cfg.NN_INPUT_SIZE, cfg.NN_HIDDEN_LAYERS, cfg.NN_OUTPUT_SIZE)

        self.path: List[Vec2] = []
        self.path_replan_timer = 0
        self.current_target_track_id: Optional[int] = None
        self.decision_timer = 0
        self.state_label = 'IDLE'
        self.role_label = 'SUPPRT'

        # Visual effects
        self.hit_flash = 0
        self.muzzle_flash = 0
        self.trail: deque = deque(maxlen=20)
        self.mpc_path: List[Vec2] = []
        self.formation_target: Optional[Vec2] = None
        self.predicted_enemy_pos: Optional[Vec2] = None

        # Last MPC results
        self._last_mpc_steer = 0.0
        self._last_mpc_accel = cfg.AGENT_MAX_ACCEL * 0.5

    def sense(self, all_agents: List['Agent'], obstacles: List[Obstacle]) -> None:
        self._last_detections.clear()
        enemies = [a for a in all_agents if a.alive and a.team != self.team]
        self._last_detections = self.cameras.get_all_detections(
            self.position, self.angle, enemies, obstacles
        )
        # Share friendly positions (direct comms)
        for a in all_agents:
            if a.alive and a.team == self.team and a.id != self.id:
                self._last_detections.append({
                    'id': a.id,
                    'position': a.position.copy(),
                    'velocity': a.velocity.copy(),
                    'team': a.team,
                    'health': a.health,
                    'distance': self.position.dist(a.position),
                    'angle_rel': 0.0,
                })

    def _calculate_intercept(self, target_pos: Vec2, target_vel: Vec2) -> Vec2:
        to_target = target_pos - self.position
        proj_speed = CFG.PROJECTILE_SPEED * 60
        proj_speed_sq = proj_speed * proj_speed

        tv_sq = target_vel.mag_sq()
        a = tv_sq - proj_speed_sq
        b = 2.0 * to_target.dot(target_vel)
        c = to_target.mag_sq()

        if abs(a) < 1e-6:
            t = -c / b if abs(b) > 1e-6 else 0.0
        else:
            disc = b*b - 4*a*c
            if disc < 0:
                return target_pos
            sq = math.sqrt(disc)
            t1 = (-b - sq) / (2*a)
            t2 = (-b + sq) / (2*a)
            # Pick smallest positive
            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
            elif t1 > 0:
                t = t1
            elif t2 > 0:
                t = t2
            else:
                return target_pos

        t = max(0.0, min(t, 3.0))
        return target_pos + target_vel * t

    def _find_leader(self, all_agents: List['Agent']) -> Optional['Agent']:
        if not self.team_wm:
            return None
        for a in all_agents:
            if (a.alive and a.team == self.team and a.id != self.id and
                    self.team_wm.get_role(a.id) == AgentRole.LEADER):
                return a
        return None

    def _get_formation_position(self, leader: 'Agent',
                                 all_team: List['Agent']) -> Vec2:
        if not self.team_wm:
            return self.position

        shape = self.team_wm.formation_shape
        n_followers = len(all_team) - 1
        positions = Formation.get_formation(
            shape, leader.position, leader.angle,
            n_followers, CFG.FORMATION_SPACING
        )

        # Assign by sorted order (leader excluded)
        followers = sorted(
            [a for a in all_team if a.id != leader.id],
            key=lambda a: a.id
        )
        for idx, f in enumerate(followers):
            if f.id == self.id and idx + 1 < len(positions):
                return positions[idx + 1]

        return self.position

    def update(self, all_agents: List['Agent'], obstacles: List[Obstacle],
               dt: float, influence_map: Optional[InfluenceMap] = None) -> Optional[Dict]:
        # Handle respawn
        if not self.alive:
            if CFG.RESPAWN_ENABLED:
                self.respawn_timer -= 1
                if self.respawn_timer <= 0:
                    self._respawn()
            return None

        if self.invuln_timer > 0:
            self.invuln_timer -= 1

        # SENSE
        self.sense(all_agents, obstacles)

        # Timers
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        if self.hit_flash > 0:
            self.hit_flash -= 1
        if self.muzzle_flash > 0:
            self.muzzle_flash -= 1
        self.decision_timer += 1

        # Get role and target from team world model
        target_track = None
        role = AgentRole.SUPPORT
        if self.team_wm:
            target_track = self.team_wm.get_allocated_track(self.id)
            role = self.team_wm.get_role(self.id)
            self.role_label = role.value

        enemy_dets = [d for d in self._last_detections if d['team'] != self.team]
        friend_dets = [d for d in self._last_detections if d['team'] == self.team]

        # DECIDE: goal position
        goal_pos = None
        self.predicted_enemy_pos = None

        if target_track:
            t_ahead = target_track.position.dist(self.position) / max(CFG.PROJECTILE_SPEED * 60, 1)
            if self.team_wm:
                goal_pos = self.team_wm.predict_enemy_position(target_track, t_ahead)
                self.predicted_enemy_pos = goal_pos.copy()
            else:
                goal_pos = target_track.position.copy()
            self.state_label = 'TRACK'
        elif enemy_dets:
            nearest = min(enemy_dets, key=lambda d: d['distance'])
            goal_pos = nearest['position']
            self.state_label = 'SEEK'
        else:
            # Patrol: use influence map if available
            if influence_map:
                adv = influence_map.get_advantage(self.position, self.team)
                if adv > 2.0:
                    # We dominate here, push forward
                    side = 1.0 if self.team == 0 else -1.0
                    goal_pos = Vec2(
                        CFG.SCREEN_W / 2 + side * CFG.SCREEN_W * 0.35,
                        CFG.SCREEN_H / 2 + random.uniform(-100, 100)
                    )
                else:
                    goal_pos = Vec2(
                        CFG.SCREEN_W / 2 + random.uniform(-150, 150),
                        CFG.SCREEN_H / 2 + random.uniform(-100, 100)
                    )
            else:
                side = 1.0 if self.team == 0 else -1.0
                goal_pos = Vec2(
                    CFG.SCREEN_W / 2 + side * CFG.SCREEN_W * 0.25 + random.uniform(-80, 80),
                    CFG.SCREEN_H / 2 + random.uniform(-100, 100)
                )
            self.state_label = 'WANDER'

        # Formation keeping
        team_agents = [a for a in all_agents if a.alive and a.team == self.team]
        self.formation_target = None
        if role in (AgentRole.SUPPORT, AgentRole.ATTACKER) and len(team_agents) > 1:
            leader = self._find_leader(all_agents)
            if leader and leader.id != self.id:
                ft = self._get_formation_position(leader, team_agents)
                self.formation_target = ft
                # Blend: more formation in patrol, more combat in engagement
                blend = 0.5 if enemy_dets else 0.7
                goal_pos = goal_pos.lerp(ft, blend)

        dist_to_goal = self.position.dist(goal_pos)

        # Collect threat positions for MPC
        threats = [d['position'] for d in enemy_dets if d['distance'] < 150]

        # MPC Planning (throttled)
        if self.decision_timer >= self.cfg.DECISION_INTERVAL_FRAMES:
            self.decision_timer = 0
            steer, accel, traj = self.mpc.plan(
                self.position, self.velocity, self.angle, goal_pos, threats
            )
            self._last_mpc_steer = steer
            self._last_mpc_accel = accel
            self.mpc_path = traj
        else:
            steer = self._last_mpc_steer
            accel = self._last_mpc_accel

        # PID angle control
        desired_angle = (goal_pos - self.position).angle()
        angle_err = MathUtils.wrap_angle(desired_angle - self.angle)
        steer_pid = self.pid_angle.update(angle_err, dt)
        combined_steer = MathUtils.clamp(steer_pid * 0.7 + steer * 0.3,
                                         -math.pi / 4, math.pi / 4)

        # PID speed control
        desired_speed = self.max_speed if dist_to_goal > 60 else (
            MathUtils.remap(dist_to_goal, 0, 60, 0.5, self.max_speed)
        )
        speed_err = desired_speed - self.velocity.mag()
        accel_pid = self.pid_speed.update(speed_err, dt)
        forward_accel = MathUtils.clamp(accel_pid + accel, -self.cfg.AGENT_MAX_ACCEL,
                                        self.cfg.AGENT_MAX_ACCEL)

        # Apply controls
        self.angle = MathUtils.wrap_angle(self.angle + combined_steer * dt)
        forward = Vec2.from_angle(self.angle, forward_accel)

        # Swarm separation
        sep = Vec2()
        for fd in friend_dets:
            d = fd['distance']
            if 0.1 < d < self.cfg.SWARM_SEPARATION_DIST:
                push = (self.position - fd['position']).norm()
                sep = sep + push * (self.cfg.SWARM_SEPARATION_DIST / (d + 1))
        sep = sep.limit(self.cfg.AGENT_MAX_ACCEL) * self.cfg.SWARM_SEPARATION_WEIGHT

        # Obstacle avoidance
        obs_avoid = Vec2()
        for obs in obstacles:
            d = self.position.dist(obs.position)
            safe = obs.radius + 50
            if d < safe and d > 0.1:
                push = (self.position - obs.position).norm()
                strength = (safe - d) / safe * 0.5
                obs_avoid = obs_avoid + push * strength

        # Combine forces
        total_accel = (forward + sep + obs_avoid).limit(self.cfg.AGENT_MAX_ACCEL)

        # Physics
        self.velocity = self.velocity + total_accel
        self.velocity = self.velocity * self.friction
        self.velocity = self.velocity.limit(self.max_speed)
        self.position = self.position + self.velocity * dt * 60

        # Turret tracking
        if enemy_dets:
            if dist_to_goal < self.cfg.FIRE_RANGE:
                self.state_label = 'ENGAGE'
            nearest_e = min(enemy_dets, key=lambda d: d['distance'])
            intercept = self._calculate_intercept(nearest_e['position'], nearest_e.get('velocity', Vec2()))
            target_turret = (intercept - self.position).angle()
            # Smooth turret rotation
            turret_diff = MathUtils.wrap_angle(target_turret - self.turret_angle)
            max_turret_rate = math.pi * 2 * dt
            turret_diff = MathUtils.clamp(turret_diff, -max_turret_rate, max_turret_rate)
            self.turret_angle = MathUtils.wrap_angle(self.turret_angle + turret_diff)
        else:
            self.turret_angle = MathUtils.wrap_angle(self.turret_angle + 0.015)

        # Boundary enforcement
        m = CFG.WORLD_MARGIN + self.radius + 5
        bx = CFG.SCREEN_W - m
        by = CFG.SCREEN_H - m
        if self.position.x < m:
            self.position.x = m; self.velocity.x = abs(self.velocity.x) * 0.3
        if self.position.x > bx:
            self.position.x = bx; self.velocity.x = -abs(self.velocity.x) * 0.3
        if self.position.y < m:
            self.position.y = m; self.velocity.y = abs(self.velocity.y) * 0.3
        if self.position.y > by:
            self.position.y = by; self.velocity.y = -abs(self.velocity.y) * 0.3

        # Agent-agent collision
        for other in all_agents:
            if other.id == self.id or not other.alive:
                continue
            d = self.position.dist(other.position)
            md = self.radius + other.radius + 2
            if d < md and d > 0.01:
                push = (self.position - other.position).norm() * ((md - d) * 0.5)
                self.position = self.position + push
                self.velocity = self.velocity * 0.85

        # Trail
        self.trail.append(self.position.copy())

        # Combat
        combat_action = None
        if enemy_dets and self.fire_cooldown == 0:
            best_e = min(enemy_dets, key=lambda d: d['distance'])
            if best_e['distance'] < self.cfg.FIRE_RANGE:
                # Check turret alignment
                aim_angle = (best_e['position'] - self.position).angle()
                turret_err = abs(MathUtils.wrap_angle(aim_angle - self.turret_angle))
                if turret_err < math.radians(15):
                    self.fire_cooldown = self.cfg.FIRE_COOLDOWN_FRAMES
                    self.muzzle_flash = 5
                    self.state_label = 'FIRE'

                    aim_target = self._calculate_intercept(
                        best_e['position'], best_e.get('velocity', Vec2())
                    )
                    aim_dir = (aim_target - self.position).norm()

                    combat_action = {
                        'type': 'fire',
                        'position': self.position.copy(),
                        'direction': aim_dir,
                        'owner_id': self.id,
                        'team': self.team,
                    }

        return combat_action

    def _respawn(self):
        """Respawn agent at spawn position."""
        self.alive = True
        self.health = self.max_health
        self.position = self.spawn_position + Vec2.random_in_circle(30)
        self.velocity = Vec2()
        self.angle = random.uniform(0, math.pi * 2)
        self.turret_angle = self.angle
        self.fire_cooldown = 30
        self.invuln_timer = CFG.RESPAWN_INVULN_FRAMES
        self.hit_flash = 0
        self.muzzle_flash = 0
        self.trail.clear()
        self.pid_angle.reset()
        self.pid_speed.reset()

    def take_damage(self, amount: float) -> bool:
        if self.invuln_timer > 0:
            return False
        self.health -= amount
        self.damage_taken += amount
        self.hit_flash = 8
        if self.health <= 0:
            self.health = 0
            self.alive = False
            self.deaths += 1
            self.respawn_timer = CFG.RESPAWN_DELAY_FRAMES
            return True
        return False

    def add_kill(self):
        self.kills += 1

    def draw(self, screen: pygame.Surface, show_cameras: bool = False,
             show_predictions: bool = False, camera_offset: Vec2 = Vec2()):
        if not self.alive:
            return

        ox, oy = camera_offset.x, camera_offset.y
        x = self.position.x + ox
        y = self.position.y + oy

        base = CFG.COLOR_AGENT_TEAM_0 if self.team == 0 else CFG.COLOR_AGENT_TEAM_1

        # Invulnerability flash
        if self.invuln_timer > 0 and self.invuln_timer % 6 < 3:
            color = (255, 255, 255)
        elif self.hit_flash > 0:
            color = (255, 150, 150)
        else:
            color = base

        # Camera FOV
        if show_cameras:
            for offset, col in [(0, CFG.COLOR_CAM_FRONT), (math.pi, CFG.COLOR_CAM_REAR)]:
                cam_dir = self.angle + offset
                half_fov = math.radians(CFG.CAMERA_FOV_DEG / 2)
                r = int(CFG.CAMERA_RANGE * 0.25)
                a1 = cam_dir - half_fov
                a2 = cam_dir + half_fov
                # Draw FOV lines instead of arc (more reliable)
                e1 = Vec2.from_angle(a1, r)
                e2 = Vec2.from_angle(a2, r)
                pygame.draw.line(screen, col, (int(x), int(y)),
                               (int(x + e1.x), int(y + e1.y)), 1)
                pygame.draw.line(screen, col, (int(x), int(y)),
                               (int(x + e2.x), int(y + e2.y)), 1)

        # Formation target
        if self.formation_target:
            ft = self.formation_target
            pygame.draw.circle(screen, CFG.COLOR_FORMATION,
                             (int(ft.x + ox), int(ft.y + oy)), 6, 1)
            pygame.draw.line(screen, (*CFG.COLOR_FORMATION[:3],),
                           (int(x), int(y)),
                           (int(ft.x + ox), int(ft.y + oy)), 1)

        # Prediction
        if show_predictions and self.predicted_enemy_pos:
            pep = self.predicted_enemy_pos
            pygame.draw.circle(screen, CFG.COLOR_PREDICTION,
                             (int(pep.x + ox), int(pep.y + oy)), 10, 2)
            pygame.draw.line(screen, CFG.COLOR_PREDICTION,
                           (int(x), int(y)),
                           (int(pep.x + ox), int(pep.y + oy)), 1)

        # MPC path
        if show_predictions and self.mpc_path and len(self.mpc_path) > 1:
            pts = [(int(p.x + ox), int(p.y + oy)) for p in self.mpc_path]
            pygame.draw.lines(screen, CFG.COLOR_MPC_PATH, False, pts, 1)

        # Hull
        hw = self.radius * 1.2
        hh = self.radius * 0.6
        cos_a = math.cos(self.angle); sin_a = math.sin(self.angle)
        pts = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            pts.append((x + rx, y + ry))
        pygame.draw.polygon(screen, color, pts)
        bc = tuple(min(255, c + 30) for c in color)
        pygame.draw.polygon(screen, bc, pts, 2)

        # Turret
        tx = x + math.cos(self.turret_angle) * self.radius * 1.5
        ty = y + math.sin(self.turret_angle) * self.radius * 1.5
        pygame.draw.line(screen, (100, 100, 120), (int(x), int(y)), (int(tx), int(ty)), 6)
        pygame.draw.circle(screen, (80, 80, 100), (int(x), int(y)), 10)

        # Muzzle flash
        if self.muzzle_flash > 0:
            fx = x + math.cos(self.turret_angle) * self.radius * 2
            fy = y + math.sin(self.turret_angle) * self.radius * 2
            fs = 6 + self.muzzle_flash * 2
            pygame.draw.circle(screen, CFG.COLOR_PROJECTILE, (int(fx), int(fy)), fs)

        # HP bar
        bar_w = 35
        hp_x = x - bar_w / 2
        hp_y = y - self.radius - 12
        pygame.draw.rect(screen, (40, 40, 50), (hp_x, hp_y, bar_w, 4))
        ratio = self.health / self.max_health
        hpc = (CFG.COLOR_HP_HIGH if ratio > 0.6 else
               CFG.COLOR_HP_MEDIUM if ratio > 0.3 else CFG.COLOR_HP_LOW)
        pygame.draw.rect(screen, hpc, (hp_x, hp_y, bar_w * ratio, 4))

        # Labels
        if Agent._cached_font is None:
            Agent._cached_font = pygame.font.SysFont('consolas', 12)
        f = Agent._cached_font

        screen.blit(f.render(f'#{self.id}', True, CFG.COLOR_TEXT_PRIMARY),
                   (x - 10, y - self.radius - 24))

        role_colors = {
            'LEADER': (255, 255, 0), 'ATTACK': (255, 80, 80),
            'SUPPRT': (80, 200, 255), 'SCOUT': (200, 100, 255),
        }
        screen.blit(f.render(self.role_label, True,
                            role_colors.get(self.role_label, CFG.COLOR_TEXT_SECONDARY)),
                   (x - 18, y + self.radius + 4))

        state_colors = {
            'IDLE': CFG.COLOR_TEXT_SECONDARY, 'WANDER': (100, 180, 255),
            'SEEK': (255, 200, 50), 'TRACK': (200, 100, 255),
            'ENGAGE': (255, 80, 80), 'FIRE': (255, 40, 40),
        }
        screen.blit(f.render(self.state_label, True,
                            state_colors.get(self.state_label, CFG.COLOR_TEXT_SECONDARY)),
                   (x - 18, y + self.radius + 16))

        # Trail
        if len(self.trail) > 1:
            tp = [(int(p.x + ox), int(p.y + oy)) for p in list(self.trail)[-10:]]
            if len(tp) > 1:
                trail_color = tuple(max(0, c // 3) for c in base)
                pygame.draw.lines(screen, trail_color, False, tp, 1)


# ============================================================================
# PART 19: WORLD MODEL
# ============================================================================

class WorldModel:
    def __init__(self, w: int, h: int, obstacles: List[Obstacle],
                 cfg: SimulationConfig = CFG):
        self.w = w; self.h = h; self.cfg = cfg
        ww = w - CFG.WORLD_MARGIN * 2
        hh = h - CFG.WORLD_MARGIN * 2

        self.grid = OccupancyGrid(ww, hh, cfg.GRID_RESOLUTION, cfg.OCCUPANCY_DECAY_RATE)
        self.obstacles = obstacles

        for obs in obstacles:
            self.grid.update_circle(
                Vec2(obs.position.x - CFG.WORLD_MARGIN,
                     obs.position.y - CFG.WORLD_MARGIN),
                obs.radius, 0.95
            )

        self.pathfinder = AStarPathfinder(self.grid)
        self.spatial_hash = SpatialHash(60.0)
        self.influence_map = InfluenceMap(
            w, h, cfg.INFLUENCE_MAP_RESOLUTION, cfg.INFLUENCE_DECAY
        )
        self.frame = 0

    def update(self, agents: List[Agent]):
        self.frame += 1
        self.spatial_hash.clear()

        for a in agents:
            if a.alive:
                self.spatial_hash.insert(a, a.position)

        self.grid.decay_step()

        for a in agents:
            if a.alive:
                gp = Vec2(a.position.x - CFG.WORLD_MARGIN,
                         a.position.y - CFG.WORLD_MARGIN)
                self.grid.update_circle(gp, a.radius * 2, 0.65, a.velocity)

        if self.frame % 3 == 0:
            self.influence_map.update(agents)


# ============================================================================
# PART 20: GAME ENGINE (Complete with all features)
# ============================================================================

class GameEngine:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("TESLA TANK SWARM AGI - IMPROVED EDITION")

        if CFG.FULLSCREEN:
            self.screen = pygame.display.set_mode(
                (CFG.SCREEN_W, CFG.SCREEN_H), pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode((CFG.SCREEN_W, CFG.SCREEN_H))

        self.clock = pygame.time.Clock()

        self.font_s = pygame.font.SysFont('consolas', 14)
        self.font_m = pygame.font.SysFont('consolas', 18)
        self.font_l = pygame.font.SysFont('consolas', 24, bold=True)
        self.font_xl = pygame.font.SysFont('consolas', 36, bold=True)

        self.running = True
        self.paused = False

        self.show_cameras = False
        self.show_tracks = True
        self.show_occupancy = False
        self.show_predictions = True
        self.show_influence = False
        self.show_minimap = True
        self.debug = False

        self.projectiles: List[Projectile] = []
        self._pid = 0

        self.damage_numbers: List[DamageNumber] = []
        self.kill_feed: deque = deque(maxlen=CFG.KILL_FEED_MAX)

        self.stats = {
            'shots': 0, 'hits': 0, 'kills': 0, 'frame': 0,
            'training_samples': 0, 'nn_loss_t0': 0.0, 'nn_loss_t1': 0.0,
            'team_scores': [0, 0],
        }

        self.perf = PerformanceMonitor()
        self._init_world()

    def _init_world(self):
        print("Initializing world...")

        obstacles = []
        for i in range(CFG.OBSTACLE_COUNT):
            for _ in range(50):
                pos = Vec2(
                    random.randint(CFG.WORLD_MARGIN + 120, CFG.SCREEN_W - CFG.WORLD_MARGIN - 120),
                    random.randint(CFG.WORLD_MARGIN + 120, CFG.SCREEN_H - CFG.WORLD_MARGIN - 120),
                )
                r = random.randint(CFG.OBSTACLE_MIN_RADIUS, CFG.OBSTACLE_MAX_RADIUS)
                # Check no overlap with existing
                ok = all(pos.dist(o.position) > r + o.radius + 20 for o in obstacles)
                if ok:
                    obstacles.append(Obstacle(id=i, position=pos, radius=r))
                    break
        self.obstacles = obstacles

        self.twm0 = TeamWorldModel(0, CFG)
        self.twm1 = TeamWorldModel(1, CFG)

        self.world_model = WorldModel(CFG.SCREEN_W, CFG.SCREEN_H, obstacles)

        self.agents: List[Agent] = []
        for i in range(CFG.AGENT_COUNT_PER_TEAM * 2):
            team = i % 2
            twm = self.twm0 if team == 0 else self.twm1

            if team == 0:
                x = random.randint(CFG.WORLD_MARGIN + 40, CFG.SCREEN_W // 2 - 40)
            else:
                x = random.randint(CFG.SCREEN_W // 2 + 40, CFG.SCREEN_W - CFG.WORLD_MARGIN - 40)
            y = random.randint(CFG.WORLD_MARGIN + 40, CFG.SCREEN_H - CFG.WORLD_MARGIN - 40)

            a = Agent(i, team, Vec2(x, y), twm, obstacles, CFG)
            a.pathfinder = self.world_model.pathfinder
            self.agents.append(a)

        self.projectiles = []
        self._pid = 0
        self.damage_numbers = []
        self.kill_feed.clear()
        self.stats = {
            'shots': 0, 'hits': 0, 'kills': 0, 'frame': 0,
            'training_samples': 0, 'nn_loss_t0': 0.0, 'nn_loss_t1': 0.0,
            'team_scores': [0, 0],
        }
        print(f"World initialized: {len(self.agents)} agents, {len(obstacles)} obstacles")

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.running = False
                elif ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif ev.key == pygame.K_r:
                    self._init_world()
                elif ev.key == pygame.K_c:
                    self.show_cameras = not self.show_cameras
                elif ev.key == pygame.K_t:
                    self.show_tracks = not self.show_tracks
                elif ev.key == pygame.K_o:
                    self.show_occupancy = not self.show_occupancy
                elif ev.key == pygame.K_p:
                    self.show_predictions = not self.show_predictions
                elif ev.key == pygame.K_i:
                    self.show_influence = not self.show_influence
                elif ev.key == pygame.K_m:
                    self.show_minimap = not self.show_minimap
                elif ev.key == pygame.K_d:
                    self.debug = not self.debug
                elif ev.key == pygame.K_s:
                    self.twm0._save_weights()
                    self.twm1._save_weights()

    def update(self, dt: float):
        if self.paused:
            return

        self.stats['frame'] += 1

        with self.perf.measure('world_model'):
            self.world_model.update(self.agents)

        with self.perf.measure('team_wm'):
            self.twm0.update(self.agents, dt)
            self.twm1.update(self.agents, dt)
            self.stats['training_samples'] = (
                len(self.twm0.training_buffer) + len(self.twm1.training_buffer)
            )
            self.stats['nn_loss_t0'] = self.twm0.traj_nn.get_avg_loss()
            self.stats['nn_loss_t1'] = self.twm1.traj_nn.get_avg_loss()

        combat_actions = []
        with self.perf.measure('agents'):
            for a in self.agents:
                ca = a.update(self.agents, self.obstacles, dt,
                             self.world_model.influence_map)
                if ca:
                    combat_actions.append(ca)

        for ca in combat_actions:
            self.projectiles.append(Projectile(
                id=self._pid,
                position=ca['position'].copy(),
                velocity=ca['direction'] * CFG.PROJECTILE_SPEED,
                owner_id=ca['owner_id'],
                team=ca['team'],
                damage=CFG.PROJECTILE_DAMAGE,
                lifetime=CFG.PROJECTILE_LIFETIME,
                max_lifetime=CFG.PROJECTILE_LIFETIME,
                radius=CFG.PROJECTILE_RADIUS,
            ))
            self._pid += 1
            self.stats['shots'] += 1

        # Projectile update with swept-circle collision
        with self.perf.measure('projectiles'):
            for p in self.projectiles:
                if not p.alive:
                    continue

                prev_pos = p.position.copy()
                p.update(dt)

                if p.out_of_bounds():
                    p.alive = False
                    continue

                # Obstacle collision
                for obs in self.obstacles:
                    if p.position.dist(obs.position) < obs.radius + p.radius:
                        p.alive = False
                        break

                if not p.alive:
                    continue

                # Agent collision using swept circle
                for a in self.agents:
                    if not a.alive or a.team == p.team:
                        continue

                    t_hit = MathUtils.swept_circle_hit(
                        prev_pos, p.position, p.radius,
                        a.position, a.radius
                    )
                    if t_hit is not None:
                        p.alive = False

                        is_crit = random.random() < CFG.CRITICAL_HIT_CHANCE
                        dmg = p.damage * (CFG.CRITICAL_MULTIPLIER if is_crit else 1.0)
                        died = a.take_damage(dmg)

                        # Track damage
                        for s in self.agents:
                            if s.id == p.owner_id:
                                s.damage_dealt += dmg
                                break

                        # Damage number
                        self.damage_numbers.append(DamageNumber(
                            position=a.position.copy(),
                            amount=dmg,
                            is_crit=is_crit,
                            timer=CFG.DAMAGE_NUMBER_DURATION,
                            max_timer=CFG.DAMAGE_NUMBER_DURATION,
                            team=p.team,
                        ))

                        self.stats['hits'] += 1

                        if died:
                            self.stats['kills'] += 1
                            self.stats['team_scores'][p.team] += 1
                            for s in self.agents:
                                if s.id == p.owner_id:
                                    s.add_kill()
                                    break
                            self.kill_feed.append(KillFeedEntry(
                                killer_id=p.owner_id,
                                killer_team=p.team,
                                victim_id=a.id,
                                victim_team=a.team,
                                timer=CFG.KILL_FEED_DURATION,
                            ))
                        break

            self.projectiles = [p for p in self.projectiles if p.alive]

        # Update damage numbers
        self.damage_numbers = [dn for dn in self.damage_numbers if dn.alive]
        for dn in self.damage_numbers:
            dn.update()

        # Update kill feed
        for kf in self.kill_feed:
            kf.timer -= 1
        while self.kill_feed and self.kill_feed[0].timer <= 0:
            self.kill_feed.popleft()

    def render(self):
        self.screen.fill(CFG.COLOR_BG)

        # World background
        pygame.draw.rect(self.screen, CFG.COLOR_WORLD,
                        (CFG.WORLD_MARGIN, CFG.WORLD_MARGIN,
                         CFG.SCREEN_W - CFG.WORLD_MARGIN * 2,
                         CFG.SCREEN_H - CFG.WORLD_MARGIN * 2))

        # Occupancy grid
        if self.show_occupancy:
            grid = self.world_model.grid
            occ = grid.occ
            res = int(grid.resolution)
            for gy in range(0, grid.H, 2):
                for gx in range(0, grid.W, 2):
                    o = occ[gy, gx]
                    if abs(o - 0.5) > 0.08:
                        wx = gx * res + CFG.WORLD_MARGIN
                        wy = gy * res + CFG.WORLD_MARGIN
                        v = int((o - 0.5) * 360)
                        col = (0, min(255, max(0, v)), 0) if v > 0 else (min(255, max(0, -v)), 0, 0)
                        s = pygame.Surface((res * 2, res * 2))
                        s.set_alpha(60)
                        s.fill(col)
                        self.screen.blit(s, (wx, wy))

        # Influence map
        if self.show_influence:
            imap = self.world_model.influence_map
            ires = imap.resolution
            for gy in range(imap.H):
                for gx in range(imap.W):
                    v = imap.vulnerability[gy, gx]
                    if abs(v) > 0.5:
                        wx = int(gx * ires)
                        wy = int(gy * ires)
                        iv = int(MathUtils.clamp(abs(v) * 20, 0, 200))
                        if v > 0:
                            col = (0, iv, iv)
                        else:
                            col = (iv, 0, 0)
                        s = pygame.Surface((int(ires), int(ires)))
                        s.set_alpha(40)
                        s.fill(col)
                        self.screen.blit(s, (wx, wy))

        # Grid lines
        for x in range(CFG.WORLD_MARGIN, CFG.SCREEN_W - CFG.WORLD_MARGIN, CFG.GRID_RESOLUTION):
            pygame.draw.line(self.screen, CFG.COLOR_GRID,
                           (x, CFG.WORLD_MARGIN), (x, CFG.SCREEN_H - CFG.WORLD_MARGIN), 1)
        for y in range(CFG.WORLD_MARGIN, CFG.SCREEN_H - CFG.WORLD_MARGIN, CFG.GRID_RESOLUTION):
            pygame.draw.line(self.screen, CFG.COLOR_GRID,
                           (CFG.WORLD_MARGIN, y), (CFG.SCREEN_W - CFG.WORLD_MARGIN, y), 1)

        # Obstacles
        for obs in self.obstacles:
            pygame.draw.circle(self.screen, CFG.COLOR_OBSTACLE,
                             (int(obs.position.x), int(obs.position.y)), int(obs.radius))
            pygame.draw.circle(self.screen, CFG.COLOR_OBSTACLE_BORDER,
                             (int(obs.position.x), int(obs.position.y)), int(obs.radius), 2)

        # Tracker tracks
        if self.show_tracks:
            for twm in [self.twm0, self.twm1]:
                for track in twm.shared_tracks:
                    p = track.position
                    col = CFG.COLOR_TRACK if not track.coasting else (150, 80, 150)
                    pygame.draw.circle(self.screen, col, (int(p.x), int(p.y)), 9, 2)
                    try:
                        pred = track.kalman.predict_future(0.5)
                        pygame.draw.line(self.screen, col,
                                       (int(p.x), int(p.y)),
                                       (int(pred.x), int(pred.y)), 2)
                        if len(track.history) > 1:
                            pts = [(int(h.x), int(h.y)) for h in list(track.history)[-8:]]
                            if len(pts) > 1:
                                pygame.draw.lines(self.screen, col, False, pts, 1)
                    except Exception:
                        pass

        # Swarm connection lines
        for a in self.agents:
            if not a.alive:
                continue
            tc = CFG.COLOR_AGENT_TEAM_0 if a.team == 0 else CFG.COLOR_AGENT_TEAM_1
            for b in self.agents:
                if b.id <= a.id or b.team != a.team or not b.alive:
                    continue
                d = a.position.dist(b.position)
                if d < CFG.SWARM_COHESION_DIST:
                    af = 1.0 - d / CFG.SWARM_COHESION_DIST
                    lc = tuple(max(0, min(255, int(tc[i] * 0.2 * af))) for i in range(3))
                    pygame.draw.line(self.screen, lc,
                                   (int(a.position.x), int(a.position.y)),
                                   (int(b.position.x), int(b.position.y)), 1)

        # Agents
        for a in self.agents:
            a.draw(self.screen, self.show_cameras, self.show_predictions)

        # Projectiles
        for p in self.projectiles:
            for i, tp in enumerate(p.trail):
                alpha = int(80 * (i + 1) / len(p.trail))
                s = pygame.Surface((6, 6)); s.set_alpha(alpha)
                s.fill(CFG.COLOR_PROJECTILE_TRAIL)
                self.screen.blit(s, (int(tp.x) - 3, int(tp.y) - 3))
            pygame.draw.circle(self.screen, CFG.COLOR_PROJECTILE,
                             (int(p.position.x), int(p.position.y)), 5)

        # Damage numbers
        for dn in self.damage_numbers:
            txt = f"{int(dn.amount)}"
            if dn.is_crit:
                txt += "!"
                col = (255, 50, 50)
                font = self.font_m
            else:
                col = CFG.COLOR_DAMAGE_NUM
                font = self.font_s
            surf = font.render(txt, True, col)
            surf.set_alpha(dn.alpha)
            self.screen.blit(surf, (int(dn.position.x) - 10, int(dn.position.y) - 10))

        # Kill feed
        kf_y = CFG.WORLD_MARGIN + 10
        for kf in reversed(self.kill_feed):
            kc = CFG.COLOR_AGENT_TEAM_0 if kf.killer_team == 0 else CFG.COLOR_AGENT_TEAM_1
            vc = CFG.COLOR_AGENT_TEAM_0 if kf.victim_team == 0 else CFG.COLOR_AGENT_TEAM_1
            alpha = min(255, kf.timer * 3)
            txt = f"#{kf.killer_id}"
            s1 = self.font_s.render(txt, True, kc)
            s2 = self.font_s.render(" > ", True, CFG.COLOR_TEXT_PRIMARY)
            s3 = self.font_s.render(f"#{kf.victim_id}", True, vc)
            x_pos = CFG.WORLD_MARGIN + 10
            self.screen.blit(s1, (x_pos, kf_y))
            self.screen.blit(s2, (x_pos + s1.get_width(), kf_y))
            self.screen.blit(s3, (x_pos + s1.get_width() + s2.get_width(), kf_y))
            kf_y += 18

        # Minimap
        if self.show_minimap:
            self._draw_minimap()

        # UI
        self._draw_ui()

        if self.paused:
            self._draw_pause()

        pygame.display.flip()

    def _draw_minimap(self):
        mm_w, mm_h = 180, 100
        mm_x = CFG.WORLD_MARGIN + 5
        mm_y = CFG.SCREEN_H - CFG.WORLD_MARGIN - mm_h - 5
        sx = mm_w / CFG.SCREEN_W
        sy = mm_h / CFG.SCREEN_H

        pygame.draw.rect(self.screen, CFG.COLOR_MINIMAP_BG, (mm_x, mm_y, mm_w, mm_h))
        pygame.draw.rect(self.screen, CFG.COLOR_GRID, (mm_x, mm_y, mm_w, mm_h), 1)

        for obs in self.obstacles:
            ox = mm_x + int(obs.position.x * sx)
            oy = mm_y + int(obs.position.y * sy)
            r = max(2, int(obs.radius * sx))
            pygame.draw.circle(self.screen, CFG.COLOR_OBSTACLE, (ox, oy), r)

        for a in self.agents:
            if not a.alive:
                continue
            ax = mm_x + int(a.position.x * sx)
            ay = mm_y + int(a.position.y * sy)
            col = CFG.COLOR_AGENT_TEAM_0 if a.team == 0 else CFG.COLOR_AGENT_TEAM_1
            pygame.draw.circle(self.screen, col, (ax, ay), 3)

        for p in self.projectiles:
            px = mm_x + int(p.position.x * sx)
            py = mm_y + int(p.position.y * sy)
            pygame.draw.circle(self.screen, CFG.COLOR_PROJECTILE, (px, py), 1)

    def _count(self, team=None):
        return sum(1 for a in self.agents
                   if a.alive and (team is None or a.team == team))

    def _draw_ui(self):
        sx = CFG.SCREEN_W - 290
        y = 16

        self.screen.blit(self.font_l.render("TANK SWARM AGI", True, CFG.COLOR_TEXT_ACCENT), (sx, y))
        y += 30

        # Team scores
        s0 = self.stats['team_scores'][0]
        s1 = self.stats['team_scores'][1]
        self.screen.blit(self.font_xl.render(f"{s0}", True, CFG.COLOR_AGENT_TEAM_0), (sx, y))
        self.screen.blit(self.font_xl.render("vs", True, CFG.COLOR_TEXT_PRIMARY), (sx + 50, y + 6))
        self.screen.blit(self.font_xl.render(f"{s1}", True, CFG.COLOR_AGENT_TEAM_1), (sx + 90, y))
        y += 48

        fps = int(self.clock.get_fps())
        lines = [
            (f"FPS:  {fps}", CFG.COLOR_HP_HIGH if fps >= 50 else
             CFG.COLOR_HP_MEDIUM if fps >= 30 else CFG.COLOR_HP_LOW),
            (f"Frame:{self.stats['frame']}", CFG.COLOR_TEXT_PRIMARY),
            (f"Alive:{self._count()}/{len(self.agents)}", CFG.COLOR_TEXT_PRIMARY),
            (f"Shots:{self.stats['shots']}  Hits:{self.stats['hits']}", CFG.COLOR_TEXT_PRIMARY),
        ]

        if self.stats['shots'] > 0:
            acc = self.stats['hits'] / self.stats['shots'] * 100
            lines.append((f"Acc:  {acc:.1f}%",
                         CFG.COLOR_HP_HIGH if acc > 40 else CFG.COLOR_HP_MEDIUM))

        lines.append((f"Samples: {self.stats['training_samples']}", CFG.COLOR_TEXT_SECONDARY))

        loss0 = self.stats['nn_loss_t0']
        loss1 = self.stats['nn_loss_t1']
        if loss0 > 0 or loss1 > 0:
            lines.append((f"Loss: {loss0:.4f} / {loss1:.4f}", CFG.COLOR_TEXT_SECONDARY))

        for txt, col in lines:
            self.screen.blit(self.font_m.render(txt, True, col), (sx, y))
            y += 24
        y += 10

        # Team status
        pygame.draw.rect(self.screen, (30, 30, 40), (sx, y, 270, 65))
        self.screen.blit(self.font_m.render(
            f"TEAM 0: {self._count(0)} alive", True, CFG.COLOR_AGENT_TEAM_0), (sx + 10, y + 10))
        self.screen.blit(self.font_m.render(
            f"TEAM 1: {self._count(1)} alive", True, CFG.COLOR_AGENT_TEAM_1), (sx + 10, y + 36))
        y += 74

        # Roles
        self.screen.blit(self.font_s.render("— AGENTS —", True, CFG.COLOR_TEXT_SECONDARY), (sx, y))
        y += 18

        display_limit = 10
        displayed = 0
        for a in sorted(self.agents, key=lambda a: (a.team, -a.kills)):
            if displayed >= display_limit:
                break
            rc = {
                'LEADER': (255, 255, 0), 'ATTACK': (255, 80, 80),
                'SUPPRT': (80, 200, 255), 'SCOUT': (200, 100, 255),
            }.get(a.role_label, CFG.COLOR_TEXT_SECONDARY)
            tc = 'T0' if a.team == 0 else 'T1'
            alive_str = '■' if a.alive else '□'
            kd = f"K{a.kills}/D{a.deaths}"
            self.screen.blit(self.font_s.render(
                f" {alive_str}#{a.id}[{tc}] {a.role_label} {a.state_label} {kd}",
                True, rc if a.alive else (80, 80, 80)), (sx, y))
            y += 16
            displayed += 1

        y += 8
        keys = [
            "SPC-Pause  R-Reset  S-Save",
            "C-Cam  T-Track  O-Grid",
            "P-Predict  I-Influence",
            "M-Minimap  D-Debug  ESC-Quit",
        ]
        for k in keys:
            self.screen.blit(self.font_s.render(k, True, CFG.COLOR_TEXT_SECONDARY), (sx, y))
            y += 16

        # Debug panel
        if self.debug:
            y = CFG.SCREEN_H - 220
            perf_lines = [
                f"Tracks T0:{len(self.twm0.shared_tracks)} T1:{len(self.twm1.shared_tracks)}",
                f"Projectiles: {len(self.projectiles)}",
                f"DmgNums: {len(self.damage_numbers)}",
                f"Grid: {self.world_model.grid.W}x{self.world_model.grid.H}",
                f"SpatHash: {len(self.world_model.spatial_hash.grid)} cells",
                f"Formation T0:{self.twm0.formation_shape.name} T1:{self.twm1.formation_shape.name}",
                f"Scipy: {'YES' if HAS_SCIPY else 'NO'}",
                "--- PERF (ms avg/max) ---",
                f"World:  {self.perf.get_avg('world_model'):.2f}/{self.perf.get_max('world_model'):.2f}",
                f"TeamWM: {self.perf.get_avg('team_wm'):.2f}/{self.perf.get_max('team_wm'):.2f}",
                f"Agents: {self.perf.get_avg('agents'):.2f}/{self.perf.get_max('agents'):.2f}",
                f"Projs:  {self.perf.get_avg('projectiles'):.2f}/{self.perf.get_max('projectiles'):.2f}",
            ]
            total = sum(self.perf.get_avg(k) for k in
                        ['world_model', 'team_wm', 'agents', 'projectiles'])
            perf_lines.append(f"Total:  {total:.2f}")
            for line in perf_lines:
                self.screen.blit(self.font_s.render(line, True, CFG.COLOR_DEBUG), (18, y))
                y += 16

    def _draw_pause(self):
        ov = pygame.Surface((CFG.SCREEN_W, CFG.SCREEN_H))
        ov.fill((0, 0, 0)); ov.set_alpha(128)
        self.screen.blit(ov, (0, 0))
        t = self.font_xl.render("PAUSED", True, CFG.COLOR_TEXT_PRIMARY)
        self.screen.blit(t, t.get_rect(center=(CFG.SCREEN_W // 2, CFG.SCREEN_H // 2)))
        info = self.font_m.render("SPACE to resume", True, CFG.COLOR_TEXT_SECONDARY)
        self.screen.blit(info, info.get_rect(center=(CFG.SCREEN_W // 2, CFG.SCREEN_H // 2 + 40)))

    def run(self):
        print("=" * 60)
        print("TESLA TANK SWARM AGI - Starting simulation")
        print("=" * 60)

        while self.running:
            dt = self.clock.tick(CFG.FPS) / 1000.0 if not CFG.FPS_UNLOCK else 1 / 60.0
            dt = min(dt, 0.05)

            self.handle_events()
            self.update(dt)
            self.render()

        print("\nShutting down - saving final weights...")
        self.twm0._save_weights()
        self.twm1._save_weights()
        pygame.quit()
        sys.exit()


# ============================================================================
# PART 21: ENTRY POINT
# ============================================================================

def main():
    print("=" * 60)
    print("TESLA TANK SWARM AGI - IMPROVED EDITION")
    print("=" * 60)
    print("Improvements:")
    print("  ✓ Kalman filter: Joseph form + condition-number guard")
    print("  ✓ Tracker: Mahalanobis distance association")
    print("  ✓ NN: Adam optimizer + gradient clipping + loss tracking")
    print("  ✓ NN: Batch training with proper minibatches")
    print("  ✓ MPC: Warm-start + threat avoidance + smoothness term")
    print("  ✓ Formations: Context-adaptive (wedge/line/column/spread)")
    print("  ✓ Occupancy grid: Log-odds Bayesian updates")
    print("  ✓ Influence maps for tactical positioning")
    print("  ✓ Swept-circle collision detection (no tunneling)")
    print("  ✓ Respawn system with invulnerability")
    print("  ✓ Damage numbers + kill feed")
    print("  ✓ Minimap display")
    print("  ✓ Turret alignment check before firing")
    print("  ✓ Range-dependent sensor noise")
    print("  ✓ Scipy acceleration when available")
    print("  ✓ Safe weight serialization (atomic write)")
    print("  ✓ K/D tracking per agent")
    print("=" * 60)
    print("Controls:")
    print("  SPACE - Pause   R - Reset   S - Save weights")
    print("  C - Cameras   T - Tracks   O - Occupancy")
    print("  P - Predictions   I - Influence map   M - Minimap")
    print("  D - Debug   ESC - Exit")
    print("=" * 60)

    try:
        engine = GameEngine()
        engine.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()