"""
HUMAN vs AI  -  TRACK LEARNING CHALLENGE
========================================
A 2D top-view racing game where a human and a reinforcement-learning (Q-learning)
agent drive the same tracks.

Run it
------
    pip install pygame numpy
    python racing_game.py                  # play the game
    python racing_game.py --pretrain 150   # (optional) train the "regular mode" AI headless first
    python racing_game.py --warm           # learning mode: start the AI from the pretrained brain

Modes (chosen in the menu)
--------------------------
1. RACE   : levels 1,2,3 are hand-designed, later levels are procedurally generated.
            You and the (pretrained) AI drive side by side. Beat the AI to score.
2. LEARN  : the "Learning Challenge". Each round = a NEW unfamiliar track. You get an
            attempt, then the AI gets exactly ONE attempt on the same track. The AI starts
            knowing nothing and keeps its experience (Q-table + replay memory) between
            attempts, so you can watch who improves faster.
3. SOLO   : just drive. No AI. (The game is fun on its own.)

Controls
--------
Arrow keys / WASD : steer (left/right), accelerate (up), brake (down)
F                 : change AI fast-forward speed during the AI attempt (1x/4x/16x/64x)
ENTER             : retry / next level      N : brand-new track      ESC : menu

File layout (read top to bottom)
--------------------------------
1. Constants
2. Geometry helpers + Track (road mask, obstacles, sensors, drawing)
3. Car (simple physics)
4. Env  (rules + rewards: shared by human and AI so the competition is fair)
5. QAgent + AIDriver (the reinforcement learning part)
6. Track generators (hand-designed + procedural)
7. Game (pygame UI, modes, results screen, charts)
8. pretrain() and main()
"""
import argparse
import math
import os
import random
import sys

import numpy as np
import pygame

# ----------------------------------------------------------------------------
# 1. CONSTANTS
# ----------------------------------------------------------------------------
W, H = 1000, 700            # window size in pixels
FPS = 60
DT = 1.0 / FPS              # fixed simulation step -> times are measured in sim time,
                            # so AI fast-forwarding does not change its lap time
SPACING = 5.0               # distance between centre-line points (px)

# Car physics (pixels / second)
ACCEL, BRAKE, COAST = 170.0, 320.0, 45.0
MAX_SPEED = 200.0
TURN_RATE = 2.7             # rad/s at full grip

# Sensors ("lidar" rays) the AI sees through. Angles are relative to the car's heading.
RAY_ANGLES = (-70, -35, 0, 35, 70)
RAY_LEN = 120

# Rules
MAX_CRASHES = 5             # an attempt ends when a driver crashes this many times
CRASH_PENALTY = 20.0
FINISH_BONUS = 100.0

# RL hyper-parameters
ACTION_REPEAT = 4           # the agent picks a new action every 4 frames (faster learning)
ALPHA, GAMMA = 0.2, 0.97    # learning rate, discount factor
EPS_MIN = 0.03              # minimum exploration rate
Q_INIT = 5.0                # OPTIMISTIC initial Q-values: untried actions look good, so the agent
                            # is pulled to try them instead of settling for "do nothing"
Q_FILE = "qtable_pretrained.npy"

# Colours
GRASS, ROAD, CURB = (40, 120, 50), (70, 70, 78), (215, 215, 215)
WHITE, BLACK, RED, BLUE, YELLOW = (245, 245, 245), (10, 10, 10), (225, 60, 50), (60, 130, 240), (250, 220, 70)


def wrap_angle(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------------
# 2. GEOMETRY + TRACK
# ----------------------------------------------------------------------------
def catmull_rom(points, samples=24):
    """Turn a few control points into a smooth curve (Catmull-Rom spline)."""
    p = [points[0]] + list(points) + [points[-1]]      # duplicate end points
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = [np.array(q, float) for q in p[i - 1:i + 3]]
        for t in np.linspace(0, 1, samples, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                              (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(np.array(p[-2], float))
    return np.array(out)


def resample(pts, step=SPACING):
    """Re-space the curve so consecutive points are ~`step` px apart.
    Uniform spacing lets us use 'index along the centre line' as a progress measure."""
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    s = np.arange(0, d[-1], step)
    return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])]


class Track:
    """A road defined by a centre line. Start = first point, finish = last point."""
    OBSTACLE_R = 11

    def __init__(self, ctrl_points, width, n_obstacles, rng, name="Track", render=True):
        self.name, self.width = name, width
        self.center = resample(catmull_rom(ctrl_points))
        self.n = len(self.center)
        self.length = self.n * SPACING
        self._build_mask()
        self.obstacles = self._place_obstacles(n_obstacles, rng)
        # Rendering is skipped for headless training (much faster).
        self.surface = self._render() if render else None

    # --- collision mask -----------------------------------------------------
    def _build_mask(self):
        """Draw the road in white on black, then keep it as a boolean array.
        is_road() becomes a cheap array lookup instead of a pixel read."""
        surf = pygame.Surface((W, H))
        surf.fill((0, 0, 0))
        for x, y in self.center:
            pygame.draw.circle(surf, (255, 255, 255), (int(x), int(y)), self.width // 2)
        self.road = pygame.surfarray.array2d(surf) != 0     # shape (W, H)

    def is_road(self, x, y):
        xi, yi = int(x), int(y)
        return 0 <= xi < W and 0 <= yi < H and bool(self.road[xi, yi])

    def hits_obstacle(self, x, y, pad=0):
        r = self.OBSTACLE_R + pad
        return any((x - ox) ** 2 + (y - oy) ** 2 < r * r for ox, oy in self.obstacles)

    def tangent(self, i):
        j = min(i + 3, self.n - 1)
        v = self.center[j] - self.center[max(0, j - 3)]
        return v / (np.hypot(*v) + 1e-9)

    def _place_obstacles(self, k, rng):
        """Spread k obstacles along the road, each pushed to the left or right side
        (30% of the width from the centre) so there is always room to pass."""
        obs = []
        lo, hi = int(0.15 * self.n), int(0.92 * self.n)   # keep start/finish areas clear
        for j in range(k):
            i = int(lo + (j + rng.uniform(0.2, 0.8)) * (hi - lo) / max(k, 1))
            t = self.tangent(i)
            normal = np.array([-t[1], t[0]])
            side = rng.choice((-1, 1))
            x, y = self.center[i] + normal * side * 0.3 * self.width
            obs.append((float(x), float(y)))
        return obs

    # --- sensors --------------------------------------------------------------
    def ray(self, x, y, ang, max_d=RAY_LEN):
        """March along a ray until it leaves the road or touches an obstacle.
        Returns the distance travelled (small = something close)."""
        dx, dy = math.cos(ang), math.sin(ang)
        d = 0.0
        while d < max_d:
            d += 4
            px, py = x + dx * d, y + dy * d
            if not self.is_road(px, py) or self.hits_obstacle(px, py):
                return d
        return max_d

    # --- drawing ---------------------------------------------------------------
    def _marker(self, surf, idx, c1, c2):
        """Checkered start/finish line across the road at centre-line index idx."""
        t = self.tangent(idx)
        normal = np.array([-t[1], t[0]])
        for k, off in enumerate(range(-self.width // 2, self.width // 2, 7)):
            x, y = self.center[idx] + normal * off
            pygame.draw.circle(surf, c1 if k % 2 == 0 else c2, (int(x), int(y)), 4)

    def _render(self):
        surf = pygame.Surface((W, H))
        surf.fill(GRASS)
        for x, y in self.center:                       # curbs (slightly wider circles)
            pygame.draw.circle(surf, CURB, (int(x), int(y)), self.width // 2 + 4)
        for x, y in self.center:                       # asphalt
            pygame.draw.circle(surf, ROAD, (int(x), int(y)), self.width // 2)
        for i in range(0, self.n - 3, 8):              # dashed centre line
            pygame.draw.line(surf, (190, 190, 190), self.center[i], self.center[i + 3], 2)
        self._marker(surf, 1, (60, 220, 90), WHITE)           # start line (green)
        self._marker(surf, self.n - 2, WHITE, BLACK)          # finish line (checkered)
        for ox, oy in self.obstacles:                  # obstacles (cones/barrels)
            pygame.draw.circle(surf, (240, 120, 30), (int(ox), int(oy)), self.OBSTACLE_R)
            pygame.draw.circle(surf, WHITE, (int(ox), int(oy)), self.OBSTACLE_R - 4, 2)
        return surf


# ----------------------------------------------------------------------------
# 3. CAR
# ----------------------------------------------------------------------------
class Car:
    """Very simple arcade car: speed goes up/down, steering turns the heading,
    and you can only steer when moving (grip scales with speed)."""

    def __init__(self, track):
        self.track = track
        self.place(3)

    def place(self, idx):
        """Put the car on the centre line at index idx, pointing along the road, at rest."""
        idx = max(0, min(self.track.n - 2, idx))
        self.idx = idx
        self.x, self.y = self.track.center[idx]
        t = self.track.tangent(idx)
        self.angle = math.atan2(t[1], t[0])
        self.speed = 0.0

    def update(self, steer, throttle):
        """steer: -1 left, 0 straight, +1 right.  throttle: +1 gas, 0 coast, -1 brake."""
        if throttle > 0:
            self.speed += ACCEL * DT
        elif throttle < 0:
            self.speed -= BRAKE * DT
        else:
            self.speed -= COAST * DT                     # rolling friction
        self.speed = max(0.0, min(MAX_SPEED, self.speed))
        grip = min(1.0, self.speed / 60.0)               # can't turn while stopped
        self.angle += steer * TURN_RATE * grip * DT      # y axis points down: +steer = right
        self.x += math.cos(self.angle) * self.speed * DT
        self.y += math.sin(self.angle) * self.speed * DT
        self._update_progress()

    def _update_progress(self):
        """Progress = index of the nearest centre-line point (searched in a small window
        around the previous index, so it can't 'jump' to another part of the track)."""
        c = self.track.center
        lo, hi = max(0, self.idx - 4), min(self.track.n, self.idx + 40)
        d = (c[lo:hi, 0] - self.x) ** 2 + (c[lo:hi, 1] - self.y) ** 2
        self.idx = lo + int(np.argmin(d))

    def heading_error(self):
        """Angle between where the car points and the road direction ~40px ahead.
        (Like a 'next waypoint' arrow. Negative = road bends left.)"""
        tgt = self.track.center[min(self.idx + 8, self.track.n - 1)]
        return wrap_angle(math.atan2(tgt[1] - self.y, tgt[0] - self.x) - self.angle)


# ----------------------------------------------------------------------------
# 4. ENVIRONMENT (the rules; used by BOTH the human and the AI)
# ----------------------------------------------------------------------------
class Env:
    """One attempt on one track.

    Rules:
      * leave the road or hit an obstacle = crash: -20 reward, car respawns on the
        road a little behind (at rest). 5 crashes ends the attempt (failure).
      * reaching the finish ends the attempt (success, +100).
      * running out of time ends the attempt (failure).

    Reward (what the AI optimises):
      +0.2 per centre-line point advanced   -> go forward
      -0.02 per frame                       -> be fast
      -20 per crash, +100 for finishing
    """

    def __init__(self, track):
        self.track = track
        self.car = Car(track)
        self.time_limit = track.length / 70.0 + 10.0       # generous but finite
        self.steps = 0
        self.crashes = 0
        self.total_reward = 0.0
        self.finished = False
        self.over = False
        self.status = "running"
        self.invuln = 0                                    # grace frames after respawn
        self.last_rays = [RAY_LEN] * len(RAY_ANGLES)       # for drawing sensors

    @property
    def time(self):
        return self.steps * DT

    @property
    def progress(self):
        return 100.0 * self.car.idx / (self.track.n - 4)

    def observe(self):
        """What the AI is allowed to see: ray distances, own speed, road direction."""
        c = self.car
        rays = [self.track.ray(c.x, c.y, c.angle + math.radians(a)) for a in RAY_ANGLES]
        self.last_rays = rays
        return rays, c.speed, c.heading_error()

    def _crashed(self):
        c = self.car
        cs, sn = math.cos(c.angle), math.sin(c.angle)
        for off in (11, 0, -11):                           # front, middle, rear of the car
            px, py = c.x + cs * off, c.y + sn * off
            if not self.track.is_road(px, py) or self.track.hits_obstacle(px, py, pad=4):
                return True
        return False

    def step(self, steer, throttle):
        """Advance the simulation one frame. Returns the reward for this frame."""
        car, prev_idx = self.car, self.car.idx
        car.update(steer, throttle)
        self.steps += 1
        r = -0.02 + 0.2 * (car.idx - prev_idx)

        if self.invuln > 0:
            self.invuln -= 1
        elif self._crashed():
            self.crashes += 1
            r -= CRASH_PENALTY
            if self.crashes >= MAX_CRASHES:
                self.over, self.status = True, "crashed out"
            else:
                car.place(car.idx - 6)                     # respawn slightly behind
                self.invuln = 40

        if car.idx >= self.track.n - 4 and not self.over:
            self.over = self.finished = True
            self.status = "finished"
            r += FINISH_BONUS
        elif self.time > self.time_limit and not self.over:
            self.over, self.status = True, "out of time"

        self.total_reward += r
        return r

    def score(self):
        """Game score ('rewards based on performance') shown on the results screen."""
        s = self.progress * 5 - self.crashes * 40
        if self.finished:
            s += 500 + max(0.0, self.time_limit - self.time) * 8
        return int(s)


# ----------------------------------------------------------------------------
# 5. REINFORCEMENT LEARNING: tabular Q-learning
# ----------------------------------------------------------------------------
class QAgent:
    """Q-learning agent.

    STATE (kept small and *relative to the car*, so what is learned on one track
    transfers to unfamiliar tracks):
        5 ray distances, each in 3 bins (near / medium / far)   -> 3^5 = 243
        speed in 3 bins (slow / medium / fast)                  -> x3
        road direction ahead in 3 bins (left / straight / right)-> x3
      = 2187 states.

    ACTIONS: 9 = steer {left, straight, right} x throttle {gas, coast, brake}

    LEARNING: after every decision we apply the Q-learning update
        Q(s,a) += alpha * ( r + gamma * max_a' Q(s',a') - Q(s,a) )
    EXPLORATION: epsilon-greedy - with probability epsilon try a random action.
    MEMORY: every transition is kept in a replay buffer. After each attempt the agent
        "reflects" by re-learning from a random sample of old transitions. Both the
        Q-table and the buffer persist between attempts (and tracks) -> the AI retains
        experience, which makes its improvement comparable to a human's.
    """
    ACTIONS = [(s, t) for t in (1, 0, -1) for s in (-1, 0, 1)]
    N_STATES = 3 ** 5 * 3 * 3

    def __init__(self, epsilon=0.35, learning=True):
        self.q = np.full((self.N_STATES, len(self.ACTIONS)), Q_INIT)
        self.visited = np.zeros(self.N_STATES, bool)        # for the "experience" display
        self.epsilon = epsilon
        self.learning = learning
        self.buffer = []
        self.attempts = 0

    # --- state encoding --------------------------------------------------------
    @staticmethod
    def encode(rays, speed, herr):
        s = 0
        for d in rays:                                       # each ray -> 0,1,2
            s = s * 3 + (0 if d < 30 else 1 if d < 70 else 2)
        s = s * 3 + (0 if speed < 80 else 1 if speed < 150 else 2)
        s = s * 3 + (0 if herr < -0.3 else 2 if herr > 0.3 else 1)
        return s

    # --- acting ----------------------------------------------------------------
    def choose(self, s):
        if self.learning and random.random() < self.epsilon:
            return random.randrange(len(self.ACTIONS))      # explore
        row = self.q[s]
        best = np.flatnonzero(row == row.max())              # exploit (random tie-break)
        return int(random.choice(best))

    # --- learning --------------------------------------------------------------
    def _update(self, s, a, r, s2, done):
        target = r if done else r + GAMMA * self.q[s2].max()
        self.q[s, a] += ALPHA * (target - self.q[s, a])

    def learn(self, s, a, r, s2, done):
        if not self.learning:
            return
        self.visited[s] = True
        self._update(s, a, r, s2, done)
        self.buffer.append((s, a, r, s2, done))
        if len(self.buffer) > 60000:
            self.buffer.pop(random.randrange(len(self.buffer)))

    def end_attempt(self):
        """Called when an attempt finishes: replay memories, then explore a bit less."""
        if self.learning and self.buffer:
            for s, a, r, s2, d in random.sample(self.buffer, min(len(self.buffer), 6000)):
                self._update(s, a, r, s2, d)
        self.attempts += 1
        self.epsilon = max(EPS_MIN, self.epsilon * 0.85)

    # --- persistence -----------------------------------------------------------
    def save(self, path=Q_FILE):
        np.save(path, self.q)

    def load(self, path=Q_FILE):
        if os.path.exists(path):
            self.q = np.load(path)
            self.visited = (self.q != Q_INIT).any(axis=1)
            return True
        return False


class AIDriver:
    """Glue between the QAgent and an Env: decide -> repeat the action a few frames ->
    collect the reward -> learn. Call tick() once per simulation frame."""

    def __init__(self, agent, env):
        self.agent, self.env = agent, env
        self.s = self.a = None
        self.acc_r = 0.0
        self.timer = 0

    def tick(self):
        agent, env = self.agent, self.env
        if self.timer == 0:                                  # time for a new decision
            rays, speed, herr = env.observe()
            s2 = agent.encode(rays, speed, herr)
            if self.s is not None:
                agent.learn(self.s, self.a, self.acc_r, s2, False)
            self.s, self.a = s2, agent.choose(s2)
            self.acc_r, self.timer = 0.0, ACTION_REPEAT
        steer, throttle = agent.ACTIONS[self.a]
        self.acc_r += env.step(steer, throttle)
        self.timer -= 1
        if env.over:                                         # final (terminal) update
            agent.learn(self.s, self.a, self.acc_r, self.s, True)


# ----------------------------------------------------------------------------
# 6. TRACK GENERATORS
# ----------------------------------------------------------------------------
# Hand-designed tracks (control points are smoothed by the spline).
HAND_TRACKS = [
    ("Warm-up Straights", [(80, 350), (250, 350), (400, 300), (550, 330), (700, 380), (850, 350), (930, 350)], 100, 0),
    ("Lazy S", [(80, 600), (250, 560), (380, 400), (500, 300), (620, 350), (740, 500), (870, 380), (930, 200)], 92, 2),
    ("The Serpent", [(80, 120), (300, 130), (450, 200), (520, 350), (420, 500), (550, 620), (750, 600), (850, 450), (780, 300), (900, 180)], 84, 4),
]


def make_procedural_track(rng, level, render=True):
    """Random track: a left-to-right path whose heading does a random walk.
    Higher level = sharper turns, narrower road, more obstacles.
    Always moving rightwards means the road never crosses itself."""
    turn = min(0.35 + 0.08 * level, 1.0)        # max heading change per control point
    x, y = 80.0, rng.uniform(200, H - 200)
    heading, pts = 0.0, [(x, y)]
    while x < W - 110 and len(pts) < 60:
        heading = max(-1.1, min(1.1, heading + rng.uniform(-turn, turn)))
        if y < 130:
            heading = max(heading, 0.35)        # bounce off the top edge
        if y > H - 130:
            heading = min(heading, -0.35)       # bounce off the bottom edge
        step = rng.uniform(70, 95)
        x += math.cos(heading) * step
        y += math.sin(heading) * step
        pts.append((x, y))
    width = max(64, 100 - 4 * level)
    return Track(pts, width, min(1 + level, 10), rng, f"Random track (difficulty {level})", render)


def make_level_track(level, render=True):
    """Regular-mode levels: 1-3 hand-designed, then procedural (seeded so a level is repeatable)."""
    if level <= len(HAND_TRACKS):
        name, pts, width, k = HAND_TRACKS[level - 1]
        return Track(pts, width, k, random.Random(level), name, render)
    return make_procedural_track(random.Random(level * 17), level, render)


# ----------------------------------------------------------------------------
# 7. GAME / UI
# ----------------------------------------------------------------------------
def car_polygon(car):
    """Corners of the car rectangle, rotated to its heading."""
    c, s = math.cos(car.angle), math.sin(car.angle)
    return [(car.x + lx * c - ly * s, car.y + lx * s + ly * c)
            for lx, ly in ((11, 6), (11, -6), (-11, -6), (-11, 6))]


def summarize(env):
    """Turn a finished Env into a record for the results table / charts."""
    return dict(status=env.status, prog=env.progress, time=env.time if env.finished else None,
                crashes=env.crashes, reward=env.total_reward, score=env.score())


class Game:
    SPEEDS = [1, 4, 16, 64]

    def __init__(self, warm=False):
        pygame.init()
        pygame.display.set_caption("Human vs AI - Track Learning Challenge")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.fonts = {s: pygame.font.SysFont(None, s) for s in (20, 26, 34, 60, 90)}
        self.warm = warm
        self.pretrained = QAgent(epsilon=0.0)
        self.has_pretrained = self.pretrained.load()
        self.state = "menu"                      # menu | play | result
        self.speed_idx = 1

    # --- text helper -----------------------------------------------------------
    def text(self, msg, pos, size=26, color=WHITE, center=False):
        img = self.fonts[size].render(msg, True, color)
        rect = img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
        self.screen.blit(img, rect)

    # --- flow control ----------------------------------------------------------
    def start_mode(self, mode):
        self.mode = mode
        self.level = self.round = self.attempt = 0
        self.level, self.round = 1, 1
        self.hist = {"human": [], "ai": []}
        self.total_score = 0
        if mode == "learn":
            self.agent = QAgent(epsilon=0.35)                  # starts with an empty brain...
            if self.warm:
                self.agent.load()                              # ...unless --warm
        else:
            self.agent = QAgent(epsilon=EPS_MIN)
            self.agent.q = self.pretrained.q.copy()            # trained opponent for race mode
            self.agent.visited = self.pretrained.visited.copy()
        self.new_track()

    def new_track(self):
        """Create the track for the next round / level."""
        if self.mode == "race":
            self.track = make_level_track(self.level)
        else:
            self.track = make_procedural_track(random.Random(), min(self.round, 8))
        self.attempt_in_round = 0
        self.begin_attempt()

    def begin_attempt(self):
        """Fresh cars on the SAME track. In learn mode the AI's single attempt follows the human's."""
        self.attempt += 1
        self.attempt_in_round += 1
        self.human_env = Env(self.track)
        self.ai_env = self.ai_driver = None
        if self.mode == "race":
            self.ai_env = Env(self.track)
            self.ai_driver = AIDriver(self.agent, self.ai_env)
        self.phase = "race" if self.mode == "race" else "human"
        self.countdown = 120
        self.state = "play"

    def start_ai_phase(self):
        self.ai_env = Env(self.track)
        self.ai_driver = AIDriver(self.agent, self.ai_env)
        self.phase, self.countdown = "ai", 60

    def finish_attempt(self):
        self.hist["human"].append(summarize(self.human_env))
        self.total_score += self.human_env.score()
        if self.ai_env is not None:
            self.hist["ai"].append(summarize(self.ai_env))
            self.agent.end_attempt()                           # replay memories + decay epsilon
        self.state = "result"

    def winner(self):
        h, a = self.human_env, self.ai_env
        if a is None:
            return "Finished!" if h.finished else f"Attempt failed ({h.status})"
        if h.finished and (not a.finished or h.time < a.time):
            return "YOU WIN!"
        if a.finished and (not h.finished or a.time < h.time):
            return "AI WINS"
        if not h.finished and not a.finished:
            return "Nobody finished - " + ("you got further" if h.progress > a.progress else "AI got further")
        return "Tie"

    # --- per-frame update ------------------------------------------------------
    def update(self):
        if self.state != "play":
            return
        if self.countdown > 0:
            self.countdown -= 1
            return
        if self.phase in ("human", "race"):
            if not self.human_env.over:
                k = pygame.key.get_pressed()
                steer = (k[pygame.K_RIGHT] or k[pygame.K_d]) - (k[pygame.K_LEFT] or k[pygame.K_a])
                throttle = (k[pygame.K_UP] or k[pygame.K_w]) - (k[pygame.K_DOWN] or k[pygame.K_s])
                self.human_env.step(steer, throttle)
            if self.phase == "race":
                # AI drives in real time; once the human is done, fast-forward it to a result.
                for _ in range(8 if self.human_env.over else 1):
                    if not self.ai_env.over:
                        self.ai_driver.tick()
                if self.human_env.over and self.ai_env.over:
                    self.finish_attempt()
            elif self.human_env.over:
                self.start_ai_phase() if self.mode == "learn" else self.finish_attempt()
        elif self.phase == "ai":
            for _ in range(self.SPEEDS[self.speed_idx]):
                if not self.ai_env.over:
                    self.ai_driver.tick()
            if self.ai_env.over:
                self.finish_attempt()

    # --- input -----------------------------------------------------------------
    def handle_key(self, key):
        if key == pygame.K_ESCAPE:
            self.state = "menu"
        elif self.state == "menu":
            if key in (pygame.K_1, pygame.K_2, pygame.K_3):
                self.start_mode({pygame.K_1: "race", pygame.K_2: "learn", pygame.K_3: "solo"}[key])
        elif key == pygame.K_f:
            self.speed_idx = (self.speed_idx + 1) % len(self.SPEEDS)
        elif self.state == "result":
            if key == pygame.K_RETURN:
                if self.mode == "race" and self.human_env.finished:
                    self.level += 1
                    self.new_track()
                else:
                    self.begin_attempt()
            elif key == pygame.K_n and self.mode != "race":
                self.round += 1
                self.new_track()

    # --- drawing ---------------------------------------------------------------
    def draw(self):
        if self.state == "menu":
            return self.draw_menu()
        self.screen.blit(self.track.surface, (0, 0))
        # AI sensors (only while the AI is driving, so you can see what it "sees")
        if self.ai_env is not None and not self.ai_env.over:
            c = self.ai_env.car
            for ang, d in zip(RAY_ANGLES, self.ai_env.last_rays):
                a = c.angle + math.radians(ang)
                pygame.draw.line(self.screen, YELLOW, (c.x, c.y), (c.x + math.cos(a) * d, c.y + math.sin(a) * d), 1)
        if self.phase != "ai":
            self.draw_car(self.human_env, BLUE)
        if self.ai_env is not None:
            self.draw_car(self.ai_env, RED)
        self.draw_hud()
        if self.countdown > 0 and self.state == "play":
            who = "AI's turn - watch it learn" if self.phase == "ai" else "Get ready"
            self.text(who, (W // 2, H // 2 - 60), 60, WHITE, True)
            self.text(str(self.countdown // 30 + 1), (W // 2, H // 2 + 20), 90, YELLOW, True)
        if self.state == "result":
            self.draw_result()

    def draw_car(self, env, color):
        car = env.car
        if env.invuln > 0 and (env.steps // 4) % 2:            # blink after respawn
            return
        pygame.draw.polygon(self.screen, color, car_polygon(car))
        nose = (car.x + math.cos(car.angle) * 11, car.y + math.sin(car.angle) * 11)
        pygame.draw.line(self.screen, WHITE, (car.x, car.y), nose, 3)

    def draw_hud(self):
        bar = pygame.Surface((W, 62), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 150))
        self.screen.blit(bar, (0, 0))
        label = {"race": f"RACE  Level {self.level}", "learn": f"LEARNING CHALLENGE  Round {self.round}",
                 "solo": f"SOLO  Track {self.round}"}[self.mode]
        self.text(f"{label}  |  {self.track.name}  |  Attempt {self.attempt_in_round}", (10, 6), 26)
        h = self.human_env
        self.text(f"YOU  time {h.time:5.1f}s  speed {int(h.car.speed)}  crashes {h.crashes}/{MAX_CRASHES}  "
                  f"progress {h.progress:3.0f}%", (10, 34), 20, BLUE if False else (140, 190, 255))
        if self.ai_env is not None:
            a = self.ai_env
            self.text(f"AI  time {a.time:5.1f}s  crashes {a.crashes}/{MAX_CRASHES}  progress {a.progress:3.0f}%  "
                      f"reward {a.total_reward:7.1f}", (470, 34), 20, (255, 150, 140))
            if self.mode == "learn" or self.phase == "ai":
                seen = int(self.agent.visited.sum())
                self.text(f"AI brain: explore rate {self.agent.epsilon:.2f}  situations seen "
                          f"{seen}/{QAgent.N_STATES}  memories {len(self.agent.buffer)}  "
                          f"fast-forward {self.SPEEDS[self.speed_idx]}x (F)", (470, 8), 20, YELLOW)
        if self.mode == "race" and not self.has_pretrained:
            self.text("No pretrained AI found - run: python racing_game.py --pretrain 150", (10, H - 24), 20, YELLOW)

    def draw_menu(self):
        self.screen.fill((25, 28, 40))
        self.text("HUMAN vs AI", (W // 2, 110), 90, WHITE, True)
        self.text("Track Learning Challenge", (W // 2, 180), 60, YELLOW, True)
        lines = ["1  -  RACE: levels + a trained AI opponent",
                 "2  -  LEARNING CHALLENGE: new tracks, one attempt each. Who learns faster?",
                 "3  -  SOLO DRIVE: just you and the road", "",
                 "Arrows / WASD: steer, gas, brake      F: AI fast-forward      ESC: menu"]
        for i, ln in enumerate(lines):
            self.text(ln, (W // 2, 300 + i * 40), 34 if i < 3 else 26, WHITE, True)
        if not self.has_pretrained:
            self.text("Tip: run 'python racing_game.py --pretrain 150' to give the Race-mode AI a trained brain.",
                      (W // 2, 560), 26, YELLOW, True)

    def draw_result(self):
        box = pygame.Surface((760, 500), pygame.SRCALPHA)
        box.fill((10, 12, 20, 232))
        self.screen.blit(box, (120, 100))
        self.text(self.winner(), (W // 2, 140), 60, YELLOW, True)

        def row(y, name, rec, color):
            t = f"{rec['time']:.1f}s" if rec["time"] is not None else "--"
            self.text(f"{name}: {rec['status']}  |  progress {rec['prog']:.0f}%  |  time {t}  |  "
                      f"crashes {rec['crashes']}  |  score {rec['score']}", (150, y), 26, color)
        row(190, "YOU", self.hist["human"][-1], (140, 190, 255))
        if self.hist["ai"]:
            ai = self.hist["ai"][-1]
            row(225, "AI ", ai, (255, 150, 140))
            self.text(f"AI total reward this attempt: {ai['reward']:.0f}   "
                      f"(it just replayed its memories and will explore less next time)", (150, 258), 20, WHITE)
        self.text(f"Your total score: {self.total_score}", (150, 285), 20, WHITE)
        self.draw_chart(pygame.Rect(150, 320, 700, 190))
        nxt = "ENTER: next level" if self.mode == "race" and self.human_env.finished else "ENTER: retry this track"
        extra = "" if self.mode == "race" else "   N: new track"
        self.text(f"{nxt}{extra}   ESC: menu", (W // 2, 565), 26, WHITE, True)

    def draw_chart(self, rect):
        """Track completion (%) per attempt for human and AI = the learning curves."""
        pygame.draw.rect(self.screen, (30, 34, 48), rect)
        pygame.draw.rect(self.screen, (90, 95, 110), rect, 1)
        self.text("Track completion % per attempt (learning curves)", (rect.x + 8, rect.y + 5), 20)
        plot = pygame.Rect(rect.x + 30, rect.y + 28, rect.w - 45, rect.h - 40)
        for pct in (0, 50, 100):
            y = plot.bottom - plot.h * pct / 100
            pygame.draw.line(self.screen, (60, 64, 80), (plot.x, y), (plot.right, y))
            self.text(str(pct), (rect.x + 4, y - 7), 20)
        for key, color in (("human", BLUE), ("ai", RED)):
            data = [r["prog"] for r in self.hist[key]][-15:]
            pts = [(plot.x + (i + 0.5) * plot.w / 15, plot.bottom - plot.h * v / 100) for i, v in enumerate(data)]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, color, False, pts, 2)
            for p in pts:
                pygame.draw.circle(self.screen, color, p, 4)
        self.text("blue = you", (rect.right - 190, rect.y + 5), 20, (140, 190, 255))
        self.text("red = AI", (rect.right - 90, rect.y + 5), 20, (255, 150, 140))

    # --- main loop ---------------------------------------------------------------
    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if e.type == pygame.KEYDOWN:
                    self.handle_key(e.key)
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)


# ----------------------------------------------------------------------------
# 8. HEADLESS PRETRAINING + MAIN
# ----------------------------------------------------------------------------
def pretrain(episodes, seed=0):
    """Train the AI without graphics on many random tracks (each episode = 1 attempt).
    The saved Q-table is the 'trained AI opponent' used by Race mode."""
    rng = random.Random(seed)
    random.seed(seed)
    agent = QAgent(epsilon=0.5)
    done_count = 0
    for ep in range(episodes):
        agent.epsilon = max(EPS_MIN, 0.5 * (1 - ep / episodes))    # explore a lot, then less
        track = make_procedural_track(rng, rng.randint(1, 6), render=False)
        env = Env(track)
        driver = AIDriver(agent, env)
        while not env.over:
            driver.tick()
        agent.end_attempt()
        done_count += env.finished
        if (ep + 1) % 10 == 0:
            print(f"episode {ep + 1}/{episodes}  finished so far: {done_count}  "
                  f"last: {env.status} ({env.progress:.0f}%)  eps={agent.epsilon:.2f}")
    agent.save()
    print(f"Saved trained brain to {Q_FILE}")


def main():
    ap = argparse.ArgumentParser(description="Human vs AI Track Learning Challenge")
    ap.add_argument("--pretrain", type=int, metavar="N", help="train the AI headless for N episodes, then exit")
    ap.add_argument("--warm", action="store_true", help="learning mode: start the AI from the pretrained brain")
    args = ap.parse_args()
    if args.pretrain:
        pretrain(args.pretrain)
    else:
        Game(warm=args.warm).run()


if __name__ == "__main__":
    main()