

#!/usr/bin/env python3
"""
Traci.py — MASAR: Multi-Agent Synchronizer for Adaptive Ramps
MAPPO + Parameter Sharing + GAT Message Passing
King Fahad Highway Ramp Metering — Cooperative Highway-Flow + Queue Control

Architecture
------------
  Agents      : 5 ramp meters  (ramp_1 ... ramp_5)
  Metering    : Real ramp-meter traffic lights with green→yellow→red transitions
  Actions     : 3 discrete  0=Block(red)  1=Slow(18 km/h)  2=Free(100 km/h)
                Yellow is an automatic 3-second intermediate when green→red.
                A safety override forces Free when ramp queue risks spill-back.
  Observation : 6-dim per agent — 4 core MASAR state variables (queue, occ
                downstream, occ upstream, mean speed) plus 2 enhancements
                (released vehicles for throughput context, previous action for
                recurrent memory).  Full breakdown:
                  obs[0] = downstream main-lane occupancy (E3, normalised)
                  obs[1] = ramp queue length (E2, normalised)
                  obs[2] = released ramp vehicles (E1, normalised)
                  obs[3] = previous stop/go action
                  obs[4] = mainline mean speed / free-flow speed
                  obs[5] = upstream main-lane occupancy (prev agent's downstream)
  GAT         : Doubly-linked-chain graph with self-loops (max in/out
                degree = 3 per node — agent i ↔ agents i±1 + self).
                GATEncoder stacks 3 GAT layers for multi-hop message passing.
  Actor       : Shared across agents (parameter sharing)
  Critic      : Centralized — sees all agents' enriched observations (CTDE)
  Reward      : Cooperative scalar  →  − time_in_system / NORM_TIS
                                       − β · ramp_queue / MAX_QUEUE_M
  Fallback    : ramps without E1 detectors use direct TraCI lane queries

Outputs
-------
  mappo_gat_runs/mappo_gat_log.csv           per-step log (all episodes)
  mappo_gat_runs/episode_summary.csv         one row per episode
  mappo_gat_runs/step_images/ep{N}/          per-step diagnostic images
  mappo_gat_runs/comparison_summary.png      MAPPO vs baselines chart
  mappo_gat_runs/weights/                    saved model checkpoints
"""
from __future__ import annotations

import os
import sys
import csv
import math
import random
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow import keras

# ── SUMO/TraCI ────────────────────────────────────────────────────────────────
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please set the SUMO_HOME environment variable.")

import traci  # noqa: E402

# ── Simulation / control ──────────────────────────────────────────────────────
SUMO_CFG      = 'king_fahad_road.sumocfg'
STEP_LEN      = 1.0    # simulation step length (seconds)
CTRL_INTERVAL = 5      # control decision every N sim-seconds


def read_sumocfg_end_time(sumocfg_path: Path, fallback: float = 7200.0) -> float:
    """Read the configured simulation end time from the SUMO config file."""
    try:
        root = ET.parse(sumocfg_path).getroot()
        end_node = root.find("./time/end")
        if end_node is not None:
            value = end_node.attrib.get("value")
            if value is not None:
                return float(value)
    except Exception:
        pass
    return float(fallback)


END_TIME = read_sumocfg_end_time(Path(SUMO_CFG))

# ── Multi-agent topology ──────────────────────────────────────────────────────
N_AGENTS = 5
# obs = [main_occ_dn, queue_len, released, prev_action, main_speed, main_occ_up]
OBS_DIM  = 6
LEGACY_OBS_DIM = 4

# Doubly-linked-chain graph with self-loops: each node connects to itself plus
# its immediate upstream and downstream neighbours (where they exist). This
# matches the physical highway topology — ramp_i's traffic only directly
# influences ramp_{i-1} and ramp_{i+1}. Max in-degree = max out-degree = 3.
def _chain_edges(n: int) -> Tuple[np.ndarray, np.ndarray]:
    src, dst = [], []
    for i in range(n):
        src.append(i); dst.append(i)              # self-loop
        if i - 1 >= 0:
            src.append(i - 1); dst.append(i)      # upstream → i
            src.append(i);     dst.append(i - 1)  # i → upstream  (doubly linked)
        if i + 1 < n:
            src.append(i + 1); dst.append(i)      # downstream → i
            src.append(i);     dst.append(i + 1)  # i → downstream
    # de-duplicate (the cross-edges above add each undirected pair twice)
    seen = set()
    uniq_src, uniq_dst = [], []
    for s, d in zip(src, dst):
        if (s, d) not in seen:
            seen.add((s, d))
            uniq_src.append(s); uniq_dst.append(d)
    return (np.array(uniq_src, dtype=np.int32),
            np.array(uniq_dst, dtype=np.int32))

EDGE_SRC, EDGE_DST = _chain_edges(N_AGENTS)

# ── Action space ──────────────────────────────────────────────────────────────
# Three discrete metering levels controlled by combining the ramp signal with
# a per-lane max-speed limit. Action 0 = block (red); Action 1 = slow metering
# (green + 18 km/h speed cap); Action 2 = free flow (green + 100 km/h).
ACTION_DIM     = 3
SPEED_TABLE    = {0: 0.5, 1: 5.0, 2: 27.78}      # m/s lane.setMaxSpeed value
RATE_VPH_TABLE = {0: 0.0, 1: 360.0, 2: 900.0}    # implied release rate (vph)
ACTION_LABEL   = {0: 'Block', 1: 'Slow(18km/h)', 2: 'Free(100km/h)'}
# Map action → (signal_char, ramp_lane_max_speed_m_s)
ACTION_TO_SIGNAL = {0: 'r', 1: 'g', 2: 'g'}

# ── Agent / detector definitions (ordered north → south by junction y-coord) ──
#
# ramp_e1   : E1 induction-loop ID on the ramp lane
#             Set to None when the ramp has no detector → TraCI lane fallback
# main_e1   : list of 4 E1 IDs on downstream mainline lanes
#             Set to None when not available → TraCI lane fallback
# main_lanes: downstream mainline lane IDs — always set; used for mean-speed
#             measurement via traci.lane.getLastStepMeanSpeed()
#
AGENT_CFG: List[Dict] = [
    # Old project 5-ramp layout (matches init simulation/alinea_TraCI_5ramps.py).
    # The first 3 ramps do not have native signalized ramp-meter junctions in the
    # current network, so evaluation overlays GUI action indicators for them.
    {
        'name'      : 'ramp_1',
        'ramp_lane' : '413583321_0',
        'ramp_e1'   : 'alinea_413583321_E1',
        'ramp_e2'   : 'alinea_413583321_E2',
        'tl_id'     : None,
        'main_e3'   : ['alinea_413583321_E3_0', 'alinea_413583321_E3_1',
                       'alinea_413583321_E3_2', 'alinea_413583321_E3_3'],
        'main_lanes': ['779862593#1-AddedOnRampEdge_0', '779862593#1-AddedOnRampEdge_1',
                       '779862593#1-AddedOnRampEdge_2', '779862593#1-AddedOnRampEdge_3'],
        'signal_mode': 'state',
    },
    {
        'name'      : 'ramp_2',
        'ramp_lane' : '413587595_0',
        'ramp_e1'   : 'alinea_413587595_E1',
        'ramp_e2'   : 'alinea_413587595_E2',
        'tl_id'     : None,
        'main_e3'   : ['alinea_413587595_E3_0', 'alinea_413587595_E3_1',
                       'alinea_413587595_E3_2', 'alinea_413587595_E3_3'],
        'main_lanes': ['779862593#3-AddedOnRampEdge_0', '779862593#3-AddedOnRampEdge_1',
                       '779862593#3-AddedOnRampEdge_2', '779862593#3-AddedOnRampEdge_3'],
        'signal_mode': 'state',
    },
    {
        'name'      : 'ramp_3',
        'ramp_lane' : '1301290506_0',
        'ramp_e1'   : 'alinea_1301290506_E1',
        'ramp_e2'   : 'alinea_1301290506_E2',
        'tl_id'     : None,
        'main_e3'   : ['alinea_1301290506_E3_0', 'alinea_1301290506_E3_1',
                       'alinea_1301290506_E3_2', 'alinea_1301290506_E3_3'],
        'main_lanes': ['266901971#0-AddedOnRampEdge_0', '266901971#0-AddedOnRampEdge_1',
                       '266901971#0-AddedOnRampEdge_2', '266901971#0-AddedOnRampEdge_3'],
        'signal_mode': 'state',
    },
    {
        'name'      : 'ramp_4',
        'ramp_lane' : '92072668_0',
        'ramp_e1'   : 'alinea_92072668_E1',
        'ramp_e2'   : 'alinea_92072668_E2',
        'tl_id'     : None,
        'main_e3'   : ['alinea_92072668_E3_0', 'alinea_92072668_E3_1',
                       'alinea_92072668_E3_2', 'alinea_92072668_E3_3'],
        'main_lanes': ['266901971#2-AddedOnRampEdge_0', '266901971#2-AddedOnRampEdge_1',
                       '266901971#2-AddedOnRampEdge_2', '266901971#2-AddedOnRampEdge_3'],
        'signal_mode': 'state',
    },
    {
        'name'      : 'ramp_5',
        'ramp_lane' : '1300204703_0',
        'ramp_e1'   : 'alinea_1300204703_E1',
        'ramp_e2'   : 'alinea_1300204703_E2',
        'tl_id'     : None,
        'main_e3'   : ['alinea_1300204703_E3_0', 'alinea_1300204703_E3_1',
                       'alinea_1300204703_E3_2', 'alinea_1300204703_E3_3'],
        'main_lanes': ['40924152#3-AddedOnRampEdge_0', '40924152#3-AddedOnRampEdge_1',
                       '40924152#3-AddedOnRampEdge_2', '40924152#3-AddedOnRampEdge_3'],
        'signal_mode': 'state',
    },
]

# TL program defined in ramp_tl.add.xml — phase index == action index
TL_PROGRAM = 'ramp_meter'

# ── Normalisation constants ───────────────────────────────────────────────────
FREE_FLOW_SPEED = 33.33   # m/s ≈ 120 km/h  (Saudi highway design speed)
MAX_RAMP_FLOW   = 20.0    # max released vehicles through ramp per CTRL_INTERVAL
MAX_MAIN_FLOW   = 80.0    # max vehicles across 4 main lanes per CTRL_INTERVAL
MAX_MAIN_OCC    = 100.0   # SUMO occupancy  0–100 %
MAX_QUEUE_M     = 120.0   # queue length normalisation for E2 jam length
TARGET_OCC      = 20.0    # reference downstream occupancy %  (cooperative reward target)
# Time-in-system normaliser: an upper-bound estimate of mean vehicles in
# the network during a control interval. Keeps the reward roughly in [-1, 0].
MAX_TIS_VEHICLES = 1500.0

# Safety constraint (Yang et al. ARM-style override).
# If the ramp queue length passes this threshold, the agent's stop action is
# replaced with a forced 'go' to drain the queue and prevent spill-back onto
# upstream surface streets — addresses the safety-mechanism gap noted in the
# literature review of Deng et al.'s MAPPO ramp-metering work.
SPILLBACK_QUEUE_M     = 100.0
UNSAFE_ACTION_PENALTY = 0.1   # Yang et al. ARM reward-penalty weight for overridden unsafe actions

# ── Neural-network hyper-parameters ──────────────────────────────────────────
GAT_EMBED  = 16
GAT_OUT    = 16
GAT_LAYERS = 3                     # stacked GAT message-passing layers (2-4 ok)
ENRICH_DIM = GAT_EMBED + GAT_OUT   # 32 per agent after GAT

# ── PPO hyper-parameters ─────────────────────────────────────────────────────
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
VALUE_COEF   = 0.5
ENTROPY_COEF = 0.05
LR           = 3e-4
PPO_EPOCHS   = 4
MINIBATCH    = 32

# ── Episode / training config ────────────────────────────────────────────────
# Baselines: each policy runs ALONE across the whole day (one SUMO run per
# baseline, all ramps under the same control method).
BASELINE_TYPES = ['random', 'all_free', 'all_restrict', 'fixed_time', 'alinea']

# MAPPO training is structured as:
#     for day in range(N_EPOCHS):                # outer "day"/epoch loop
#         for episode in range(N_EPISODES_PER_DAY):  # 1-2 hr windows in a day
#             for it in range(N_ITERATIONS):    # PPO updates per episode
#                 train_evaluate()
# Each epoch is one full-day SUMO run; the day is sliced into rollout windows
# (EPISODE_LEN_S long), and at every window boundary we fire N_ITERATIONS
# of PPO updates over the experience collected during that window.
# Quick-debug toggle: MASAR_DEBUG=1 (env var) clamps N_EPOCHS=1,
# N_ITERATIONS=1, EPISODE_LEN_S=3600 so smoke runs finish quickly.
# Baselines still run a full simulated day so comparisons stay meaningful.
DEBUG_MODE = os.environ.get('MASAR_DEBUG', '0') not in ('', '0', 'false', 'False')

if DEBUG_MODE:
    N_EPOCHS         = 1
    EPISODE_LEN_S    = 3600
    N_ITERATIONS     = 1
    CKPT_EVERY_EPOCH = 1
else:
    N_EPOCHS         = 3
    EPISODE_LEN_S    = 3600       # 1-hour episodes (set to 7200 for 2-hour)
    N_ITERATIONS     = 4          # PPO iterations per episode boundary
    CKPT_EVERY_EPOCH = 1          # save a tagged checkpoint every N epochs

N_EPISODES_PER_DAY = max(1, int(END_TIME // EPISODE_LEN_S))

ALINEA_CSV  = Path('init simulation/alinea_log.csv')

# ── Output paths ──────────────────────────────────────────────────────────────
LOG_DIR     = Path('mappo_gat_runs')
IMG_DIR     = LOG_DIR / 'step_images'
CSV_PATH    = LOG_DIR / 'mappo_gat_log.csv'
EP_CSV_PATH = LOG_DIR / 'episode_summary.csv'
WEIGHTS_DIR = LOG_DIR / 'weights'
RESULTS_DIR = Path('results')   # per-method full-day CSVs (baselines + final eval)
LOG_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf',
]

_SIGNAL_OFF    = (50, 50, 50, 255)
_SIGNAL_RED    = (220, 40, 40, 255)
_SIGNAL_YELLOW = (255, 200, 0, 255)
_SIGNAL_GREEN  = (40, 200, 70, 255)
SIGNAL_STATE_BY_ACTION = ACTION_TO_SIGNAL  # back-compat alias

YELLOW_DURATION = 3    # seconds a light stays yellow before going red
UPDATE_INTERVAL = 1    # do a PPO weight update every N control steps (1 = every step)


# ─────────────────────────────────────────────────────────────────────────────
# TraCI measurement helpers  (detector-optional fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _ramp_veh(cfg: Dict) -> int:
    """Vehicles counted on ramp in the last sim step."""
    if cfg['ramp_e1'] is not None:
        return traci.inductionloop.getLastStepVehicleNumber(cfg['ramp_e1'])
    return traci.lane.getLastStepVehicleNumber(cfg['ramp_lane'])


def _ramp_queue_m(cfg: Dict) -> float:
    """Estimated queue length (m) on the ramp over the last sim step."""
    if cfg.get('ramp_e2') is not None:
        return float(traci.lanearea.getJamLengthMeters(cfg['ramp_e2']))
    halted = traci.lane.getLastStepHaltingNumber(cfg['ramp_lane'])
    return float(halted) * 7.5


def _main_flow(cfg: Dict) -> float:
    """Total vehicles on downstream mainline lanes in the last sim step."""
    if cfg.get('main_e3') is not None:
        return float(sum(
            traci.lanearea.getLastStepVehicleNumber(d) for d in cfg['main_e3']))
    return float(sum(
        traci.lane.getLastStepVehicleNumber(l) for l in cfg['main_lanes']))


def _main_occ(cfg: Dict) -> float:
    """Mean occupancy (%) on downstream mainline lanes in the last sim step."""
    if cfg.get('main_e3') is not None:
        return float(np.mean([
            traci.lanearea.getLastStepOccupancy(d) for d in cfg['main_e3']]))
    return float(np.mean([
        traci.lane.getLastStepOccupancy(l) for l in cfg['main_lanes']]))


def _main_speed(cfg: Dict) -> float:
    """Mean speed (m/s) on downstream mainline lanes in the last sim step.
    SUMO returns -1 for lanes with no vehicles; treat those as free-flow."""
    speeds = [traci.lane.getLastStepMeanSpeed(l) for l in cfg['main_lanes']]
    valid  = [s for s in speeds if s >= 0.0]
    return float(np.mean(valid)) if valid else FREE_FLOW_SPEED


def _circle_shape(x: float, y: float, radius: float, n_pts: int = 10):
    return [
        (
            x + radius * math.cos(2.0 * math.pi * k / n_pts),
            y + radius * math.sin(2.0 * math.pi * k / n_pts),
        )
        for k in range(n_pts)
    ]


def _indicator_positions_for_lane(lane_id: str):
    shape = traci.lane.getShape(lane_id)
    if len(shape) < 2:
        x0, y0 = shape[0]
        dx, dy = 1.0, 0.0
    else:
        x0, y0 = shape[0]
        x1, y1 = shape[1]
        dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / norm, dx / norm
    base_x = x0 + dx / norm * 10.0 + nx * 5.0
    base_y = y0 + dy / norm * 10.0 + ny * 5.0
    spacing = 3.2
    return [
        (base_x, base_y + spacing),  # red
        (base_x, base_y),            # amber
        (base_x, base_y - spacing),  # green
    ]


def init_gui_signal_indicators(cfgs: List[Dict]):
    try:
        existing = set(traci.polygon.getIDList())
    except traci.exceptions.TraCIException:
        existing = set()
    for cfg in cfgs:
        positions = _indicator_positions_for_lane(cfg['ramp_lane'])
        cfg['indicator_ids'] = []
        for idx, (x, y) in enumerate(positions):
            poly_id = f"gui_signal_{cfg['name']}_{idx}"
            if poly_id in existing:
                traci.polygon.remove(poly_id)
            traci.polygon.add(
                poly_id,
                _circle_shape(x, y, radius=1.1),
                _SIGNAL_OFF,
                fill=True,
                polygonType='ramp_signal',
                layer=10,
                lineWidth=0.2,
            )
            cfg['indicator_ids'].append(poly_id)


def update_gui_signal_indicator(cfg: Dict, state_char: str):
    """Illuminate the correct circle in the GUI 3-light stack: 'r', 'y', or 'g'."""
    if 'indicator_ids' not in cfg:
        return
    colors = [_SIGNAL_OFF, _SIGNAL_OFF, _SIGNAL_OFF]
    if state_char == 'r':
        colors[0] = _SIGNAL_RED
    elif state_char == 'y':
        colors[1] = _SIGNAL_YELLOW
    else:
        colors[2] = _SIGNAL_GREEN
    for poly_id, color in zip(cfg['indicator_ids'], colors):
        traci.polygon.setColor(poly_id, color)


def apply_signal_state(cfg: Dict, state_char: str):
    """Set ramp signal to 'r', 'y', or 'g' — updates both GUI indicator and SUMO TL."""
    update_gui_signal_indicator(cfg, state_char)
    tl_id = cfg.get('tl_id')
    if tl_id is None:
        return
    if cfg.get('signal_mode') == 'state':
        traci.trafficlight.setRedYellowGreenState(tl_id, state_char)
        return
    # Phase-based fallback: phase 0=red, 1=yellow, 2=green
    phase_map = {'r': 0, 'y': 1, 'g': 2}
    traci.trafficlight.setProgram(tl_id, TL_PROGRAM)
    traci.trafficlight.setPhase(tl_id, phase_map.get(state_char, 0))


def apply_ramp_speed_limit(cfg: Dict, action: int):
    """Set the ramp lane's max speed to the value mapped from `action`.
    Action 0 → very slow / blocked, Action 1 → 18 km/h, Action 2 → 100 km/h."""
    try:
        traci.lane.setMaxSpeed(cfg['ramp_lane'], float(SPEED_TABLE[int(action)]))
    except traci.exceptions.TraCIException:
        pass


def apply_ramp_signal_action(cfg: Dict, action: int):
    """Apply discrete metering action: signal state + ramp-lane speed limit."""
    apply_signal_state(cfg, ACTION_TO_SIGNAL[int(action)])
    apply_ramp_speed_limit(cfg, action)


def process_yellow_transitions(runtime_cfg: List[Dict],
                               signal_chars: List[str],
                               yellow_timers: List[float],
                               dt: float = STEP_LEN):
    """Advance yellow countdown timers by one simulation step (dt seconds).
    When a timer expires the signal flips to red automatically."""
    for i, cfg in enumerate(runtime_cfg):
        if yellow_timers[i] > 0:
            yellow_timers[i] -= dt
            if yellow_timers[i] <= 0:
                yellow_timers[i] = 0.0
                signal_chars[i] = 'r'
                try:
                    apply_signal_state(cfg, 'r')
                except traci.exceptions.TraCIException:
                    pass


def transition_signal(runtime_cfg: List[Dict],
                      i: int,
                      desired_action: int,
                      signal_chars: List[str],
                      yellow_timers: List[float]):
    """Apply a desired metering action for agent i.
    Signal transitions: any green-state → red passes through yellow;
    transitions between Slow (1) and Free (2) are speed-limit-only (signal
    stays green); red → green is immediate. Speed limit always tracks the
    desired action so 1 and 2 are distinguishable on the ramp lane."""
    desired_char = ACTION_TO_SIGNAL[int(desired_action)]
    current = signal_chars[i]

    # Always apply the lane speed limit for the desired action immediately.
    # While in mid-yellow we still want the speed limit lowered toward 0.
    apply_ramp_speed_limit(runtime_cfg[i], desired_action)

    if desired_char == 'r' and current == 'g':
        # Must go yellow first
        signal_chars[i] = 'y'
        yellow_timers[i] = float(YELLOW_DURATION)
        apply_signal_state(runtime_cfg[i], 'y')
    elif desired_char == 'g' and current != 'g':
        # Cancel any pending yellow; go green immediately
        yellow_timers[i] = 0.0
        signal_chars[i] = 'g'
        apply_signal_state(runtime_cfg[i], 'g')
    # same state or already mid-yellow → only the speed-limit was changed


# ─────────────────────────────────────────────────────────────────────────────
# Neural Networks
# ─────────────────────────────────────────────────────────────────────────────

class GATLayer(keras.layers.Layer):
    """Single-head Graph Attention layer over the N-agent fully-connected graph."""

    def __init__(self, out_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.W = self.add_weight(
            shape=(in_dim, self.out_dim), initializer='glorot_uniform', name='W')
        self.a = self.add_weight(
            shape=(2 * self.out_dim,),   initializer='glorot_uniform', name='a')
        super().build(input_shape)

    def call(self, h: tf.Tensor) -> tf.Tensor:
        """h : [N, in_dim]  →  [N, out_dim]"""
        Wh     = h @ self.W                                    # [N, out_dim]
        src_tf = tf.constant(EDGE_SRC)
        dst_tf = tf.constant(EDGE_DST)

        Wh_src = tf.gather(Wh, src_tf)                        # [E, out_dim]
        Wh_dst = tf.gather(Wh, dst_tf)                        # [E, out_dim]

        # Attention logit: LeakyReLU( a^T [Wh_i || Wh_j] )
        cat = tf.concat([Wh_src, Wh_dst], axis=-1)            # [E, 2*out_dim]
        e   = tf.nn.leaky_relu(
                  tf.reduce_sum(cat * self.a, axis=-1), alpha=0.2)  # [E]

        # Per-destination softmax (numerically stable)
        e_max     = tf.math.unsorted_segment_max(e, dst_tf, N_AGENTS)
        e_shifted = e - tf.gather(e_max, dst_tf)
        e_exp     = tf.exp(e_shifted)
        e_sum     = tf.math.unsorted_segment_sum(e_exp, dst_tf, N_AGENTS)
        alpha     = e_exp / (tf.gather(e_sum, dst_tf) + 1e-8)      # [E]

        # Weighted aggregation
        weighted = alpha[:, tf.newaxis] * Wh_src               # [E, out_dim]
        agg      = tf.math.unsorted_segment_sum(weighted, dst_tf, N_AGENTS)
        return tf.nn.elu(agg)                                  # [N, out_dim]

class GATEncoder(keras.Model):
    """Embeds local obs then enriches via stacked GAT message passing.
    Multiple GAT layers (GAT_LAYERS) propagate information across the chain
    so that distant ramps can influence each other through hop-by-hop
    message passing on the doubly-linked graph.
    output = concat(embed, GAT_stack(embed))  →  [N, ENRICH_DIM]"""

    def __init__(self, obs_dim: int, embed_dim: int, gat_out: int,
                 num_layers: int = GAT_LAYERS):
        super().__init__(name='GATEncoder')
        self.embed = keras.layers.Dense(embed_dim, activation='relu', name='embed')
        # First GAT layer maps embed_dim → gat_out, subsequent ones gat_out → gat_out
        self.gat_layers = [
            GATLayer(gat_out, name=f'gat_{k}') for k in range(num_layers)
        ]

    def call(self, obs: tf.Tensor) -> tf.Tensor:
        h = self.embed(obs)              # [N, embed_dim]
        g = h
        for layer in self.gat_layers:
            g = layer(g)                 # [N, gat_out]
        return tf.concat([h, g], -1)     # [N, ENRICH_DIM]


class SharedActor(keras.Model):
    """Shared policy network (parameter sharing across all agents)."""

    def __init__(self, enriched_dim: int, action_dim: int):
        super().__init__(name='SharedActor')
        self.net = keras.Sequential([
            keras.layers.Dense(64, activation='tanh', name='a1'),
            keras.layers.Dense(64, activation='tanh', name='a2'),
            keras.layers.Dense(action_dim,             name='a_out'),
        ], name='actor_net')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.net(x)   # logits  [*, action_dim]


class CentralizedCritic(keras.Model):
    """Centralized critic: sees global state = all agents' enriched observations."""

    def __init__(self, global_dim: int):
        super().__init__(name='CentralizedCritic')
        self.net = keras.Sequential([
            keras.layers.Dense(128, activation='tanh', name='c1'),
            keras.layers.Dense(128, activation='tanh', name='c2'),
            keras.layers.Dense(1,                      name='c_out'),
        ], name='critic_net')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.net(x)   # [*, 1]


# ─────────────────────────────────────────────────────────────────────────────
# MAPPO Agent
# ─────────────────────────────────────────────────────────────────────────────

class MAPPOAgent:

    def __init__(self, obs_dim: int = OBS_DIM):
        self.obs_dim   = int(obs_dim)
        self.encoder   = GATEncoder(self.obs_dim, GAT_EMBED, GAT_OUT)
        self.actor     = SharedActor(ENRICH_DIM, ACTION_DIM)
        self.critic    = CentralizedCritic(N_AGENTS * ENRICH_DIM)
        self.critic_loaded = True
        self.using_legacy_checkpoint = False
        self.optimizer = keras.optimizers.Adam(LR)
        self._build(np.zeros((N_AGENTS, self.obs_dim), dtype=np.float32))

    def _build(self, dummy_obs: np.ndarray):
        enriched = self.encoder(tf.constant(dummy_obs))
        self.actor(enriched)
        self.critic(tf.reshape(enriched, [1, -1]))

    @property
    def all_variables(self) -> list:
        return (self.encoder.trainable_variables
                + self.actor.trainable_variables
                + self.critic.trainable_variables)

    def act(self, obs: np.ndarray) -> Tuple[List[int], List[float], float]:
        """Stochastic action for training. Returns (actions, log_probs, value)."""
        obs_tf   = tf.constant(obs, dtype=tf.float32)
        enriched = self.encoder(obs_tf)
        logits   = self.actor(enriched)
        actions  = tf.squeeze(
            tf.random.categorical(logits, 1, dtype=tf.int32), -1)
        log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions, logits=logits)
        value = (float(tf.squeeze(self.critic(tf.reshape(enriched, [1, -1]))))
                 if self.critic_loaded else 0.0)
        return actions.numpy().tolist(), log_probs.numpy().tolist(), value

    def greedy_act(self, obs: np.ndarray
                   ) -> Tuple[List[int], List[float], float, np.ndarray]:
        """Greedy (argmax) action for evaluation. Returns actions, log_probs, value, probs."""
        obs_tf   = tf.constant(obs, dtype=tf.float32)
        enriched = self.encoder(obs_tf)
        logits   = self.actor(enriched)
        probs    = tf.nn.softmax(logits).numpy()
        actions  = tf.argmax(logits, axis=-1, output_type=tf.int32)
        log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions, logits=logits)
        value = (float(tf.squeeze(self.critic(tf.reshape(enriched, [1, -1]))))
                 if self.critic_loaded else 0.0)
        return actions.numpy().tolist(), log_probs.numpy().tolist(), value, probs

    def get_value(self, obs: np.ndarray) -> float:
        if not self.critic_loaded:
            return 0.0
        obs_tf   = tf.constant(obs, dtype=tf.float32)
        enriched = self.encoder(obs_tf)
        return float(tf.squeeze(self.critic(tf.reshape(enriched, [1, -1]))))

    def save(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        self.encoder.save_weights(str(directory / 'encoder.weights.h5'))
        self.actor.save_weights(str(directory / 'actor.weights.h5'))
        self.critic.save_weights(str(directory / 'critic.weights.h5'))

    @staticmethod
    def _read_h5_datasets(path: Path) -> Dict[str, np.ndarray]:
        import h5py

        arrays: Dict[str, np.ndarray] = {}
        with h5py.File(path, 'r') as h5f:
            def _collect(name, obj):
                if hasattr(obj, 'shape'):
                    arrays[name] = np.array(obj)
            h5f.visititems(_collect)
        return arrays

    def _manual_load_encoder(self, path: Path):
        arrays = self._read_h5_datasets(path)
        self.encoder.embed.set_weights([
            arrays['embed/vars/0'],
            arrays['embed/vars/1'],
        ])
        for k, layer in enumerate(self.encoder.gat_layers):
            # Newer checkpoints use gat_{k}; legacy single-layer checkpoints
            # only had 'gat' — fall back to that for the first layer.
            prefix = f'gat_{k}'
            if f'{prefix}/vars/0' not in arrays and k == 0 and 'gat/vars/0' in arrays:
                prefix = 'gat'
            if f'{prefix}/vars/0' in arrays:
                layer.set_weights([
                    arrays[f'{prefix}/vars/0'],
                    arrays[f'{prefix}/vars/1'],
                ])

    def _manual_load_sequential(self, model: keras.Model, path: Path):
        arrays = self._read_h5_datasets(path)
        dense_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
        for idx, layer in enumerate(dense_layers):
            prefix = f'layers/sequential/layers/dense'
            if idx > 0:
                prefix += f'_{idx}'
            layer.set_weights([
                arrays[f'{prefix}/vars/0'],
                arrays[f'{prefix}/vars/1'],
            ])

    def load(self, directory: Path):
        directory = Path(directory)
        if not (directory / 'encoder.weights.h5').exists():
            raise FileNotFoundError(
                f"No checkpoint at {directory}. Run 'python Traci.py' first.")
        try:
            self.encoder.load_weights(str(directory / 'encoder.weights.h5'))
        except Exception:
            self._manual_load_encoder(directory / 'encoder.weights.h5')
        try:
            self.actor.load_weights(str(directory / 'actor.weights.h5'))
        except Exception as exc:
            try:
                self._manual_load_sequential(self.actor.net, directory / 'actor.weights.h5')
            except Exception as inner_exc:
                raise RuntimeError(
                    "Actor checkpoint is incompatible with the current action space. "
                    f"Current ACTION_DIM={ACTION_DIM}; retrain MAPPO on this ALINEA-matched setup."
                ) from inner_exc
        try:
            self.critic.load_weights(str(directory / 'critic.weights.h5'))
        except Exception:
            try:
                self._manual_load_sequential(self.critic.net, directory / 'critic.weights.h5')
            except Exception as exc:
                self.critic_loaded = False
                print("  [checkpoint warning] critic weights are incompatible with "
                      f"the current setup; continuing without critic values ({exc})")
        self.using_legacy_checkpoint = (self.obs_dim != OBS_DIM) or (not self.critic_loaded)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs:       List[np.ndarray] = []
        self.actions:   List[List[int]]  = []
        self.log_probs: List[List[float]]= []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[float]      = []

    def add(self, obs, actions, log_probs, reward, value, done):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(list(actions))
        self.log_probs.append(list(log_probs))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation."""
        T   = len(self.rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else self.values[t + 1]
            mask   = 1.0 - self.dones[t]
            delta  = self.rewards[t] + GAMMA * next_v * mask - self.values[t]
            gae    = delta + GAMMA * GAE_LAMBDA * mask * gae
            adv[t] = gae
        returns = adv + np.array(self.values, dtype=np.float32)
        # Normalising with a single sample yields adv ≡ 0 (zero gradient), so
        # fall back to raw GAE when the buffer holds just one transition.
        if T > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns


# ─────────────────────────────────────────────────────────────────────────────
# PPO Update
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(agent: MAPPOAgent, buffer: RolloutBuffer,
               last_value: float = 0.0) -> Dict[str, float]:
    """Run PPO_EPOCHS minibatch updates over the rollout buffer.
    Returns the loss components from the most recent minibatch:
        {'loss', 'actor_loss', 'critic_loss', 'entropy'}.
    """
    empty = {'loss': 0.0, 'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
    if len(buffer) == 0:
        return empty

    adv_arr, ret_arr = buffer.compute_gae(last_value=last_value)
    obs_arr = np.array(buffer.obs,       dtype=np.float32)   # [T, N, D]
    act_arr = np.array(buffer.actions,   dtype=np.int32)     # [T, N]
    lp_arr  = np.array(buffer.log_probs, dtype=np.float32)   # [T, N]
    T       = obs_arr.shape[0]
    last_components = dict(empty)

    for _ in range(PPO_EPOCHS):
        idx = np.random.permutation(T)
        for start in range(0, T, MINIBATCH):
            b = idx[start: start + MINIBATCH]
            if len(b) < 1:
                continue
            B     = len(b)
            obs_b = tf.constant(obs_arr[b])
            act_b = tf.constant(act_arr[b])
            lp_b  = tf.constant(lp_arr[b])
            adv_b = tf.constant(adv_arr[b], dtype=tf.float32)
            ret_b = tf.constant(ret_arr[b], dtype=tf.float32)

            with tf.GradientTape() as tape:
                enriched_batch = tf.stack([
                    agent.encoder(obs_b[t]) for t in range(B)
                ])                                            # [B, N, ENRICH_DIM]

                logits    = agent.actor(
                    tf.reshape(enriched_batch, [B * N_AGENTS, ENRICH_DIM]))
                acts_flat = tf.reshape(act_b, [-1])
                new_lp    = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=acts_flat, logits=logits)
                new_lp    = tf.reshape(new_lp, [B, N_AGENTS])

                probs   = tf.nn.softmax(logits)
                entropy = tf.reduce_mean(
                    -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1))

                global_s = tf.reshape(enriched_batch, [B, -1])
                values   = tf.squeeze(agent.critic(global_s), -1)

                ratio   = tf.exp(new_lp - lp_b)
                adv_exp = tf.broadcast_to(adv_b[:, tf.newaxis], [B, N_AGENTS])
                surr1   = ratio * adv_exp
                surr2   = tf.clip_by_value(
                    ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_exp

                actor_loss     = -tf.reduce_mean(tf.minimum(surr1, surr2))
                values_clipped = tf.clip_by_value(
                    values, ret_b - CLIP_EPS, ret_b + CLIP_EPS)
                critic_loss    = tf.reduce_mean(tf.maximum(
                    tf.square(values - ret_b),
                    tf.square(values_clipped - ret_b)))
                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

            grads = tape.gradient(loss, agent.all_variables)
            grads, _ = tf.clip_by_global_norm(grads, 0.5)
            agent.optimizer.apply_gradients(zip(grads, agent.all_variables))
            last_components = {
                'loss':        float(loss),
                'actor_loss':  float(actor_loss),
                'critic_loss': float(critic_loss),
                'entropy':     float(entropy),
            }

    return last_components


# ─────────────────────────────────────────────────────────────────────────────
# Observation & Reward
# ─────────────────────────────────────────────────────────────────────────────

def build_obs(ramp_release_accum: List[float],
              queue_accum:      List[float],
              main_occ_accum:   List[float],
              prev_actions:    List[int],
              main_spd_accum:  Optional[List[float]] = None,
              obs_dim:         int = OBS_DIM) -> np.ndarray:
    """
    Build [N, obs_dim] observation tensor — MASAR per-agent state.

    Per the project specification (Solution §1.4.2), each agent's local view
    contains queue length, mainline occupancy upstream and downstream, and
    average mainline speed. Cross-agent / neighbour information is propagated
    on top of this via the GAT layer.

      obs[:,0] = downstream main-lane occupancy   (E3)
      obs[:,1] = ramp queue length                (E2)
      obs[:,2] = released ramp vehicles           (E1)
      obs[:,3] = previous stop/go action          (recurrent context)
      obs[:,4] = mainline mean speed / free-flow  (NEW — flow indicator)
      obs[:,5] = upstream main-lane occupancy     (NEW — prev agent's E3)

    obs_dim < 6 trims trailing features for backward compat with old checkpoints.
    """
    obs = np.zeros((N_AGENTS, obs_dim), dtype=np.float32)
    for i in range(N_AGENTS):
        avg_occ_pct = main_occ_accum[i] / CTRL_INTERVAL
        avg_queue_m = queue_accum[i] / CTRL_INTERVAL
        obs[i, 0]   = avg_occ_pct / MAX_MAIN_OCC
        obs[i, 1]   = avg_queue_m / MAX_QUEUE_M
        obs[i, 2]   = ramp_release_accum[i] / MAX_RAMP_FLOW
        if obs_dim >= 4:
            obs[i, 3] = prev_actions[i] / max(ACTION_DIM - 1, 1)
        if obs_dim >= 5 and main_spd_accum is not None:
            avg_spd_ms = main_spd_accum[i] / CTRL_INTERVAL
            obs[i, 4]  = avg_spd_ms / FREE_FLOW_SPEED
        if obs_dim >= 6:
            # Upstream context: previous agent's downstream is this agent's
            # upstream. For the first agent (no upstream) reuse own value.
            up_idx = max(i - 1, 0)
            up_occ = main_occ_accum[up_idx] / CTRL_INTERVAL
            obs[i, 5] = up_occ / MAX_MAIN_OCC
    return np.clip(obs, 0.0, 1.0)


def apply_safety_override(actions: List[int], queue_accum: List[float]) -> List[int]:
    """ARM-style safety constraint: when a ramp's queue is approaching
    spill-back, force the action to Free (ACTION_DIM-1) to drain it. The
    override is purely operational — it does not alter policy gradients."""
    out = list(actions)
    free_action = ACTION_DIM - 1
    for i, a in enumerate(actions):
        avg_q = queue_accum[i] / CTRL_INTERVAL
        if avg_q >= SPILLBACK_QUEUE_M and a != free_action:
            out[i] = free_action
    return out


def infer_obs_dim_from_checkpoint(directory: Path, default: int = OBS_DIM) -> int:
    path = Path(directory) / 'encoder.weights.h5'
    if not path.exists():
        return default
    try:
        arrays = MAPPOAgent._read_h5_datasets(path)
        kernel = arrays.get('embed/vars/0')
        if kernel is not None and kernel.ndim == 2:
            return int(kernel.shape[0])
    except Exception:
        pass
    return default


def get_runtime_agent_cfg(verbose: bool = True) -> List[Dict]:
    """Clone AGENT_CFG and gracefully fall back when some runtime IDs are absent."""
    cfgs = copy.deepcopy(AGENT_CFG)

    try:
        loop_ids = set(traci.inductionloop.getIDList())
    except traci.exceptions.TraCIException:
        loop_ids = set()
    try:
        lanearea_ids = set(traci.lanearea.getIDList())
    except traci.exceptions.TraCIException:
        lanearea_ids = set()
    try:
        tl_ids = set(traci.trafficlight.getIDList())
    except traci.exceptions.TraCIException:
        tl_ids = set()
    try:
        lane_ids = set(traci.lane.getIDList())
    except traci.exceptions.TraCIException:
        lane_ids = set()

    for cfg in cfgs:
        if cfg['ramp_lane'] not in lane_ids:
            raise RuntimeError(
                f"Configured ramp lane '{cfg['ramp_lane']}' for {cfg['name']} "
                "does not exist in the loaded SUMO network.")
        missing_main = [lane for lane in cfg['main_lanes'] if lane not in lane_ids]
        if missing_main:
            raise RuntimeError(
                f"Configured mainline lanes missing for {cfg['name']}: {missing_main}")

        if cfg['ramp_e1'] not in loop_ids:
            if verbose:
                print(f"  [detector fallback] {cfg['name']} ramp detector "
                      f"'{cfg['ramp_e1']}' not loaded; using lane counts")
            cfg['ramp_e1'] = None

        if cfg.get('ramp_e2') not in lanearea_ids:
            if verbose:
                print(f"  [detector fallback] {cfg['name']} ramp queue detector "
                      f"'{cfg.get('ramp_e2')}' not loaded; using halted-vehicle estimate")
            cfg['ramp_e2'] = None

        present_main = [det for det in cfg['main_e3'] if det in lanearea_ids]
        if len(present_main) != len(cfg['main_e3']):
            missing_det = [det for det in cfg['main_e3'] if det not in lanearea_ids]
            if verbose:
                print(f"  [detector fallback] {cfg['name']} missing main detectors "
                      f"{missing_det}; using lane metrics")
            cfg['main_e3'] = None

        if cfg['tl_id'] is not None and cfg['tl_id'] not in tl_ids and verbose:
            print(f"  [TL fallback] {cfg['name']} traffic light '{cfg['tl_id']}' "
                  "not loaded; only GUI indicators will be updated")

    if verbose:
        print(f"  Runtime E1 detectors      : {len(loop_ids)}")
        print(f"  Runtime E2/E3 detectors   : {len(lanearea_ids)}")
        print(f"  Runtime traffic lights    : {len(tl_ids)}")
        ramps_with_ramp_e1 = sum(1 for cfg in cfgs if cfg['ramp_e1'] is not None)
        ramps_with_ramp_e2 = sum(1 for cfg in cfgs if cfg.get('ramp_e2') is not None)
        ramps_with_main_e3 = sum(1 for cfg in cfgs if cfg.get('main_e3') is not None)
        print(f"  Ramp release E1 OK        : {ramps_with_ramp_e1}/{len(cfgs)}")
        print(f"  Ramp queue E2 OK          : {ramps_with_ramp_e2}/{len(cfgs)}")
        print(f"  Main detector groups OK   : {ramps_with_main_e3}/{len(cfgs)}")

    return cfgs


QUEUE_REWARD_WEIGHT = 0.5  # β: relative weight of ramp-queue penalty vs TIS

def compute_reward(tis_accum:    float,
                   queue_accum:  List[float],
                   n_unsafe:     int = 0) -> float:
    """
    MASAR cooperative reward — minimises Time In System (TIS) and ramp queues,
    with an additional penalty for unsafe actions (Yang et al. ARM).

    The three terms have minimal inductive bias:
      • TIS = total vehicle-seconds spent in the network during the interval.
        Reducing TIS is equivalent to maximising mainline throughput (since
        Little's law links them via arrivals), so this single signal subsumes
        the previous speed/occupancy proxies without baking in a TARGET_OCC.
      • Ramp queue length — penalises spill-back / unfairness at the ramps.
      • Unsafe actions — penalises steps where the ARM forced an override,
        discouraging repeated unsafe behaviour (Yang et al. reward-penalty ARM).

    All terms are normalised so the per-step reward stays roughly in
    [-(1 + β + γ), 0].  Higher (less negative) = better.

        tis_pen    = mean_vehicles_in_network / MAX_TIS_VEHICLES
        queue_pen  = mean(ramp_queue_m) / MAX_QUEUE_M
        unsafe_pen = n_unsafe / N_AGENTS
        r          = -tis_pen - β·queue_pen - γ·unsafe_pen
    """
    mean_vehicles = float(tis_accum) / max(CTRL_INTERVAL, 1)
    avg_queue     = float(np.mean(queue_accum)) / max(CTRL_INTERVAL, 1)

    tis_pen    = mean_vehicles / MAX_TIS_VEHICLES
    queue_pen  = avg_queue     / MAX_QUEUE_M
    unsafe_pen = n_unsafe / max(N_AGENTS, 1)

    return float(-tis_pen - QUEUE_REWARD_WEIGHT * queue_pen
                 - UNSAFE_ACTION_PENALTY * unsafe_pen)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline policies
# ─────────────────────────────────────────────────────────────────────────────

def fixed_policy(policy_type: str) -> Tuple[List[int], List[float]]:
    if policy_type == 'random':
        acts = [random.randint(0, ACTION_DIM - 1) for _ in range(N_AGENTS)]
    elif policy_type == 'all_free':
        acts = [ACTION_DIM - 1] * N_AGENTS              # action 2 = free flow
    elif policy_type == 'all_restrict':
        acts = [0] * N_AGENTS                           # action 0 = block
    else:
        acts = [ACTION_DIM - 1] * N_AGENTS
    lp = [math.log(1.0 / ACTION_DIM)] * N_AGENTS
    return acts, lp


# ── Stronger (realistic) baselines ───────────────────────────────────────────
# These are stateful controllers that need memory across control steps, so
# they're invoked through `step()` once per ctrl decision rather than via the
# stateless `fixed_policy` helper.

class FixedTimePolicy:
    """Cycles every ramp through a fixed Free → Block duty cycle.
    Default 30 s green / 10 s red simulates a deterministic timer."""

    def __init__(self, green_s: float = 30.0, red_s: float = 10.0):
        self.green_s = float(green_s)
        self.red_s   = float(red_s)
        self.period  = self.green_s + self.red_s

    def step(self, sim_time: float, *_args, **_kw
             ) -> Tuple[List[int], List[float]]:
        phase = sim_time % self.period
        free  = ACTION_DIM - 1
        block = 0
        a     = free if phase < self.green_s else block
        acts  = [a] * N_AGENTS
        lp    = [math.log(1.0 / ACTION_DIM)] * N_AGENTS
        return acts, lp


class ALINEAPolicy:
    """Per-ramp ALINEA occupancy-feedback metering, mapped to the 3-action
    discrete space.  r(k+1) = r(k) + K_R · (o_target − o_meas), clipped to
    [r_min, r_max]; the resulting rate is bucketed into Block/Slow/Free."""

    def __init__(self, kr: float = 70.0, o_target: float = 18.0,
                 r_min: float = 0.0, r_max: float = 900.0):
        self.kr       = float(kr)
        self.o_target = float(o_target)
        self.r_min    = float(r_min)
        self.r_max    = float(r_max)
        self.rates    = [self.r_max] * N_AGENTS

    def step(self, _sim_time: float,
             main_occ_accum: List[float],
             *_args, **_kw) -> Tuple[List[int], List[float]]:
        acts: List[int] = []
        for i in range(N_AGENTS):
            o_meas       = main_occ_accum[i] / max(CTRL_INTERVAL, 1)
            self.rates[i] = float(np.clip(
                self.rates[i] + self.kr * (self.o_target - o_meas),
                self.r_min, self.r_max))
            r = self.rates[i]
            if r < 100.0:
                acts.append(0)              # Block
            elif r < 600.0:
                acts.append(1)              # Slow metering
            else:
                acts.append(ACTION_DIM - 1)  # Free flow
        lp = [math.log(1.0 / ACTION_DIM)] * N_AGENTS
        return acts, lp


def make_baseline_controller(ep_type: str):
    """Returns a stateful controller instance for stronger baselines, or
    None for stateless ones (random / all_free / all_restrict)."""
    if ep_type == 'fixed_time':
        return FixedTimePolicy()
    if ep_type == 'alinea':
        return ALINEAPolicy()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-step Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_step_image(episode:      int,
                    ctrl_step:    int,
                    sim_time:     float,
                    ep_type:      str,
                    rewards_hist: List[float],
                    cum_hist:     List[float],
                    actions_hist: List[List[int]],
                    flows_hist:   List[List[float]],
                    speed_hist:   List[List[float]],
                    occ_hist:     List[List[float]]):
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f'[{ep_type.upper()}]  Ep {episode}  |  Step {ctrl_step}'
        f'  |  t={sim_time:.0f}s',
        fontsize=11, fontweight='bold')
    gs    = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)
    steps = list(range(1, len(rewards_hist) + 1))

    # ── Row 0 ─────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, rewards_hist, color='#4CAF50', linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
    ax.set_title('Step Reward (norm. speed)'); ax.set_xlabel('Ctrl step')
    ax.set_ylabel('Reward'); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, cum_hist, color='#FF9800', linewidth=1.5)
    ax.set_title('Cumulative Reward'); ax.set_xlabel('Ctrl step')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    for ai in range(N_AGENTS):
        acts = [ah[ai] for ah in actions_hist]
        ax.step(steps[:len(acts)], acts, color=_COLORS[ai % len(_COLORS)],
                label=AGENT_CFG[ai]['name'], linewidth=1.0, where='post')
    ax.set_yticks(list(range(ACTION_DIM)))
    ax.set_yticklabels([ACTION_LABEL[a] for a in range(ACTION_DIM)], fontsize=7)
    ax.set_title('Action per Agent')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for ai in range(N_AGENTS):
        ax.plot(steps[:len(flows_hist)], [f[ai] for f in flows_hist],
                color=_COLORS[ai % len(_COLORS)],
                label=AGENT_CFG[ai]['name'], linewidth=1.0)
    ax.set_title('Ramp Flow (veh/interval)'); ax.set_xlabel('Ctrl step')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    for ai in range(N_AGENTS):
        spds_kmh = [s[ai] * 3.6 for s in speed_hist]
        ax.plot(steps[:len(spds_kmh)], spds_kmh,
                color=_COLORS[ai % len(_COLORS)],
                label=AGENT_CFG[ai]['name'], linewidth=1.0)
    ax.axhline(FREE_FLOW_SPEED * 3.6, color='gray', linestyle='--',
               linewidth=0.8, label=f'Free-flow {FREE_FLOW_SPEED*3.6:.0f} km/h')
    ax.set_title('Mean Speed (km/h) — objective'); ax.set_xlabel('Ctrl step')
    ax.set_ylim(0, FREE_FLOW_SPEED * 3.6 * 1.1)
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(ACTION_DIM); w = min(0.8 / max(N_AGENTS, 1), 0.25)
    for ai in range(N_AGENTS):
        counts = [sum(1 for ah in actions_hist if ah[ai] == a)
                  for a in range(ACTION_DIM)]
        ax.bar(x + (ai - (N_AGENTS - 1) / 2.0) * w, counts, w,
               label=AGENT_CFG[ai]['name'],
               color=_COLORS[ai % len(_COLORS)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ACTION_LABEL[a] for a in range(ACTION_DIM)], fontsize=7)
    ax.set_title('Action Distribution')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, axis='y', alpha=0.3)

    ep_dir = IMG_DIR / f'ep{episode:03d}'
    ep_dir.mkdir(exist_ok=True)
    fig.savefig(ep_dir / f'step{ctrl_step:05d}.png', dpi=72, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Training comparison / summary plot
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_plot(summaries: List[Dict]):
    """Six-panel summary across all policies (incl. MAPPO and mappo_eval).
    Panels: cum reward · learning curve · mean TIS · mean queue · throughput
    · mean waiting time. Bar plots aggregate by ep_type."""
    TYPE_COLOR = {
        'random':       '#9E9E9E',
        'all_free':     '#4CAF50',
        'all_restrict': '#F44336',
        'fixed_time':   '#FF9800',
        'alinea':       '#9C27B0',
        'mappo':        '#2196F3',
        'mappo_eval':   '#0D47A1',
    }
    color_for = lambda t: TYPE_COLOR.get(t, '#2196F3')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('MAPPO+GAT vs Baselines — Training & Evaluation Summary',
                 fontsize=13, fontweight='bold')

    episodes = [s['episode']     for s in summaries]
    rewards  = [s['cum_reward']  for s in summaries]
    ep_types = [s['ep_type']     for s in summaries]
    bar_cols = [color_for(t)     for t in ep_types]

    # (0,0) cumulative reward bar by episode
    axes[0, 0].bar(episodes, rewards, color=bar_cols)
    axes[0, 0].set_title('Cumulative Reward per Episode')
    axes[0, 0].set_xlabel('Episode'); axes[0, 0].set_ylabel('Cum reward')
    axes[0, 0].grid(True, axis='y', alpha=0.3)
    patches = [plt.Rectangle((0, 0), 1, 1, color=c, label=k)
               for k, c in TYPE_COLOR.items() if k in set(ep_types)]
    axes[0, 0].legend(handles=patches, fontsize=7, ncol=2)

    # (0,1) MAPPO learning curve overlaid on baselines
    mappo_eps  = [s['episode']     for s in summaries if s['ep_type'] == 'mappo']
    mappo_avg  = [s['mean_reward'] for s in summaries if s['ep_type'] == 'mappo']
    if mappo_eps:
        axes[0, 1].plot(mappo_eps, mappo_avg, color=TYPE_COLOR['mappo'],
                        linewidth=2, marker='o', label='MAPPO (train)')
    for btype in BASELINE_TYPES + ['mappo_eval']:
        vals = [s['mean_reward'] for s in summaries if s['ep_type'] == btype]
        if vals:
            axes[0, 1].axhline(float(np.mean(vals)), linestyle='--',
                               color=color_for(btype), linewidth=1.2,
                               label=f'{btype} ({np.mean(vals):.3f})')
    axes[0, 1].set_title('Mean Reward — MAPPO vs Baselines')
    axes[0, 1].set_xlabel('Episode'); axes[0, 1].set_ylabel('Mean step reward')
    axes[0, 1].legend(fontsize=7); axes[0, 1].grid(True, alpha=0.3)

    # Helper: aggregate `field` by ep_type (mean across episodes for that type)
    def _by_type(field: str):
        out_labels, out_means, out_cols = [], [], []
        for t in BASELINE_TYPES + ['mappo', 'mappo_eval']:
            vals = [float(s.get(field, 0.0))
                    for s in summaries if s['ep_type'] == t]
            if vals:
                out_labels.append(t)
                out_means.append(float(np.mean(vals)))
                out_cols.append(color_for(t))
        return out_labels, out_means, out_cols

    panel_specs = [
        ((0, 2), 'mean_tis',          'Mean Time-In-System (vehicles)'),
        ((1, 0), 'mean_queue',        'Mean Ramp Queue (m)'),
        ((1, 1), 'throughput',        'Throughput (vehicles served)'),
        ((1, 2), 'mean_waiting_time', 'Mean Waiting Time (s/veh)'),
    ]
    for (r, c), field, title in panel_specs:
        labels, means, cols = _by_type(field)
        axes[r, c].bar(labels, means, color=cols)
        axes[r, c].set_title(title)
        axes[r, c].set_ylabel(field)
        axes[r, c].grid(True, axis='y', alpha=0.3)
        axes[r, c].tick_params(axis='x', rotation=20)
        for xi, val in enumerate(means):
            offset = abs(val) * 0.03 + (0.5 if abs(val) > 100 else 0.01)
            axes[r, c].text(xi, val + offset, f'{val:.2g}',
                            ha='center', fontsize=7)

    path = LOG_DIR / 'comparison_summary.png'
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Episode Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(episode:      int,
                ep_type:      str,
                agent:        MAPPOAgent,
                buffer:       RolloutBuffer,
                csv_writer,
                ep_csv_writer,
                epoch:        int = 1,
                eval_mode:    bool = False,
                method_csv_writer = None) -> Dict:
    """Run one full-day episode.
    For ep_type=='mappo' (and not eval_mode), the day is internally
    sub-divided into EPISODE_LEN_S windows; PPO updates fire N_ITERATIONS
    times at each window boundary. Baselines and eval_mode runs do no PPO.
    `method_csv_writer`, if provided, receives the same per-step rows as
    `csv_writer` and is used for results/{method}_day.csv outputs."""
    traci.start([
        'sumo', '-c', SUMO_CFG,
        '--step-length', str(STEP_LEN),
        '--end',         str(END_TIME),
        '--no-warnings', 'true',
        '--quit-on-end', 'false',
    ])

    runtime_cfg = get_runtime_agent_cfg(verbose=(episode == 1))
    init_gui_signal_indicators(runtime_cfg)

    # Initialise all ramp meters to the go (green) phase
    for cfg in runtime_cfg:
        try:
            apply_ramp_signal_action(cfg, 1)
        except traci.exceptions.TraCIException as exc:
            print(f"  [TL init warning] {cfg['name']}: {exc}")

    prev_actions    = [1] * N_AGENTS
    passed_totals   = [0] * N_AGENTS
    ramp_flow_accum = [0.0] * N_AGENTS
    queue_accum     = [0.0] * N_AGENTS
    main_flow_accum = [0.0] * N_AGENTS
    main_occ_accum  = [0.0] * N_AGENTS
    main_spd_accum  = [0.0] * N_AGENTS
    tis_accum       = 0.0   # Σ vehicles_in_network over current CTRL_INTERVAL window
    tis_episode_sum = 0.0   # Σ tis_accum across windows (for mean_tis KPI)
    waiting_sum     = 0.0   # Σ accumulated waiting time (vehicle-seconds)
    sim_steps_accum = 0
    baseline_ctrl   = make_baseline_controller(ep_type)
    is_mappo_train  = (ep_type == 'mappo' and not eval_mode)

    # Traffic-light state tracking for yellow transition
    signal_chars  = ['g'] * N_AGENTS   # actual current signal per ramp
    yellow_timers = [0.0] * N_AGENTS   # countdown (seconds) to flip yellow→red

    rewards_hist : List[float]       = []
    cum_hist     : List[float]       = []
    actions_hist : List[List[int]]   = []
    flows_hist   : List[List[float]] = []
    speed_hist   : List[List[float]] = []
    occ_hist     : List[List[float]] = []
    queue_hist   : List[List[float]] = []  # per-ctrl-step mean queue length per ramp

    # KPI accumulators (Objective 4: Total Travel Time + Avg Queue Length)
    ttt_seconds        = 0.0   # Σ vehicles_in_network × dt   (vehicle-seconds)
    queue_sum_m        = 0.0   # Σ mean ramp queue per ctrl step
    safety_overrides   = 0     # number of agent-steps where ARM forced green

    cum_reward  = 0.0
    ctrl_step   = 0
    last_loss_components = {'loss': 0.0, 'actor_loss': 0.0,
                            'critic_loss': 0.0, 'entropy': 0.0}
    ppo_loss    = 0.0
    steps_since_update = 0
    sub_episode = 0   # rollout-window index within the current day (MAPPO only)
    sim_start   = traci.simulation.getTime()

    print(f"\n{'='*70}")
    if is_mappo_train:
        print(f"  Epoch {epoch}  Episode {episode:3d}  [{ep_type}]  "
              f"(day split into {N_EPISODES_PER_DAY} × {EPISODE_LEN_S}s windows, "
              f"{N_ITERATIONS} PPO iters/window)")
    elif ep_type == 'mappo' and eval_mode:
        print(f"  Episode {episode:3d}  [mappo-eval]  (no learning — full day)")
    else:
        print(f"  Episode {episode:3d}  [{ep_type}]  (baseline — full day)")
    print(f"{'='*70}")
    print(f"  {'t(s)':>6}  {'signals':>12}  {'spd(km/h)':>18}  {'rwd':>8}  {'cum':>10}  {'loss':>10}")
    print(f"  {'-'*75}")

    try:
        while True:
            sim_time = traci.simulation.getTime()
            if sim_time >= END_TIME:
                break

            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                break
            sim_time = traci.simulation.getTime()

            # Advance yellow countdowns before accumulating measurements
            process_yellow_transitions(runtime_cfg, signal_chars, yellow_timers)

            # KPI + reward signal: Total Travel Time = Σ (vehicles in network × dt)
            # waiting_sum (vehicle-seconds of halt) is a SUMO-style delay proxy
            # — sum of halted vehicles per sim step × dt, accumulated across
            # all main + ramp lanes touched by the controlled ramps.
            try:
                vehs_now     = traci.vehicle.getIDCount()
                ttt_seconds += vehs_now * STEP_LEN
                tis_accum   += float(vehs_now)
                halt_now = 0
                for cfg in runtime_cfg:
                    halt_now += traci.lane.getLastStepHaltingNumber(cfg['ramp_lane'])
                    for lane in cfg['main_lanes']:
                        halt_now += traci.lane.getLastStepHaltingNumber(lane)
                waiting_sum += halt_now * STEP_LEN
            except traci.exceptions.TraCIException:
                pass

            # Accumulate per-step measurements for all agents
            for i, cfg in enumerate(runtime_cfg):
                ramp_flow_accum[i] += _ramp_veh(cfg)
                queue_accum[i]     += _ramp_queue_m(cfg)
                main_flow_accum[i] += _main_flow(cfg)
                main_occ_accum[i]  += _main_occ(cfg)
                main_spd_accum[i]  += _main_speed(cfg)

            sim_steps_accum += 1
            if sim_steps_accum < CTRL_INTERVAL:
                continue

            # ── Control decision ──────────────────────────────────────────────
            obs = build_obs(ramp_flow_accum, queue_accum,
                            main_occ_accum, prev_actions,
                            main_spd_accum=main_spd_accum,
                            obs_dim=agent.obs_dim)

            if ep_type == 'mappo':
                if eval_mode:
                    actions, log_probs, value, _ = agent.greedy_act(obs)
                else:
                    actions, log_probs, value = agent.act(obs)
            elif baseline_ctrl is not None:
                actions, log_probs = baseline_ctrl.step(sim_time, main_occ_accum)
                value = agent.get_value(obs)
            else:
                actions, log_probs = fixed_policy(ep_type)
                value = agent.get_value(obs)

            # ARM safety override — force green if a ramp's queue threatens spill-back
            raw_actions = actions[:]
            actions     = apply_safety_override(actions, queue_accum)
            n_unsafe    = sum(1 for a, b in zip(raw_actions, actions) if a != b)
            safety_overrides += n_unsafe

            # Apply signals with yellow transition when going green → red
            for i, cfg in enumerate(runtime_cfg):
                try:
                    transition_signal(runtime_cfg, i, actions[i],
                                      signal_chars, yellow_timers)
                except traci.exceptions.TraCIException as exc:
                    print(f"  [TL warning] {cfg['name']} action={actions[i]}: {exc}")
                passed_totals[i] += int(ramp_flow_accum[i])

            reward     = compute_reward(tis_accum, queue_accum, n_unsafe)
            cum_reward += reward
            done       = (sim_time >= END_TIME - CTRL_INTERVAL)
            tis_episode_sum += tis_accum  # accumulate window-mean for KPI

            # Convert accumulators to per-step averages for logging / plotting
            avg_spd_per_agent   = [main_spd_accum[i] / CTRL_INTERVAL
                                   for i in range(N_AGENTS)]
            avg_occ_per_agent   = [main_occ_accum[i] / CTRL_INTERVAL
                                   for i in range(N_AGENTS)]
            avg_queue_per_agent = [queue_accum[i] / CTRL_INTERVAL
                                   for i in range(N_AGENTS)]

            # KPI: avg queue length contribution this control step
            queue_sum_m += float(np.mean(avg_queue_per_agent))

            rewards_hist.append(reward)
            cum_hist.append(cum_reward)
            actions_hist.append(actions[:])
            flows_hist.append(list(ramp_flow_accum))
            speed_hist.append(avg_spd_per_agent)
            occ_hist.append(avg_occ_per_agent)
            queue_hist.append(avg_queue_per_agent)

            sig_str = '[' + ','.join(signal_chars) + ']'
            spd_kmh_str = str([f'{v*3.6:.1f}' for v in avg_spd_per_agent])
            print(
                f"  {sim_time:>6.0f}  {sig_str:>12}  "
                f"{spd_kmh_str:>18}  "
                f"{reward:>+8.4f}  {cum_reward:>10.2f}  {ppo_loss:>10.6f}"
            )

            # Per-step CSV (one row per agent) — written to both the global
            # log and the optional per-method results file.
            for i, cfg in enumerate(runtime_cfg):
                row = {
                    'episode':          episode,
                    'ep_type':          ep_type,
                    'sim_time':         f'{sim_time:.1f}',
                    'ramp':             cfg['name'],
                    'action':           actions[i],
                    'speed_ms':         SPEED_TABLE[actions[i]],
                    'implied_rate_vph': RATE_VPH_TABLE[actions[i]],
                    'ramp_flow':        f'{ramp_flow_accum[i]:.1f}',
                    'main_flow':        f'{main_flow_accum[i]:.1f}',
                    'main_speed_ms':    f'{avg_spd_per_agent[i]:.4f}',
                    'occ_mean_pct':     f'{avg_occ_per_agent[i]:.3f}',
                    'reward':           f'{reward:.6f}',
                    'cum_reward':       f'{cum_reward:.4f}',
                    'value':            f'{value:.4f}',
                    'passed_total':     passed_totals[i],
                }
                csv_writer.writerow(row)
                if method_csv_writer is not None:
                    method_csv_writer.writerow(row)

            # Per-step diagnostic image
            save_step_image(episode, ctrl_step, sim_time, ep_type,
                            rewards_hist, cum_hist,
                            actions_hist, flows_hist, speed_hist, occ_hist)

            if is_mappo_train:
                buffer.add(obs, actions, log_probs, reward, value, done)
                steps_since_update += 1

                # Episode-boundary PPO trigger: every EPISODE_LEN_S of sim time
                # we close the current rollout window and run N_ITERATIONS of
                # PPO updates over it. This implements the inner two loops of
                #   for episode in range(N_EPISODES_PER_DAY):
                #       for it in range(N_ITERATIONS): train_evaluate()
                # within a single day-long SUMO run.
                episode_boundary = (sim_time >= (sub_episode + 1) * EPISODE_LEN_S
                                    or done)
                if episode_boundary and len(buffer) >= 1:
                    last_val = agent.get_value(
                        np.array(buffer.obs[-1], dtype=np.float32))
                    for _it in range(N_ITERATIONS):
                        last_loss_components = ppo_update(agent, buffer, last_val)
                    ppo_loss = last_loss_components['loss']
                    buffer.reset()
                    steps_since_update = 0
                    agent.save(WEIGHTS_DIR)
                    print(f"    [epoch {epoch} ep {sub_episode + 1}/"
                          f"{N_EPISODES_PER_DAY}] {N_ITERATIONS} PPO iters  "
                          f"loss={ppo_loss:.6f}  "
                          f"actor={last_loss_components['actor_loss']:.4f}  "
                          f"critic={last_loss_components['critic_loss']:.4f}  "
                          f"H={last_loss_components['entropy']:.4f}")
                    sub_episode += 1

            ramp_flow_accum = [0.0] * N_AGENTS
            queue_accum     = [0.0] * N_AGENTS
            main_flow_accum = [0.0] * N_AGENTS
            main_occ_accum  = [0.0] * N_AGENTS
            main_spd_accum  = [0.0] * N_AGENTS
            tis_accum       = 0.0
            sim_steps_accum = 0
            prev_actions    = actions[:]
            ctrl_step      += 1

    finally:
        traci.close()

    # End-of-day flush: drain any remaining experience and persist weights
    if is_mappo_train and len(buffer) >= 1:
        last_val = agent.get_value(np.array(buffer.obs[-1], dtype=np.float32))
        for _it in range(N_ITERATIONS):
            last_loss_components = ppo_update(agent, buffer, last_val)
        ppo_loss = last_loss_components['loss']
        buffer.reset()
        agent.save(WEIGHTS_DIR)
    elif is_mappo_train:
        agent.save(WEIGHTS_DIR)

    avg_reward = float(np.mean(rewards_hist)) if rewards_hist else 0.0
    avg_spd_kmh = (
        float(np.mean([s for step in speed_hist for s in step])) * 3.6
        if speed_hist else 0.0
    )
    # KPIs (Objective 4)
    ttt_hours          = ttt_seconds / 3600.0
    avg_queue_m        = queue_sum_m / max(ctrl_step, 1)
    total_throughput   = int(sum(passed_totals))   # vehicles served via ramps
    mean_waiting_time  = waiting_sum / max(total_throughput, 1)
    mean_tis           = tis_episode_sum / max(ctrl_step, 1) / max(CTRL_INTERVAL, 1)

    summary = {
        'epoch':            epoch,
        'episode':          episode,
        'ep_type':          ep_type,
        'start_time':       float(sim_start),
        'end_time':         float(traci.simulation.getTime()
                                  if not eval_mode else END_TIME),
        'cum_reward':       cum_reward,
        'mean_reward':      avg_reward,
        'avg_reward':       avg_reward,            # back-compat alias
        'mean_tis':         mean_tis,
        'mean_queue':       avg_queue_m,
        'throughput':       total_throughput,
        'mean_waiting_time': mean_waiting_time,
        'mean_speed':       avg_spd_kmh,
        'avg_speed_kmh':    avg_spd_kmh,           # back-compat alias
        'total_passed_r1':  passed_totals[0],
        'total_passed_r2':  passed_totals[1],
        'n_steps':          ctrl_step,
        'ppo_loss':         ppo_loss,
        'actor_loss':       last_loss_components['actor_loss'],
        'critic_loss':      last_loss_components['critic_loss'],
        'entropy':          last_loss_components['entropy'],
        'ttt_hours':        ttt_hours,
        'avg_queue_m':      avg_queue_m,           # back-compat alias
        'safety_overrides': safety_overrides,
    }

    ep_csv_writer.writerow(summary)

    print(f"\n  ↳ cum={cum_reward:.2f}  avg_rwd={avg_reward:.4f}  "
          f"avg_spd={avg_spd_kmh:.1f} km/h  steps={ctrl_step}"
          f"  ppo_loss={ppo_loss:.6f}")
    print(f"    KPI ▸ TTT={ttt_hours:.2f} veh·h   "
          f"mean_TIS={mean_tis:.1f} veh   "
          f"mean_queue={avg_queue_m:.1f} m   "
          f"throughput={total_throughput}   "
          f"mean_wait={mean_waiting_time:.2f}s   "
          f"safety_overrides={safety_overrides}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Summary  (console)
# ─────────────────────────────────────────────────────────────────────────────

def print_alinea_comparison(summaries: List[Dict]):
    """Load the existing ALINEA CSV and print a side-by-side comparison table."""
    if not ALINEA_CSV.exists():
        print(f"  [ALINEA comparison] CSV not found at {ALINEA_CSV} — skipping")
        return

    try:
        al_occ_by_ramp:  Dict[str, List[float]] = {}
        al_passed_final: Dict[str, int]          = {}
        al_rate_by_ramp: Dict[str, List[float]]  = {}

        with open(ALINEA_CSV, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                ramp = row['ramp'].strip()
                al_occ_by_ramp.setdefault(ramp, []).append(float(row['occ_mean_pct']))
                al_rate_by_ramp.setdefault(ramp, []).append(float(row['rate_vph']))
                al_passed_final[ramp] = int(float(row['passed_total']))

        al_avg_occ  = float(np.mean([v for vals in al_occ_by_ramp.values()  for v in vals]))
        al_avg_rate = float(np.mean([v for vals in al_rate_by_ramp.values() for v in vals]))
        al_total_passed = sum(al_passed_final.values())

    except Exception as exc:
        print(f"  [ALINEA comparison] Could not parse CSV: {exc}")
        return

    mappo_sums = [s for s in summaries if s['ep_type'] == 'mappo']
    best_mappo = max(mappo_sums, key=lambda s: s['avg_reward']) if mappo_sums else None

    print("\n" + "=" * 70)
    print("  ALINEA vs MAPPO+GAT — HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<30} {'ALINEA':>12} {'MAPPO (best ep)':>16}")
    print("  " + "-" * 60)

    mappo_occ_str = "—"
    mappo_pass_str = "—"
    if best_mappo:
        mappo_pass_str = str(best_mappo['total_passed_r1'] + best_mappo.get('total_passed_r2', 0))

    print(f"  {'Avg downstream occ (%)':<30} {al_avg_occ:>12.2f} {mappo_occ_str:>16}")
    print(f"  {'Avg metering rate (vph)':<30} {al_avg_rate:>12.1f} {'(stop=0 / go=900)':>16}")
    print(f"  {'Total passed (r1+r2 veh)':<30} {al_total_passed:>12} {mappo_pass_str:>16}")

    if best_mappo:
        print(f"  {'Best MAPPO avg reward':<30} {'—':>12} {best_mappo['avg_reward']:>16.4f}")
        print(f"  {'Best MAPPO avg speed (km/h)':<30} {'—':>12} {best_mappo['avg_speed_kmh']:>16.1f}")

    print("=" * 70)
    print(f"\n  (Full per-step comparison: python evaluate.py  →  eval_results/comparison.png)")
    print()


def print_baseline_summary(summaries: List[Dict]):
    print("\n" + "=" * 95)
    print("  MASAR FINAL COMPARISON SUMMARY")
    print("  (reward = global flow − queue penalty;  KPIs per Objective 4)")
    print("=" * 95)
    hdr = (f"  {'Policy':<14} {'Ep':>4} {'CumRwd':>10} "
           f"{'AvgRwd':>10} {'AvgSpd(km/h)':>13} "
           f"{'TTT(h)':>10} {'Queue(m)':>10}")
    print(hdr)
    print("  " + "-" * 78)
    for s in summaries:
        print(f"  {s['ep_type']:<14} {s['episode']:>4} "
              f"{s['cum_reward']:>10.2f} {s['avg_reward']:>10.4f} "
              f"{s['avg_speed_kmh']:>13.1f} "
              f"{s.get('ttt_hours', 0.0):>10.2f} "
              f"{s.get('avg_queue_m', 0.0):>10.1f}")
    print("=" * 95)
    print()
    for t in BASELINE_TYPES + ['mappo']:
        vals  = [s['avg_reward']     for s in summaries if s['ep_type'] == t]
        spds  = [s['avg_speed_kmh']  for s in summaries if s['ep_type'] == t]
        ttts  = [s.get('ttt_hours', 0.0)   for s in summaries if s['ep_type'] == t]
        qs    = [s.get('avg_queue_m', 0.0) for s in summaries if s['ep_type'] == t]
        if vals:
            print(f"  {t:<14}  avg_rwd={np.mean(vals):.4f}  "
                  f"max_rwd={max(vals):.4f}  "
                  f"avg_spd={np.mean(spds):.1f} km/h  "
                  f"TTT={np.mean(ttts):.2f} h  "
                  f"queue={np.mean(qs):.1f} m")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    print("=" * 70)
    print("  MASAR — Multi-Agent Synchronizer for Adaptive Ramps")
    print("  MAPPO + Parameter Sharing + GAT  |  King Fahad Highway")
    print("  Objective: minimise time-in-system + ramp queues (cooperative)")
    print("=" * 70)
    print(f"  Agents        : {N_AGENTS} ({[c['name'] for c in AGENT_CFG]})")
    print(f"  OBS_DIM       : {OBS_DIM}  "
          f"[occ_dn, queue, released, prev_act, speed, occ_up]")
    print(f"  Action space  : {ACTION_LABEL}")
    print(f"                  (yellow {YELLOW_DURATION}s on any green→red)")
    print(f"  Safety override: forced Free when ramp queue ≥ {SPILLBACK_QUEUE_M:.0f} m")
    print(f"  GAT           : chain topology, {GAT_LAYERS} stacked layers, "
          f"embed={GAT_EMBED} out={GAT_OUT} → enriched={ENRICH_DIM}")
    print(f"  Baselines     : {len(BASELINE_TYPES)}  "
          f"({', '.join(BASELINE_TYPES)})")
    print(f"                  each runs ALONE for one full simulated day")
    print(f"  MAPPO training: {N_EPOCHS} epoch(s) × "
          f"{N_EPISODES_PER_DAY} episode(s)/day × {N_ITERATIONS} PPO iters/episode"
          f"{'  [DEBUG_MODE]' if DEBUG_MODE else ''}")
    print(f"  Day length    : {END_TIME}s  |  episode={EPISODE_LEN_S}s  "
          f"|  ctrl_interval={CTRL_INTERVAL}s")
    print(f"  PPO  γ={GAMMA} λ={GAE_LAMBDA} ε={CLIP_EPS} lr={LR}")
    print(f"  Reward        : -tis_pen - {QUEUE_REWARD_WEIGHT}·queue_pen  "
          f"(no speed/occupancy proxies)")
    print(f"  KPIs reported : reward, mean_tis, mean_queue, throughput, "
          f"mean_waiting_time, mean_speed")
    print(f"  Outputs       : {LOG_DIR}/  +  {RESULTS_DIR}/")
    print()

    agent     = MAPPOAgent()
    buffer    = RolloutBuffer()
    summaries : List[Dict] = []

    # Save initial (untrained) weights immediately so evaluate.py / simulate.py
    # can load *something* even before the first MAPPO update fires. Subsequent
    # online updates will overwrite this with progressively better weights.
    agent.save(WEIGHTS_DIR)
    print(f"  [weights] initial random policy saved → {WEIGHTS_DIR}/")
    print()

    step_csv_fields = [
        'episode', 'ep_type', 'sim_time', 'ramp',
        'action', 'speed_ms', 'implied_rate_vph',
        'ramp_flow', 'main_flow', 'main_speed_ms', 'occ_mean_pct',
        'reward', 'cum_reward', 'value', 'passed_total',
    ]
    ep_csv_fields = [
        'epoch', 'episode', 'ep_type',
        'start_time', 'end_time',
        'cum_reward', 'mean_reward', 'avg_reward',
        'mean_tis', 'mean_queue', 'throughput',
        'mean_waiting_time', 'mean_speed', 'avg_speed_kmh',
        'total_passed_r1', 'total_passed_r2', 'n_steps',
        'ppo_loss', 'actor_loss', 'critic_loss', 'entropy',
        'ttt_hours', 'avg_queue_m', 'safety_overrides',
    ]

    def _open_method_csv(stack, method: str):
        """Open results/{method}_day.csv, register on the exit stack, return writer."""
        path = RESULTS_DIR / f'{method}_day.csv'
        fh   = stack.enter_context(open(path, 'w', newline='', encoding='utf-8'))
        w    = csv.DictWriter(fh, fieldnames=step_csv_fields)
        w.writeheader()
        return w, path

    import contextlib
    with contextlib.ExitStack() as stack:
        sf = stack.enter_context(open(CSV_PATH,    'w', newline='', encoding='utf-8'))
        ef = stack.enter_context(open(EP_CSV_PATH, 'w', newline='', encoding='utf-8'))

        step_writer = csv.DictWriter(sf, fieldnames=step_csv_fields)
        ep_writer   = csv.DictWriter(ef, fieldnames=ep_csv_fields,
                                     extrasaction='ignore')
        step_writer.writeheader()
        ep_writer.writeheader()

        ep_id = 0

        # 1) Baselines — each policy runs ALONE across one full day.
        #    Per-method CSV: results/{method}_day.csv
        for btype in BASELINE_TYPES:
            ep_id += 1
            method_writer, method_path = _open_method_csv(stack, btype)
            summary = run_episode(ep_id, btype, agent, buffer,
                                  step_writer, ep_writer, epoch=0,
                                  method_csv_writer=method_writer)
            summaries.append(summary)
            sf.flush(); ef.flush()
            print(f"  [results] {btype} → {method_path}")

        # 2) MAPPO training — nested epoch × episode × iteration loop.
        for epoch in range(1, N_EPOCHS + 1):
            ep_id += 1
            summary = run_episode(ep_id, 'mappo', agent, buffer,
                                  step_writer, ep_writer, epoch=epoch)
            summaries.append(summary)
            sf.flush(); ef.flush()

            if epoch % CKPT_EVERY_EPOCH == 0:
                ckpt_dir = WEIGHTS_DIR / f'epoch{epoch:03d}'
                agent.save(ckpt_dir)
                print(f"  [checkpoint] saved → {ckpt_dir}")

        # 3) Final MAPPO evaluation — greedy, no learning, full day.
        ep_id += 1
        eval_writer, eval_path = _open_method_csv(stack, 'mappo_eval')
        eval_summary = run_episode(ep_id, 'mappo', agent, buffer,
                                   step_writer, ep_writer, epoch=0,
                                   eval_mode=True,
                                   method_csv_writer=eval_writer)
        eval_summary['ep_type'] = 'mappo_eval'
        summaries.append(eval_summary)
        sf.flush(); ef.flush()
        print(f"  [results] final MAPPO eval → {eval_path}")

    agent.save(WEIGHTS_DIR)
    print(f"\n  [weights] final save → {WEIGHTS_DIR}/")

    print_baseline_summary(summaries)
    print_alinea_comparison(summaries)
    save_comparison_plot(summaries)

    print(f"\n  Done.")
    print(f"  Step log        → {CSV_PATH}")
    print(f"  Episode summary → {EP_CSV_PATH}")
    print(f"  Per-method CSVs → {RESULTS_DIR}/")
    print(f"  Weights         → {WEIGHTS_DIR}/")
    print(f"  Images          → {IMG_DIR}/")
    print(f"  Comparison plot → {LOG_DIR / 'comparison_summary.png'}")
    print(f"\n  To evaluate:  python evaluate.py")


if __name__ == '__main__':
    main()
