#!/usr/bin/env python3
"""
simulate.py — Visual simulation of a trained MAPPO+GAT ramp-metering policy.

Opens SUMO-GUI so you can watch the highway with live traffic lights:
  - Green  : ramp is OPEN  (vehicles enter freely)
  - Yellow : transitioning from green to red  (3-second warning)
  - Red    : ramp is METERED  (vehicles held back)

The policy runs at every control step and the SUMO traffic-light objects
plus GUI polygon indicators are updated in real time.

Usage
-----
    python simulate.py                        # loads best checkpoint, GUI at 1× speed
    python simulate.py --delay 50             # slow down to 50 ms per sim step
    python simulate.py --weights path/        # custom checkpoint directory
    python simulate.py --best                 # auto-select best checkpoint
    python simulate.py --end 3600             # run one simulated hour
    python simulate.py --stochastic           # sample from policy (not argmax)
    python simulate.py --nogui               # headless (for servers without DISPLAY)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# ── SUMO/TraCI ────────────────────────────────────────────────────────────────
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please set the SUMO_HOME environment variable.")

import traci

# ── Import from training module ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import Traci

N_AGENTS        = Traci.N_AGENTS
OBS_DIM         = Traci.OBS_DIM
ACTION_DIM      = Traci.ACTION_DIM
CTRL_INTERVAL   = Traci.CTRL_INTERVAL
STEP_LEN        = Traci.STEP_LEN
SUMO_CFG        = Traci.SUMO_CFG
AGENT_CFG       = Traci.AGENT_CFG
SPEED_TABLE     = Traci.SPEED_TABLE
RATE_VPH_TABLE  = Traci.RATE_VPH_TABLE
ACTION_LABEL    = Traci.ACTION_LABEL
FREE_FLOW_SPEED = Traci.FREE_FLOW_SPEED
TARGET_OCC      = Traci.TARGET_OCC
YELLOW_DURATION   = Traci.YELLOW_DURATION
SPILLBACK_QUEUE_M = Traci.SPILLBACK_QUEUE_M
MAPPOAgent      = Traci.MAPPOAgent
build_obs       = Traci.build_obs
compute_reward  = Traci.compute_reward
apply_safety_override         = Traci.apply_safety_override
infer_obs_dim_from_checkpoint = Traci.infer_obs_dim_from_checkpoint
get_runtime_agent_cfg         = Traci.get_runtime_agent_cfg
init_gui_signal_indicators    = Traci.init_gui_signal_indicators
apply_signal_state            = Traci.apply_signal_state
process_yellow_transitions    = Traci.process_yellow_transitions
transition_signal             = Traci.transition_signal
_ramp_veh       = Traci._ramp_veh
_ramp_queue_m   = Traci._ramp_queue_m
_main_flow      = Traci._main_flow
_main_occ       = Traci._main_occ
_main_speed     = Traci._main_speed

DEFAULT_WEIGHTS  = Path('mappo_gat_runs/weights')
EPISODE_SUMMARY  = Path('mappo_gat_runs/episode_summary.csv')
DEFAULT_END_TIME = Traci.END_TIME
SUMO_SEED        = 42

SIM_DIR = Path('sim_results')
SIM_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers (shared with evaluate.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def find_best_checkpoint(weights_dir: Path) -> Path:
    summaries: dict = {}
    if EPISODE_SUMMARY.exists():
        with open(EPISODE_SUMMARY, newline='') as f:
            for row in csv.DictReader(f):
                summaries[int(row['episode'])] = float(row['avg_reward'])

    candidates: dict = {}
    for d in sorted(weights_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith('ep')):
            continue
        if not (d / 'encoder.weights.h5').exists():
            continue
        try:
            ep_num = int(d.name[2:])
            score  = summaries.get(ep_num + 1, summaries.get(ep_num, -1.0))
            candidates[d] = score
        except ValueError:
            pass

    if (weights_dir / 'encoder.weights.h5').exists():
        last_ep = max(summaries) if summaries else 0
        candidates[weights_dir] = summaries.get(last_ep, -1.0)

    if not candidates:
        return weights_dir

    best       = max(candidates, key=candidates.__getitem__)
    best_score = candidates[best]
    label      = best.name if best != weights_dir else 'final'
    print(f"  Auto-selected checkpoint : {label}  (avg_reward={best_score:.4f})")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Console banner helpers
# ─────────────────────────────────────────────────────────────────────────────

_SIG_ICON = {'r': '🔴', 'y': '🟡', 'g': '🟢'}

def _sig_icon(c: str) -> str:
    return _SIG_ICON.get(c, '⚫')


def print_status(ctrl_step: int, sim_time: float, actions: List[int],
                 signal_chars: List[str], yellow_timers: List[float],
                 avg_spd: List[float], avg_occ: List[float],
                 reward: float, cum_reward: float):
    """Print a live status block for the current control step."""
    os.system('clear' if os.name == 'posix' else 'cls')
    print("=" * 72)
    print("  MAPPO+GAT  |  King Fahad Highway  |  Live Ramp-Meter Simulation")
    print("=" * 72)
    print(f"  Ctrl step : {ctrl_step:>5}     Sim time  : {sim_time:>7.0f} s"
          f"   ({sim_time/3600:.2f} h)")
    print(f"  Reward    : {reward:>+8.4f}    Cum reward: {cum_reward:>10.2f}")
    print()
    print(f"  {'Ramp':<10} {'Signal':>8}  {'Yt':>5}  {'Action':>10}  "
          f"{'Speed':>9}  {'Occ':>7}")
    print("  " + "-" * 60)
    for i, cfg in enumerate(AGENT_CFG):
        sc   = signal_chars[i]
        icon = _sig_icon(sc)
        yt   = f'{yellow_timers[i]:.1f}s' if yellow_timers[i] > 0 else '  —  '
        act  = ACTION_LABEL[actions[i]]
        spd  = f'{avg_spd[i]*3.6:.1f} km/h'
        occ  = f'{avg_occ[i]:.1f}%'
        print(f"  {cfg['name']:<10} {icon} {sc.upper():>4}   {yt:>5}  "
              f"{act:>10}  {spd:>9}  {occ:>7}")
    print()
    print(f"  Free-flow speed: {FREE_FLOW_SPEED*3.6:.0f} km/h  |  "
          f"Mean speed: {np.mean(avg_spd)*3.6:.1f} km/h  |  "
          f"Ctrl interval: {CTRL_INTERVAL} s")
    print("=" * 72)
    print("  (Running SUMO-GUI — interact with the SUMO window to pan/zoom)")


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate(weights_dir: Path, end_time: float, use_gui: bool,
             stochastic: bool, best: bool, delay_ms: int):

    weights_dir = Path(weights_dir)
    if best:
        weights_dir = find_best_checkpoint(weights_dir)

    missing = [f for f in ('encoder.weights.h5', 'actor.weights.h5', 'critic.weights.h5')
               if not (weights_dir / f).exists()]
    if missing:
        print("=" * 60)
        print("ERROR: No trained weights found.")
        print(f"  Expected checkpoint at: {weights_dir.resolve()}")
        print(f"  Missing files: {missing}")
        print()
        print("  Run training first:  python Traci.py")
        print("  Then evaluate first: python evaluate.py")
        print("=" * 60)
        sys.exit(1)

    obs_dim = infer_obs_dim_from_checkpoint(weights_dir)
    agent   = MAPPOAgent(obs_dim=obs_dim)
    try:
        agent.load(weights_dir)
    except RuntimeError as exc:
        print(f"ERROR loading checkpoint: {exc}")
        sys.exit(1)

    tag = 'Stochastic' if stochastic else 'Greedy'
    print()
    print("=" * 72)
    print("  MASAR — Multi-Agent Synchronizer for Adaptive Ramps  (Visual Sim)")
    print("=" * 72)
    print(f"  Checkpoint    : {weights_dir.resolve()}")
    print(f"  Policy mode   : MAPPO+GAT [{tag}]")
    print(f"  Episode length: {end_time:.0f} s  |  ctrl interval: {CTRL_INTERVAL} s")
    print(f"  Step delay    : {delay_ms} ms")
    print(f"  Traffic lights: green → yellow ({YELLOW_DURATION}s) → red")
    print(f"  Safety guard  : ramp queue ≥ {SPILLBACK_QUEUE_M:.0f} m forces go (ARM)")
    print()
    print("  Starting SUMO" + ("-GUI" if use_gui else " (headless)") + " …")

    binary = 'sumo-gui' if use_gui else 'sumo'
    if use_gui and not os.environ.get('DISPLAY'):
        print("  [fallback] DISPLAY not set; running headless.")
        binary = 'sumo'
        use_gui = False

    sumo_cmd = [binary, '-c', SUMO_CFG,
                '--step-length', str(STEP_LEN),
                '--seed',        str(SUMO_SEED),
                '--no-warnings', 'true',
                '--quit-on-end', 'false']
    if use_gui and Path('eval_viewsettings.xml').exists():
        sumo_cmd += ['--gui-settings-file', 'eval_viewsettings.xml']
    if delay_ms > 0 and use_gui:
        sumo_cmd += ['--delay', str(delay_ms)]

    traci.start(sumo_cmd)

    runtime_cfg = get_runtime_agent_cfg(verbose=True)
    init_gui_signal_indicators(runtime_cfg)

    # Start all ramps green
    signal_chars  = ['g'] * N_AGENTS
    yellow_timers = [0.0] * N_AGENTS
    for cfg in runtime_cfg:
        try:
            apply_signal_state(cfg, 'g')
        except traci.exceptions.TraCIException as exc:
            print(f"  [TL init warning] {cfg['name']}: {exc}")

    prev_actions    = [1] * N_AGENTS
    ramp_flow_accum = [0.0] * N_AGENTS
    queue_accum     = [0.0] * N_AGENTS
    main_occ_accum  = [0.0] * N_AGENTS
    main_spd_accum  = [0.0] * N_AGENTS
    sim_steps_accum = 0
    cum_reward      = 0.0
    ctrl_step       = 0
    actions         = [1] * N_AGENTS
    avg_spd         = [FREE_FLOW_SPEED] * N_AGENTS
    avg_occ         = [0.0] * N_AGENTS
    reward          = 0.0

    # MASAR KPIs (Objective 4)
    ttt_seconds      = 0.0
    queue_sum_m      = 0.0
    safety_overrides = 0

    sim_log_path = SIM_DIR / 'sim_log.csv'
    fields = ['ctrl_step', 'sim_time', 'ramp', 'signal', 'action',
              'speed_kmh', 'occ_pct', 'reward', 'cum_reward']

    try:
        with open(sim_log_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=fields)
            writer.writeheader()

            while True:
                sim_time = traci.simulation.getTime()
                if sim_time >= end_time:
                    break

                traci.simulationStep()
                sim_time = traci.simulation.getTime()

                # Advance yellow countdowns
                process_yellow_transitions(runtime_cfg, signal_chars, yellow_timers)

                # KPI: Total Travel Time = Σ vehicles × dt
                try:
                    ttt_seconds += traci.vehicle.getIDCount() * STEP_LEN
                except traci.exceptions.TraCIException:
                    pass

                for i, cfg in enumerate(runtime_cfg):
                    ramp_flow_accum[i] += _ramp_veh(cfg)
                    queue_accum[i]     += _ramp_queue_m(cfg)
                    main_occ_accum[i]  += _main_occ(cfg)
                    main_spd_accum[i]  += _main_speed(cfg)

                sim_steps_accum += 1
                if sim_steps_accum < CTRL_INTERVAL:
                    continue

                # ── Control decision ──────────────────────────────────────────
                obs = build_obs(ramp_flow_accum, queue_accum,
                                main_occ_accum, prev_actions,
                                main_spd_accum=main_spd_accum,
                                obs_dim=obs_dim)

                if stochastic:
                    actions, _, _ = agent.act(obs)
                else:
                    actions, _, _, _ = agent.greedy_act(obs)

                # ARM safety override
                raw_actions = actions[:]
                actions     = apply_safety_override(actions, queue_accum)
                safety_overrides += sum(1 for a, b in zip(raw_actions, actions) if a != b)

                # Apply signals with yellow transition
                for i in range(N_AGENTS):
                    try:
                        transition_signal(runtime_cfg, i, actions[i],
                                          signal_chars, yellow_timers)
                    except traci.exceptions.TraCIException as exc:
                        print(f"  [TL warning] {AGENT_CFG[i]['name']}: {exc}")

                avg_spd   = [main_spd_accum[i] / CTRL_INTERVAL for i in range(N_AGENTS)]
                avg_occ   = [main_occ_accum[i] / CTRL_INTERVAL for i in range(N_AGENTS)]
                avg_queue = [queue_accum[i]    / CTRL_INTERVAL for i in range(N_AGENTS)]
                queue_sum_m += float(np.mean(avg_queue))
                reward  = compute_reward(main_spd_accum, main_occ_accum, queue_accum)
                cum_reward += reward

                # Console display
                print_status(ctrl_step, sim_time, actions, signal_chars, yellow_timers,
                             avg_spd, avg_occ, reward, cum_reward)

                # Log
                for i, cfg in enumerate(runtime_cfg):
                    writer.writerow({
                        'ctrl_step'  : ctrl_step,
                        'sim_time'   : f'{sim_time:.1f}',
                        'ramp'       : cfg['name'],
                        'signal'     : signal_chars[i],
                        'action'     : actions[i],
                        'speed_kmh'  : f'{avg_spd[i]*3.6:.2f}',
                        'occ_pct'    : f'{avg_occ[i]:.2f}',
                        'reward'     : f'{reward:.6f}',
                        'cum_reward' : f'{cum_reward:.4f}',
                    })

                ramp_flow_accum = [0.0] * N_AGENTS
                queue_accum     = [0.0] * N_AGENTS
                main_occ_accum  = [0.0] * N_AGENTS
                main_spd_accum  = [0.0] * N_AGENTS
                sim_steps_accum = 0
                prev_actions    = actions[:]
                ctrl_step      += 1

    finally:
        traci.close()

    ttt_hours   = ttt_seconds / 3600.0
    avg_queue_m = queue_sum_m / max(ctrl_step, 1)
    print()
    print("=" * 72)
    print(f"  Simulation finished — {ctrl_step} control steps")
    print(f"  Cumulative reward    : {cum_reward:.4f}")
    print(f"  Mean highway speed   : {np.mean(avg_spd)*3.6:.1f} km/h")
    print(f"  Total Travel Time    : {ttt_hours:.2f} veh·h   (Objective 4 KPI)")
    print(f"  Avg ramp queue length: {avg_queue_m:.2f} m    (Objective 4 KPI)")
    print(f"  Safety overrides     : {safety_overrides}")
    print(f"  Log saved            → {sim_log_path}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Visual SUMO-GUI simulation of a trained MAPPO+GAT ramp-metering policy. '
                    'Traffic lights show green/yellow/red transitions in real time.')
    p.add_argument('--weights',    default=str(DEFAULT_WEIGHTS),
                   help='Checkpoint directory  (default: mappo_gat_runs/weights)')
    p.add_argument('--best',       action='store_true',
                   help='Auto-select best checkpoint by training proxy')
    p.add_argument('--end',        type=float, default=DEFAULT_END_TIME,
                   help=f'Simulation end time in seconds  (default: {DEFAULT_END_TIME:.0f})')
    p.add_argument('--delay',      type=int,   default=100,
                   help='Milliseconds to pause between simulation steps in GUI (default: 100)')
    p.add_argument('--stochastic', action='store_true',
                   help='Sample from policy distribution instead of argmax')
    p.add_argument('--gui',        dest='gui', action='store_true',
                   help='Open SUMO-GUI (default)')
    p.add_argument('--nogui',      dest='gui', action='store_false',
                   help='Run headless SUMO without GUI')
    p.set_defaults(gui=True)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    simulate(
        weights_dir = Path(args.weights),
        end_time    = args.end,
        use_gui     = args.gui,
        stochastic  = args.stochastic,
        best        = args.best,
        delay_ms    = args.delay,
    )
