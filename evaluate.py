#!/usr/bin/env python3
"""
evaluate.py — Greedy evaluation of a trained MAPPO+GAT ramp-metering policy.

Objective: maximise mean highway speed (matches training reward).

Usage
-----
    python evaluate.py                       # greedy, SUMO-GUI by default, 2-h episode
    python evaluate.py --nogui              # run headless SUMO
    python evaluate.py --end 3600            # run only 1 simulated hour
    python evaluate.py --stochastic          # sample from policy instead of argmax
    python evaluate.py --weights path/       # custom checkpoint directory
    python evaluate.py --best                # auto-select best checkpoint
    python evaluate.py --images              # save a dashboard image every ctrl step

Outputs  (all inside eval_results/)
------
    eval_log.csv            per-step data
    eval_summary.png        full-episode 6-panel dashboard
    comparison.png          MAPPO vs ALINEA overlay  (if ALINEA CSV found)
    step_images/            one dashboard image per control decision  (--images)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── SUMO/TraCI ────────────────────────────────────────────────────────────────
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please set the SUMO_HOME environment variable.")

import traci

# ── Import everything from the training module ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import Traci  # noqa: E402  — imports constants / classes only, not main()

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
_COLORS         = Traci._COLORS
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
ALINEA_CSV       = Path('init simulation/alinea_log.csv')
SUMO_SEED        = 42
DEFAULT_END_TIME = Traci.END_TIME


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint selection
# ─────────────────────────────────────────────────────────────────────────────

def find_best_checkpoint(weights_dir: Path) -> Path:
    """Return the checkpoint with highest avg_reward proxy from episode_summary.csv."""
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
            ep_num  = int(d.name[2:])
            score   = summaries.get(ep_num + 1, summaries.get(ep_num, -1.0))
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
    print(f"  Auto-selected checkpoint : {label}  "
          f"(proxy avg_reward={best_score:.4f})")
    print(f"  Checkpoint path          : {best.resolve()}")
    return best


# ── Output paths ──────────────────────────────────────────────────────────────
EVAL_DIR  = Path('eval_results')
EVAL_IMGS = EVAL_DIR / 'step_images'
EVAL_CSV  = EVAL_DIR / 'eval_log.csv'
EVAL_DIR.mkdir(exist_ok=True)
EVAL_IMGS.mkdir(exist_ok=True)

EVAL_CSV_FIELDS = [
    'episode',
    'ep_type',
    'sim_time',
    'ramp',
    'action',
    'speed_ms',
    'implied_rate_vph',
    'ramp_flow',
    'main_flow',
    'main_speed_ms',
    'occ_mean_pct',
    'reward',
    'cum_reward',
    'value',
    'passed_total',
    'prob_stop',
    'prob_go',
]


# ─────────────────────────────────────────────────────────────────────────────
# Action selection
# ─────────────────────────────────────────────────────────────────────────────

def select_action(agent: MAPPOAgent, obs: np.ndarray, stochastic: bool):
    """Returns (actions, log_probs, value, probs[N, A])."""
    if stochastic:
        import tensorflow as tf
        actions, log_probs, value = agent.act(obs)
        obs_tf   = tf.constant(obs, dtype=tf.float32)
        enriched = agent.encoder(obs_tf)
        logits   = agent.actor(enriched)
        probs    = tf.nn.softmax(logits).numpy()
    else:
        actions, log_probs, value, probs = agent.greedy_act(obs)
    return actions, log_probs, value, probs


# ─────────────────────────────────────────────────────────────────────────────
# Per-step dashboard image
# ─────────────────────────────────────────────────────────────────────────────

def save_step_image(ctrl_step:    int,
                    sim_time:     float,
                    stochastic:   bool,
                    rewards_hist: List[float],
                    cum_hist:     List[float],
                    actions_hist: List[List[int]],
                    flows_hist:   List[List[float]],
                    speed_hist:   List[List[float]],
                    occ_hist:     List[List[float]],
                    rate_hist:    List[List[float]],
                    probs_hist:   List[np.ndarray]):
    tag = 'Stochastic' if stochastic else 'Greedy'
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f'MAPPO+GAT Evaluation [{tag}]  |  ctrl step {ctrl_step}'
        f'  |  sim time {sim_time:.0f} s'
        f'  |  objective: max mean highway speed',
        fontsize=12, fontweight='bold')
    gs    = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.40)
    steps = list(range(1, len(rewards_hist) + 1))

    # ── Row 0: reward ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, rewards_hist, color='#4CAF50', linewidth=1.0, alpha=0.7)
    if len(rewards_hist) >= 5:
        w    = min(10, len(rewards_hist))
        roll = np.convolve(rewards_hist, np.ones(w) / w, mode='valid')
        ax.plot(range(w, len(rewards_hist) + 1), roll,
                color='#1B5E20', linewidth=1.8, label=f'{w}-step avg')
    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
    ax.set_title('Step Reward (norm. speed)'); ax.set_xlabel('Ctrl step')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, cum_hist, color='#FF9800', linewidth=1.5)
    ax.set_title('Cumulative Reward'); ax.set_xlabel('Ctrl step')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    for ai in range(N_AGENTS):
        ax.step(steps[:len(actions_hist)],
                [ah[ai] for ah in actions_hist],
                color=_COLORS[ai], label=AGENT_CFG[ai]['name'],
                linewidth=1.0, where='post')
    ax.set_yticks(list(range(ACTION_DIM)))
    ax.set_yticklabels([ACTION_LABEL[a] for a in range(ACTION_DIM)], fontsize=7)
    ax.set_title('Action per Agent'); ax.set_xlabel('Ctrl step')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # ── Row 1: speed (primary objective), occupancy, metering rate ────────────
    ax = fig.add_subplot(gs[1, 0])
    for ai in range(N_AGENTS):
        spds_kmh = [s[ai] * 3.6 for s in speed_hist]
        ax.plot(steps[:len(spds_kmh)], spds_kmh,
                color=_COLORS[ai], label=AGENT_CFG[ai]['name'],
                linewidth=1.0, alpha=0.85)
    ax.axhline(FREE_FLOW_SPEED * 3.6, color='gray', linestyle='--',
               linewidth=0.9,
               label=f'Free-flow {FREE_FLOW_SPEED*3.6:.0f} km/h')
    ax.set_title('Mean Speed (km/h) — PRIMARY OBJECTIVE')
    ax.set_xlabel('Ctrl step'); ax.set_ylim(0, FREE_FLOW_SPEED * 3.6 * 1.1)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    for ai in range(N_AGENTS):
        ax.plot(steps[:len(occ_hist)], [o[ai] for o in occ_hist],
                color=_COLORS[ai], label=AGENT_CFG[ai]['name'],
                linewidth=1.0, alpha=0.85)
    ax.axhline(TARGET_OCC, color='gray', linestyle='--', linewidth=0.8,
               label=f'Ref {TARGET_OCC}%')
    ax.set_title('Main-lane Occupancy (%)'); ax.set_xlabel('Ctrl step')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    for ai in range(N_AGENTS):
        ax.plot(steps[:len(rate_hist)], [r[ai] for r in rate_hist],
                color=_COLORS[ai], label=AGENT_CFG[ai]['name'],
                linewidth=1.0, alpha=0.85, drawstyle='steps-post')
    ax.set_title('Implied Metering Rate (veh/h)')
    ax.set_xlabel('Ctrl step'); ax.set_ylim(-50, 1000)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # ── Row 2: policy analysis ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    x = np.arange(ACTION_DIM); w = min(0.8 / max(N_AGENTS, 1), 0.25)
    for ai in range(N_AGENTS):
        counts = [sum(1 for ah in actions_hist if ah[ai] == a)
                  for a in range(ACTION_DIM)]
        ax.bar(x + (ai - (N_AGENTS - 1) / 2.0) * w, counts, w,
               label=AGENT_CFG[ai]['name'], color=_COLORS[ai], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ACTION_LABEL[a] for a in range(ACTION_DIM)], fontsize=7)
    ax.set_title('Action Distribution (so far)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, axis='y', alpha=0.3)

    # Policy probability heatmap (last step)
    ax = fig.add_subplot(gs[2, 1])
    if probs_hist:
        last_p = probs_hist[-1]   # [N, A]
        im = ax.imshow(last_p, aspect='auto', cmap='YlGn', vmin=0, vmax=1)
        ax.set_xticks(range(ACTION_DIM))
        ax.set_xticklabels([ACTION_LABEL[a] for a in range(ACTION_DIM)], fontsize=7)
        ax.set_yticks(range(N_AGENTS))
        ax.set_yticklabels([c['name'] for c in AGENT_CFG], fontsize=7)
        ax.set_title('Policy Probs (this step)')
        for (r, c), v in np.ndenumerate(last_p):
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=9,
                    color='black' if v < 0.7 else 'white')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # KPI summary text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    if rewards_hist:
        la  = actions_hist[-1] if actions_hist else ['?'] * N_AGENTS
        ls  = speed_hist[-1]   if speed_hist   else [0.0] * N_AGENTS
        lo  = occ_hist[-1]     if occ_hist     else [0.0] * N_AGENTS
        lr  = rate_hist[-1]    if rate_hist    else [0.0] * N_AGENTS
        avg_spd_now = np.mean(ls) * 3.6
        lines = [
            f'Policy     : MAPPO+GAT [{tag}]',
            f'Ctrl step  : {ctrl_step}',
            f'Sim time   : {sim_time:.0f} s',
            '',
            f'Step rwd   : {rewards_hist[-1]:+.4f}',
            f'Cum. rwd   : {cum_hist[-1]:.2f}',
            f'Avg rwd    : {np.mean(rewards_hist):.4f}',
            f'Best step  : {max(rewards_hist):+.4f}',
            f'Mean spd   : {avg_spd_now:.1f} km/h',
            '',
        ]
        for ai in range(N_AGENTS):
            lines += [
                f'{AGENT_CFG[ai]["name"]}:',
                f'  act={la[ai]}  rate={lr[ai]:.0f} vph',
                f'  spd={ls[ai]*3.6:.1f} km/h  occ={lo[ai]:.1f}%',
            ]
        ax.text(0.04, 0.97, '\n'.join(lines), transform=ax.transAxes,
                fontsize=7.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))

    fig.savefig(EVAL_IMGS / f'step{ctrl_step:05d}.png', dpi=72, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Final summary plot
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_plot(rewards_hist:  List[float],
                      cum_hist:      List[float],
                      actions_hist:  List[List[int]],
                      flows_hist:    List[List[float]],
                      speed_hist:    List[List[float]],
                      occ_hist:      List[List[float]],
                      rate_hist:     List[List[float]],
                      passed_totals: List[int],
                      stochastic:    bool):
    tag   = 'Stochastic' if stochastic else 'Greedy'
    T     = len(rewards_hist)
    steps = list(range(1, T + 1))
    total_passed = sum(passed_totals)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f'MAPPO+GAT Evaluation Summary [{tag}]  —  {T} control steps  |  '
        f'total released: {total_passed} veh',
        fontsize=13, fontweight='bold')

    # Reward
    axes[0, 0].plot(steps, rewards_hist, color='#4CAF50', linewidth=1.0, alpha=0.65)
    if T >= 5:
        w    = min(15, T)
        roll = np.convolve(rewards_hist, np.ones(w) / w, mode='valid')
        axes[0, 0].plot(range(w, T + 1), roll,
                        color='#1B5E20', linewidth=2, label=f'{w}-step avg')
    axes[0, 0].axhline(np.mean(rewards_hist), color='orange', linestyle='--',
                       linewidth=1.2, label=f'mean={np.mean(rewards_hist):.4f}')
    axes[0, 0].set_title('Step Reward (norm. highway speed)')
    axes[0, 0].set_xlabel('Control step'); axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

    # Cumulative reward
    axes[0, 1].plot(steps, cum_hist, color='#FF9800', linewidth=1.5)
    axes[0, 1].set_title('Cumulative Reward'); axes[0, 1].set_xlabel('Control step')
    axes[0, 1].grid(True, alpha=0.3)

    # Actions over time
    for ai in range(N_AGENTS):
        axes[0, 2].step(steps, [ah[ai] for ah in actions_hist],
                        color=_COLORS[ai], label=AGENT_CFG[ai]['name'],
                        linewidth=1.0, where='post')
    axes[0, 2].set_yticks(list(range(ACTION_DIM)))
    axes[0, 2].set_yticklabels([ACTION_LABEL[a] for a in range(ACTION_DIM)])
    axes[0, 2].set_title('Actions over Time'); axes[0, 2].set_xlabel('Control step')
    axes[0, 2].legend(fontsize=8, ncol=2); axes[0, 2].grid(True, alpha=0.3)

    # Mean speed (primary objective)
    for ai in range(N_AGENTS):
        spds_kmh = [s[ai] * 3.6 for s in speed_hist]
        axes[1, 0].plot(steps, spds_kmh,
                        color=_COLORS[ai], label=AGENT_CFG[ai]['name'],
                        linewidth=1.0)
    # Global mean speed (bold)
    global_spd = [np.mean(s) * 3.6 for s in speed_hist]
    axes[1, 0].plot(steps, global_spd, 'k-', linewidth=2.5,
                    label=f'Overall mean ({np.mean(global_spd):.1f} km/h)')
    axes[1, 0].axhline(FREE_FLOW_SPEED * 3.6, color='gray', linestyle='--',
                       linewidth=0.9,
                       label=f'Free-flow {FREE_FLOW_SPEED*3.6:.0f} km/h')
    axes[1, 0].set_title('Mean Mainline Speed (km/h) — PRIMARY OBJECTIVE')
    axes[1, 0].set_xlabel('Control step')
    axes[1, 0].set_ylim(0, FREE_FLOW_SPEED * 3.6 * 1.1)
    axes[1, 0].legend(fontsize=7, ncol=2); axes[1, 0].grid(True, alpha=0.3)

    # Occupancy
    for ai in range(N_AGENTS):
        axes[1, 1].plot(steps, [o[ai] for o in occ_hist],
                        color=_COLORS[ai], label=AGENT_CFG[ai]['name'],
                        linewidth=1.0)
    axes[1, 1].axhline(TARGET_OCC, color='gray', linestyle='--',
                       linewidth=0.9, label=f'Ref {TARGET_OCC}%')
    axes[1, 1].set_title('Main-lane Occupancy (%)'); axes[1, 1].set_xlabel('Control step')
    axes[1, 1].legend(fontsize=8, ncol=2); axes[1, 1].grid(True, alpha=0.3)

    # Action distribution
    x = np.arange(ACTION_DIM); w = min(0.8 / max(N_AGENTS, 1), 0.25)
    for ai in range(N_AGENTS):
        counts = [sum(1 for ah in actions_hist if ah[ai] == a)
                  for a in range(ACTION_DIM)]
        pcts   = [c / T * 100 for c in counts]
        bars   = axes[1, 2].bar(
            x + (ai - (N_AGENTS - 1) / 2.0) * w, counts, w,
            label=AGENT_CFG[ai]['name'], color=_COLORS[ai], alpha=0.8)
        for b, p in zip(bars, pcts):
            axes[1, 2].text(b.get_x() + b.get_width() / 2,
                            b.get_height() + 0.5,
                            f'{p:.0f}%', ha='center', va='bottom', fontsize=6)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(['Stop\nred', 'Go\ngreen'])
    axes[1, 2].set_title('Action Distribution'); axes[1, 2].legend(fontsize=7, ncol=2)
    axes[1, 2].grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    path = EVAL_DIR / 'eval_summary.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Summary plot  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ALINEA vs MAPPO comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_plot(eval_csv: Path, alinea_csv: Path):
    """
    Overlay MAPPO eval_log.csv and ALINEA alinea_log.csv for shared ramps.
    Columns compared: ramp_flow, implied_rate_vph, occ_mean_pct, passed_total.
    Also adds a speed comparison if the ALINEA log contains main_speed_ms.
    """
    try:
        import pandas as pd
    except ImportError:
        print("  [comparison] pandas not installed — skipping comparison plot")
        return

    RAMP_MAP     = {
        'ramp_1': 'ramp_1',
        'ramp_2': 'ramp_2',
        'ramp_3': 'ramp_3',
        'ramp_4': 'ramp_4',
        'ramp_5': 'ramp_5',
    }
    ALINEA_COLOR = '#2196F3'
    MAPPO_COLOR  = '#F44336'

    try:
        al = pd.read_csv(alinea_csv); al.columns = al.columns.str.strip()
        mp = pd.read_csv(eval_csv);   mp.columns = mp.columns.str.strip()
    except Exception as exc:
        print(f"  [comparison] Could not load CSVs: {exc}")
        return

    has_speed = 'main_speed_ms' in al.columns
    n_rows    = 5 if has_speed else 4
    fig       = plt.figure(figsize=(20, 4 * n_rows))
    fig.suptitle('ALINEA vs MAPPO+GAT — Ramp Metering Comparison',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(n_rows, len(RAMP_MAP), figure=fig,
                           hspace=0.60, wspace=0.30)

    for col, (ali_ramp, mappo_ramp) in enumerate(RAMP_MAP.items()):
        a = al[al['ramp'] == ali_ramp].copy()
        m = mp[mp['ramp'] == mappo_ramp].copy()

        if a.empty or m.empty:
            print(f"  [comparison] No data for {ali_ramp}/{mappo_ramp} — skipping")
            continue

        at = a['sim_time'].values
        mt = m['sim_time'].values

        # Row 0: Released / Ramp flow
        ax = fig.add_subplot(gs[0, col])
        if 'released_veh_interval' in a.columns:
            ax.plot(at, a['released_veh_interval'],
                    color=ALINEA_COLOR, linewidth=1.1, alpha=0.85, label='ALINEA')
        ax.plot(mt, m['ramp_flow'],
                color=MAPPO_COLOR, linewidth=1.1, alpha=0.85, label='MAPPO+GAT')
        ax.set_title(f'{ali_ramp} / {mappo_ramp}\nReleased veh per interval')
        ax.set_xlabel('Sim time (s)'); ax.set_ylabel('Vehicles')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Row 1: Metering rate
        ax = fig.add_subplot(gs[1, col])
        if 'rate_vph' in a.columns:
            ax.plot(at, a['rate_vph'],
                    color=ALINEA_COLOR, linewidth=1.1, alpha=0.85, label='ALINEA')
        ax.step(mt, m['implied_rate_vph'],
                color=MAPPO_COLOR, linewidth=1.1, alpha=0.85,
                where='post', label='MAPPO+GAT')
        ax.set_title('Metering Rate (veh/h)')
        ax.set_xlabel('Sim time (s)'); ax.set_ylabel('Rate (veh/h)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Row 2: Downstream occupancy
        ax = fig.add_subplot(gs[2, col])
        if 'occ_mean_pct' in a.columns:
            ax.plot(at, a['occ_mean_pct'],
                    color=ALINEA_COLOR, linewidth=1.1, alpha=0.85, label='ALINEA')
        if 'occ_mean_pct' in m.columns:
            ax.plot(mt, m['occ_mean_pct'],
                    color=MAPPO_COLOR, linewidth=1.1, alpha=0.85, label='MAPPO+GAT')
        ax.axhline(TARGET_OCC, color='gray', linestyle='--',
                   linewidth=0.8, label=f'Ref {TARGET_OCC}%')
        ax.set_title('Downstream Occupancy (%)')
        ax.set_xlabel('Sim time (s)'); ax.set_ylabel('Occupancy (%)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Row 3: Cumulative passed
        ax = fig.add_subplot(gs[3, col])
        if 'passed_total' in a.columns:
            ax.plot(at, a['passed_total'],
                    color=ALINEA_COLOR, linewidth=1.5, label='ALINEA')
        if 'passed_total' in m.columns:
            ax.plot(mt, m['passed_total'],
                    color=MAPPO_COLOR, linewidth=1.5, label='MAPPO+GAT')
        ax.set_title('Cumulative Passed Vehicles')
        ax.set_xlabel('Sim time (s)'); ax.set_ylabel('Total passed (veh)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Row 4: Mean speed (if available)
        if has_speed:
            ax = fig.add_subplot(gs[4, col])
            ax.plot(at, a['main_speed_ms'].values * 3.6,
                    color=ALINEA_COLOR, linewidth=1.1, alpha=0.85, label='ALINEA')
            ax.plot(mt, m['main_speed_ms'].values * 3.6,
                    color=MAPPO_COLOR, linewidth=1.1, alpha=0.85, label='MAPPO+GAT')
            ax.axhline(FREE_FLOW_SPEED * 3.6, color='gray', linestyle='--',
                       linewidth=0.8,
                       label=f'Free-flow {FREE_FLOW_SPEED*3.6:.0f} km/h')
            ax.set_title('Mean Mainline Speed (km/h) — OBJECTIVE')
            ax.set_xlabel('Sim time (s)'); ax.set_ylabel('Speed (km/h)')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Summary stats box
    try:
        stat_lines = ['SUMMARY STATISTICS\n']
        for ali_ramp, mappo_ramp in RAMP_MAP.items():
            a = al[al['ramp'] == ali_ramp]
            m = mp[mp['ramp'] == mappo_ramp]
            if a.empty or m.empty:
                continue
            a_pass  = int(a['passed_total'].iloc[-1]) if 'passed_total' in a else 'N/A'
            m_pass  = int(m['passed_total'].iloc[-1]) if 'passed_total' in m else 'N/A'
            a_rate  = f"{a['rate_vph'].mean():.1f}"       if 'rate_vph'       in a else 'N/A'
            m_rate  = f"{m['implied_rate_vph'].mean():.1f}" if 'implied_rate_vph' in m else 'N/A'
            a_occ   = f"{a['occ_mean_pct'].mean():.2f}"   if 'occ_mean_pct'   in a else 'N/A'
            m_occ   = f"{m['occ_mean_pct'].mean():.2f}"   if 'occ_mean_pct'   in m else 'N/A'
            a_spd   = (f"{a['main_speed_ms'].mean()*3.6:.1f} km/h"
                       if 'main_speed_ms' in a else 'N/A')
            m_spd   = (f"{m['main_speed_ms'].mean()*3.6:.1f} km/h"
                       if 'main_speed_ms' in m else 'N/A')
            stat_lines += [
                f'{ali_ramp} / {mappo_ramp}',
                f'  Passed   ALINEA={a_pass}   MAPPO={m_pass}',
                f'  Rate vph ALINEA={a_rate}  MAPPO={m_rate}',
                f'  Occ %    ALINEA={a_occ}   MAPPO={m_occ}',
                f'  Spd      ALINEA={a_spd}   MAPPO={m_spd}',
                '',
            ]
        fig.text(0.50, 0.005, '\n'.join(stat_lines),
                 ha='center', va='bottom', fontsize=8.5, fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    except Exception:
        pass

    path = EVAL_DIR / 'comparison.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Comparison    → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(weights_dir: Path, end_time: float, use_gui: bool,
             stochastic: bool, save_images: bool, best: bool):

    weights_dir = Path(weights_dir)
    if best:
        weights_dir = find_best_checkpoint(weights_dir)
    else:
        print(f"  Checkpoint               : {weights_dir.resolve()}")

    missing = [f for f in ('encoder.weights.h5', 'actor.weights.h5', 'critic.weights.h5')
               if not (weights_dir / f).exists()]
    if missing:
        print("=" * 60)
        print("ERROR: No trained weights found.")
        print(f"  Expected checkpoint at: {weights_dir.resolve()}")
        print(f"  Missing files: {missing}")
        print()
        print("  Run training first:")
        print("    python Traci.py")
        print("=" * 60)
        sys.exit(1)

    obs_dim = infer_obs_dim_from_checkpoint(weights_dir)
    if obs_dim != OBS_DIM:
        print("=" * 60)
        print("ERROR: Checkpoint is not compatible with the current ALINEA-matched MAPPO setup.")
        print(f"  Checkpoint obs dim : {obs_dim}")
        print(f"  Required obs dim   : {OBS_DIM}")
        print()
        print("  This evaluation now uses the same 5-ramp E1/E2/E3 stop/go setup as ALINEA.")
        print("  Retrain MAPPO on this setup first:")
        print("    ./venv/bin/python Traci.py")
        print("=" * 60)
        sys.exit(1)
    agent = MAPPOAgent(obs_dim=obs_dim)
    try:
        agent.load(weights_dir)
    except RuntimeError as exc:
        print("=" * 60)
        print(f"ERROR: {exc}")
        print()
        print("  Evaluation now expects the retrained 2-action stop/go policy.")
        print("  Train a fresh checkpoint with:")
        print("    ./venv/bin/python Traci.py")
        print("=" * 60)
        sys.exit(1)
    tag = 'Stochastic' if stochastic else 'Greedy'
    print("=" * 66)
    print("  MASAR — Multi-Agent Synchronizer for Adaptive Ramps  (Evaluation)")
    print("=" * 66)
    print(f"  Policy mode              : MAPPO+GAT [{tag}]")
    print(f"  Objective                : Max mean highway speed, min queues")
    print(f"  Controlled ramps         : {N_AGENTS} mainline-entry ramps")
    print(f"  Yellow phase duration    : {YELLOW_DURATION} s on green→red")
    print(f"  Spill-back safety guard  : ramp queue ≥ {SPILLBACK_QUEUE_M:.0f} m forces go")
    print(f"  Free-flow speed          : {FREE_FLOW_SPEED} m/s  "
          f"({FREE_FLOW_SPEED*3.6:.0f} km/h)")
    print(f"  Episode length           : {end_time:.0f} s")
    print(f"  Control interval         : {CTRL_INTERVAL} s")
    print(f"  SUMO seed                : {SUMO_SEED}")
    print(f"  Checkpoint obs dim       : {obs_dim}")
    if agent.using_legacy_checkpoint:
        print("  [checkpoint warning] Critic weights were not loaded; "
              "value estimates are disabled.")
    print(f"  Per-step images          : "
          f"{'yes' if save_images else 'no (use --images to enable)'}")
    print()

    binary = 'sumo-gui' if use_gui else 'sumo'
    if use_gui and not os.environ.get('DISPLAY'):
        print("  [GUI fallback] DISPLAY is not set; starting headless SUMO instead.")
        binary = 'sumo'
        use_gui = False

    sumo_cmd = [binary, '-c', SUMO_CFG,
                '--step-length', str(STEP_LEN),
                '--seed',        str(SUMO_SEED),
                '--no-warnings', 'true',
                '--quit-on-end', 'false']
    if use_gui and Path('eval_viewsettings.xml').exists():
        sumo_cmd += ['--gui-settings-file', 'eval_viewsettings.xml']

    traci.start(sumo_cmd)

    runtime_cfg = get_runtime_agent_cfg(verbose=True)
    init_gui_signal_indicators(runtime_cfg)

    # Initialise all ramp meters to go (green) phase
    for cfg in runtime_cfg:
        try:
            apply_signal_state(cfg, 'g')
        except traci.exceptions.TraCIException as exc:
            print(f"  [TL init warning] {cfg['name']}: {exc}")

    prev_actions    = [1] * N_AGENTS
    passed_totals   = [0] * N_AGENTS
    ramp_flow_accum = [0.0] * N_AGENTS
    queue_accum     = [0.0] * N_AGENTS
    main_flow_accum = [0.0] * N_AGENTS
    main_occ_accum  = [0.0] * N_AGENTS
    main_spd_accum  = [0.0] * N_AGENTS
    tis_accum       = 0.0
    sim_steps_accum = 0

    # Traffic-light state tracking for yellow transition
    signal_chars  = ['g'] * N_AGENTS
    yellow_timers = [0.0] * N_AGENTS

    # MASAR KPIs (Objective 4)
    ttt_seconds      = 0.0
    queue_sum_m      = 0.0
    safety_overrides = 0

    rewards_hist : List[float]       = []
    cum_hist     : List[float]       = []
    actions_hist : List[List[int]]   = []
    flows_hist   : List[List[float]] = []
    speed_hist   : List[List[float]] = []
    occ_hist     : List[List[float]] = []
    rate_hist    : List[List[float]] = []
    probs_hist   : List[np.ndarray]  = []

    cum_reward = 0.0
    ctrl_step  = 0

    hdr = (f"{'Step':>6}  {'t(s)':>7}  {'actions':>10}  "
           f"{'speed(km/h)':>22}  {'rwd':>8}  {'cum':>10}")
    print(hdr)
    print("-" * len(hdr))

    try:
        with open(EVAL_CSV, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=EVAL_CSV_FIELDS)
            writer.writeheader()

            while True:
                sim_time = traci.simulation.getTime()
                if sim_time >= end_time:
                    break

                traci.simulationStep()
                sim_time = traci.simulation.getTime()

                # Advance yellow countdowns each sim step
                process_yellow_transitions(runtime_cfg, signal_chars, yellow_timers)

                # KPI + reward signal: Total Travel Time = Σ vehicles × dt
                try:
                    vehs_now      = traci.vehicle.getIDCount()
                    ttt_seconds  += vehs_now * STEP_LEN
                    tis_accum    += float(vehs_now)
                except traci.exceptions.TraCIException:
                    pass

                for i, cfg in enumerate(runtime_cfg):
                    ramp_flow_accum[i] += _ramp_veh(cfg)
                    queue_accum[i]     += _ramp_queue_m(cfg)
                    main_flow_accum[i] += _main_flow(cfg)
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

                actions, _, value, probs = select_action(agent, obs, stochastic)

                # ARM safety override
                raw_actions = actions[:]
                actions     = apply_safety_override(actions, queue_accum)
                safety_overrides += sum(1 for a, b in zip(raw_actions, actions) if a != b)

                for i, cfg in enumerate(runtime_cfg):
                    try:
                        transition_signal(runtime_cfg, i, actions[i],
                                          signal_chars, yellow_timers)
                    except traci.exceptions.TraCIException as exc:
                        print(f"  [TL warning] {cfg['name']} a={actions[i]}: {exc}")
                    passed_totals[i] += int(ramp_flow_accum[i])

                reward     = compute_reward(tis_accum, queue_accum)
                cum_reward += reward

                avg_spd   = [main_spd_accum[i] / CTRL_INTERVAL for i in range(N_AGENTS)]
                avg_occ   = [main_occ_accum[i]  / CTRL_INTERVAL for i in range(N_AGENTS)]
                avg_queue = [queue_accum[i]     / CTRL_INTERVAL for i in range(N_AGENTS)]
                cur_rates = [RATE_VPH_TABLE[actions[i]] for i in range(N_AGENTS)]
                queue_sum_m += float(np.mean(avg_queue))

                rewards_hist.append(reward)
                cum_hist.append(cum_reward)
                actions_hist.append(actions[:])
                flows_hist.append(list(ramp_flow_accum))
                speed_hist.append(avg_spd)
                occ_hist.append(avg_occ)
                rate_hist.append(cur_rates)
                probs_hist.append(probs.copy())

                spd_str = str([f'{v*3.6:.1f}' for v in avg_spd])
                print(
                    f"{ctrl_step:>6}  {sim_time:>7.0f}  "
                    f"{str(actions):>10}  "
                    f"{spd_str:>22}  "
                    f"{reward:>+8.4f}  {cum_reward:>10.2f}"
                )

                for i, cfg in enumerate(runtime_cfg):
                    writer.writerow({
                        'episode':          -1,
                        'ep_type':          'eval',
                        'sim_time':         f'{sim_time:.1f}',
                        'ramp':             cfg['name'],
                        'action':           actions[i],
                        'speed_ms':         SPEED_TABLE[actions[i]],
                        'implied_rate_vph': RATE_VPH_TABLE[actions[i]],
                        'ramp_flow':        f'{ramp_flow_accum[i]:.1f}',
                        'main_flow':        f'{main_flow_accum[i]:.1f}',
                        'main_speed_ms':    f'{avg_spd[i]:.4f}',
                        'occ_mean_pct':     f'{avg_occ[i]:.4f}',
                        'reward':           f'{reward:.6f}',
                        'cum_reward':       f'{cum_reward:.4f}',
                        'value':            f'{value:.4f}',
                        'passed_total':     passed_totals[i],
                        'prob_stop':        f'{probs[i, 0]:.4f}',
                        'prob_go':          f'{probs[i, 1]:.4f}',
                    })

                if save_images:
                    save_step_image(ctrl_step, sim_time, stochastic,
                                    rewards_hist, cum_hist, actions_hist,
                                    flows_hist, speed_hist, occ_hist,
                                    rate_hist, probs_hist)

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

    print("-" * len(hdr))
    if not rewards_hist:
        print("No control steps completed — check SUMO config.")
        return

    global_speeds_kmh = [np.mean(s) * 3.6 for s in speed_hist]

    print(f"\nEvaluation complete — {ctrl_step} control steps")
    print(f"  Cumulative reward     : {cum_reward:.4f}")
    print(f"  Average step reward   : {np.mean(rewards_hist):.4f}")
    print(f"  Std step reward       : {np.std(rewards_hist):.4f}")
    print(f"  Best / worst step     : "
          f"{max(rewards_hist):+.4f} / {min(rewards_hist):+.4f}")
    print(f"  Mean highway speed    : {np.mean(global_speeds_kmh):.1f} km/h")
    print(f"  Free-flow speed       : {FREE_FLOW_SPEED*3.6:.0f} km/h")
    print(f"  Speed utilisation     : "
          f"{np.mean(global_speeds_kmh)/(FREE_FLOW_SPEED*3.6)*100:.1f}%")
    # MASAR KPIs (Objective 4)
    ttt_hours = ttt_seconds / 3600.0
    avg_queue_m = queue_sum_m / max(ctrl_step, 1)
    print(f"  Total Travel Time     : {ttt_hours:.2f} veh·h   (Objective 4 KPI)")
    print(f"  Avg ramp queue length : {avg_queue_m:.2f} m    (Objective 4 KPI)")
    print(f"  Safety overrides used : {safety_overrides}     (ARM forced go)")
    print(f"  Total veh released    :")
    for i, cfg in enumerate(runtime_cfg):
        print(f"    {cfg['name']}: {passed_totals[i]} vehicles")

    print(f"\n  Action frequency:")
    for ai in range(N_AGENTS):
        counts = [sum(1 for ah in actions_hist if ah[ai] == a)
                  for a in range(ACTION_DIM)]
        total  = sum(counts) or 1
        parts  = [f"{Traci.ACTION_LABEL[a].split('(')[0].strip()}={c/total*100:.1f}%"
                  for a, c in enumerate(counts)]
        print(f"    {AGENT_CFG[ai]['name']}: {' | '.join(parts)}")

    print()
    save_summary_plot(rewards_hist, cum_hist, actions_hist,
                      flows_hist, speed_hist, occ_hist, rate_hist,
                      passed_totals, stochastic)

    if ALINEA_CSV.exists():
        save_comparison_plot(EVAL_CSV, ALINEA_CSV)
    else:
        print(f"  [comparison] ALINEA CSV not found at {ALINEA_CSV} — skipping")

    print(f"\n  CSV log      → {EVAL_CSV}")
    print(f"  Step images  → {EVAL_IMGS}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Evaluate trained MAPPO+GAT ramp-metering policy '
                    '(objective: maximise mean highway speed)')
    p.add_argument('--weights',    default=str(DEFAULT_WEIGHTS),
                   help='Checkpoint directory  (default: mappo_gat_runs/weights)')
    p.add_argument('--best',       action='store_true',
                   help='Auto-select the best checkpoint by training proxy')
    p.add_argument('--end',        type=float, default=DEFAULT_END_TIME,
                   help=f'Simulation end time in seconds  (default: sumocfg end = {DEFAULT_END_TIME:.0f})')
    p.add_argument('--gui',        dest='gui', action='store_true',
                   help='Open SUMO-GUI to visualise the simulation (default)')
    p.add_argument('--nogui',      dest='gui', action='store_false',
                   help='Run headless SUMO without opening the GUI')
    p.add_argument('--stochastic', action='store_true',
                   help='Sample from policy distribution instead of argmax')
    p.add_argument('--images',     action='store_true',
                   help='Save a dashboard image at every control step (slow)')
    p.set_defaults(gui=True)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(
        weights_dir = Path(args.weights),
        end_time    = args.end,
        use_gui     = args.gui,
        stochastic  = args.stochastic,
        save_images = args.images,
        best        = args.best,
    )
