#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List


def _import_sumo_tools():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    try:
        import traci  # type: ignore
        from sumolib import checkBinary  # type: ignore
        return traci, checkBinary
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import traci/sumolib. Set SUMO_HOME and run from a SUMO Python environment."
        ) from exc


traci, checkBinary = _import_sumo_tools()


@dataclass
class RampMeter:
    name: str
    tls_id: str
    e1_id: str  # induction loop after the meter, counts released vehicles
    e2_id: str  # laneArea on ramp, used for queue length
    e3_ids: List[str]  # downstream laneArea detectors across all mainline lanes
    target_occ: float = 20.0
    kr: float = 70.0
    control_interval: int = 20
    green_time: float = 2.0
    min_rate_vph: float = 240.0
    max_rate_vph: float = 900.0
    queue_override_m: float = 25.0
    queue_release_rate_vph: float = 900.0
    rate_vph: float = 600.0
    red_time: float = field(init=False)
    state: str = field(default="r")
    state_remaining: float = field(default=0.0)
    occ_samples: Deque[float] = field(default_factory=deque)
    queue_samples_m: Deque[float] = field(default_factory=deque)
    speed_samples_ms: Deque[float] = field(default_factory=deque)
    passed_total: int = 0
    released_this_interval: int = 0
    last_occ_mean: float = 0.0
    last_queue_max_m: float = 0.0
    last_speed_mean_ms: float = 0.0

    def __post_init__(self) -> None:
        self.red_time = self._rate_to_red(self.rate_vph)

    def _rate_to_red(self, rate_vph: float) -> float:
        cycle = 3600.0 / max(rate_vph, 1e-6)
        return max(0.0, cycle - self.green_time)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def collect_step_data(self) -> None:
        passed = traci.inductionloop.getLastStepVehicleNumber(self.e1_id)
        occ_vals = [traci.lanearea.getLastStepOccupancy(det_id) for det_id in self.e3_ids]
        occ = sum(occ_vals) / len(occ_vals) if occ_vals else 0.0
        q_m = traci.lanearea.getJamLengthMeters(self.e2_id)
        spd_vals = [traci.lanearea.getLastStepMeanSpeed(det_id) for det_id in self.e3_ids]
        spd_vals = [v for v in spd_vals if v >= 0.0]
        spd = sum(spd_vals) / len(spd_vals) if spd_vals else 0.0

        self.passed_total += int(passed)
        self.released_this_interval += int(passed)
        self.occ_samples.append(float(occ))
        self.queue_samples_m.append(float(q_m))
        self.speed_samples_ms.append(float(spd))

    def update_alinea(self) -> Dict[str, float]:
        occ_mean = sum(self.occ_samples) / len(self.occ_samples) if self.occ_samples else 0.0
        queue_max_m = max(self.queue_samples_m) if self.queue_samples_m else 0.0
        speed_mean_ms = (sum(self.speed_samples_ms) / len(self.speed_samples_ms)
                         if self.speed_samples_ms else 0.0)

        new_rate = self.rate_vph + self.kr * (self.target_occ - occ_mean)
        new_rate = self._clamp(new_rate, self.min_rate_vph, self.max_rate_vph)

        if queue_max_m >= self.queue_override_m:
            new_rate = max(new_rate, self.queue_release_rate_vph)

        self.rate_vph = self._clamp(new_rate, self.min_rate_vph, self.max_rate_vph)
        self.red_time = self._rate_to_red(self.rate_vph)
        self.last_occ_mean = occ_mean
        self.last_queue_max_m = queue_max_m
        self.last_speed_mean_ms = speed_mean_ms

        self.occ_samples.clear()
        self.queue_samples_m.clear()
        self.speed_samples_ms.clear()

        stats = {
            "occ_mean": occ_mean,
            "ramp_queue_m": queue_max_m,
            "speed_mean_ms": speed_mean_ms,
            "released": self.released_this_interval,
            "rate_vph": self.rate_vph,
            "red_time_s": self.red_time,
        }
        self.released_this_interval = 0
        return stats

    def step_signal(self, dt: float) -> None:
        ramp_veh = traci.lanearea.getLastStepVehicleNumber(self.e2_id)
        ramp_q_m = traci.lanearea.getJamLengthMeters(self.e2_id)

        if ramp_veh == 0 and ramp_q_m <= 0.1:
            self.state = "r"
            self.state_remaining = 0.0
            traci.trafficlight.setRedYellowGreenState(self.tls_id, self.state)
            return

        self.state_remaining -= dt

        if self.state_remaining <= 0:
            if self.state == "r":
                self.state = "g"
                self.state_remaining = self.green_time
            else:
                self.state = "r"
                self.state_remaining = self.red_time

        traci.trafficlight.setRedYellowGreenState(self.tls_id, self.state)


def make_ramps() -> List[RampMeter]:
    return [
        RampMeter(
            name="ramp_1",
            tls_id="J0",
            e1_id="alinea_413583321_E1",
            e2_id="alinea_413583321_E2",
            e3_ids=[
                "alinea_413583321_E3_0",
                "alinea_413583321_E3_1",
                "alinea_413583321_E3_2",
                "alinea_413583321_E3_3",
            ],
        ),
        RampMeter(
            name="ramp_2",
            tls_id="J4",
            e1_id="alinea_413587595_E1",
            e2_id="alinea_413587595_E2",
            e3_ids=[
                "alinea_413587595_E3_0",
                "alinea_413587595_E3_1",
                "alinea_413587595_E3_2",
                "alinea_413587595_E3_3",
            ],
        ),
        RampMeter(
            name="ramp_3",
            tls_id="J10",
            e1_id="alinea_1301290506_E1",
            e2_id="alinea_1301290506_E2",
            e3_ids=[
                "alinea_1301290506_E3_0",
                "alinea_1301290506_E3_1",
                "alinea_1301290506_E3_2",
                "alinea_1301290506_E3_3",
            ],
        ),
        RampMeter(
            name="ramp_4",
            tls_id="J14",
            e1_id="alinea_92072668_E1",
            e2_id="alinea_92072668_E2",
            e3_ids=[
                "alinea_92072668_E3_0",
                "alinea_92072668_E3_1",
                "alinea_92072668_E3_2",
                "alinea_92072668_E3_3",
            ],
        ),
        RampMeter(
            name="ramp_5",
            tls_id="J16",
            e1_id="alinea_1300204703_E1",
            e2_id="alinea_1300204703_E2",
            e3_ids=[
                "alinea_1300204703_E3_0",
                "alinea_1300204703_E3_1",
                "alinea_1300204703_E3_2",
                "alinea_1300204703_E3_3",
            ],
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Five-ramp ALINEA controller for SUMO/TraCI")
    parser.add_argument("-c", "--config", default="king_fahad_road.sumocfg",
                        help="Path to .sumocfg  (default: king_fahad_road.sumocfg)")
    parser.add_argument("--nogui", action="store_true", help="Run sumo instead of sumo-gui")
    parser.add_argument("--step-length", type=float, default=1.0, help="Simulation step length in seconds")
    parser.add_argument("--end", type=float, default=None, help="Stop simulation at this time (s)")
    parser.add_argument("--target-occ", type=float, default=20.0, help="ALINEA target downstream occupancy (%)")
    parser.add_argument("--kr", type=float, default=70.0, help="ALINEA regulator gain (veh/h per occupancy point)")
    parser.add_argument("--green", type=float, default=2.0, help="Green time per release (s)")
    parser.add_argument("--control-interval", type=int, default=20, help="ALINEA update period (s)")
    parser.add_argument("--min-rate", type=float, default=240.0, help="Minimum metering rate (veh/h)")
    parser.add_argument("--max-rate", type=float, default=900.0, help="Maximum metering rate (veh/h)")
    parser.add_argument("--queue-override-m", type=float, default=25.0, help="If E2 jam length exceeds this, force high release")
    parser.add_argument("--queue-release-rate", type=float, default=900.0, help="Release rate when queue override activates")
    parser.add_argument("--csv", type=str, default="alinea_log.csv", help="Optional CSV log output")
    return parser.parse_args()


def run() -> None:
    args = parse_args()

    ramps = make_ramps()
    for ramp in ramps:
        ramp.target_occ = args.target_occ
        ramp.kr = args.kr
        ramp.green_time = args.green
        ramp.control_interval = args.control_interval
        ramp.min_rate_vph = args.min_rate
        ramp.max_rate_vph = args.max_rate
        ramp.queue_override_m = args.queue_override_m
        ramp.queue_release_rate_vph = args.queue_release_rate
        ramp.rate_vph = 600.0
        ramp.red_time = ramp._rate_to_red(ramp.rate_vph)
        ramp.state = "r"
        ramp.state_remaining = 0.0

    binary = checkBinary("sumo" if args.nogui else "sumo-gui")
    sumo_cmd = [binary, "-c", args.config, "--step-length", str(args.step_length)]
    traci.start(sumo_cmd)

    csv_path = Path(args.csv) if args.csv else None
    csv_file = None
    csv_writer = None
    if csv_path is not None:
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "sim_time",
            "ramp",
            "tls",
            "occ_mean_pct",
            "ramp_queue_m",
            "main_speed_ms",
            "released_veh_interval",
            "rate_vph",
            "red_time_s",
            "passed_total",
        ])

    try:
        step = 0
        dt = args.step_length
        while traci.simulation.getMinExpectedNumber() > 0:
            if args.end is not None and step * dt >= args.end:
                break

            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            for ramp in ramps:
                ramp.collect_step_data()
                ramp.step_signal(dt)

            if step > 0 and step % args.control_interval == 0:
                print(f"\n=== t={sim_time:.0f}s ===")
                for ramp in ramps:
                    stats = ramp.update_alinea()
                    print(
                        f"{ramp.name} ({ramp.tls_id}) | "
                        f"occ={stats['occ_mean']:.2f}% | "
                        f"queue={stats['ramp_queue_m']:.2f} m | "
                        f"spd={stats['speed_mean_ms']*3.6:.1f} km/h | "
                        f"released={stats['released']} | "
                        f"rate={stats['rate_vph']:.1f} veh/h | "
                        f"red={stats['red_time_s']:.2f} s | "
                        f"passed_total={ramp.passed_total}"
                    )
                    if csv_writer is not None:
                        csv_writer.writerow([
                            f"{sim_time:.2f}",
                            ramp.name,
                            ramp.tls_id,
                            f"{stats['occ_mean']:.6f}",
                            f"{stats['ramp_queue_m']:.6f}",
                            f"{stats['speed_mean_ms']:.6f}",
                            stats['released'],
                            f"{stats['rate_vph']:.6f}",
                            f"{stats['red_time_s']:.6f}",
                            ramp.passed_total,
                        ])

            step += 1
    finally:
        traci.close()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    run()
