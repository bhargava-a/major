"""
Realistic traffic simulator — CESNET baseline with randomised attack injection.

Key realism features:
 - Normal periods are random length (20–120s), NOT fixed
 - 25% chance an attack is skipped (just stays normal longer)
 - Attack durations are random (8–35s per scenario)
 - Brief micro-anomalies (1–3s blips) can fire randomly even during normal
 - Gradual build-up into attack, gradual wind-down after
 - Time-of-day traffic variation (busier during "day", quieter at "night")
 - Random chance of back-to-back attacks (10% probability)
"""

import random
import math
import time
from dataclasses import dataclass, field


# ── CESNET baseline (mean, std) ───────────────────────────────────────────────
BASELINE = {
    "n_flows":                 (6_500_000,   400_000),
    "n_packets":               (1_400_000_000, 100_000_000),
    "n_bytes":                 (1_400_000_000_000, 80_000_000_000),
    "sum_n_dest_asn":          (900_000,     50_000),
    "average_n_dest_asn":      (10.2,        0.4),
    "std_n_dest_asn":          (21.5,        1.0),
    "sum_n_dest_ports":        (950_000,     60_000),
    "average_n_dest_ports":    (11.0,        1.5),
    "std_n_dest_ports":        (100.0,       15.0),
    "sum_n_dest_ip":           (3_200_000,   150_000),
    "average_n_dest_ip":       (37.0,        3.0),
    "std_n_dest_ip":           (125.0,       12.0),
    "tcp_udp_ratio_packets":   (0.74,        0.03),
    "tcp_udp_ratio_bytes":     (0.74,        0.03),
    "dir_ratio_packets":       (0.47,        0.03),
    "dir_ratio_bytes":         (0.42,        0.03),
    "avg_duration":            (24.5,        2.0),
    "avg_ttl":                 (134.0,       3.0),
}


@dataclass
class AttackScenario:
    name: str
    min_duration: int
    max_duration: int
    overrides: dict = field(default_factory=dict)


ATTACKS = [
    AttackScenario("Port Scan", 12, 30, {
        "average_n_dest_ports": (18.0, 2.0),
        "std_n_dest_ports":     (300.0, 30.0),
        "average_n_dest_ip":    (80.0, 10.0),
        "n_flows":              (12_000_000, 800_000),
        "avg_duration":         (2.0, 0.8),
        "avg_ttl":              (64.0, 5.0),
    }),
    AttackScenario("DDoS", 10, 28, {
        "n_flows":              (25_000_000, 2_000_000),
        "n_packets":            (6_000_000_000, 400_000_000),
        "n_bytes":              (5_000_000_000_000, 500_000_000_000),
        "tcp_udp_ratio_packets":(0.05, 0.02),
        "tcp_udp_ratio_bytes":  (0.05, 0.02),
        "dir_ratio_packets":    (0.10, 0.03),
        "dir_ratio_bytes":      (0.08, 0.03),
    }),
    AttackScenario("Data Exfiltration", 15, 35, {
        "dir_ratio_packets":    (0.85, 0.04),
        "dir_ratio_bytes":      (0.90, 0.03),
        "n_bytes":              (4_500_000_000_000, 300_000_000_000),
        "avg_duration":         (120.0, 15.0),
        "average_n_dest_ip":    (5.0, 1.5),
        "average_n_dest_ports": (2.0, 0.8),
    }),
    AttackScenario("Network Scan", 10, 25, {
        "average_n_dest_asn":   (45.0, 5.0),
        "sum_n_dest_asn":       (4_000_000, 300_000),
        "average_n_dest_ip":    (120.0, 15.0),
        "sum_n_dest_ip":        (8_000_000, 600_000),
        "avg_ttl":              (55.0, 8.0),
        "avg_duration":         (1.5, 0.5),
        "n_flows":              (15_000_000, 1_000_000),
    }),
]

# Brief micro-anomaly — lasts 1–3 ticks, subtle enough to be a blip
MICRO_ANOMALY = AttackScenario("Micro Spike", 1, 3, {
    "n_flows":   (10_000_000, 1_000_000),
    "n_packets": (3_000_000_000, 200_000_000),
    "avg_ttl":   (90.0, 10.0),
})


def _gauss(mean: float, std: float) -> float:
    return max(0.0, random.gauss(mean, std))


def _smooth(current: float, target: float, alpha: float) -> float:
    return current * (1 - alpha) + target * alpha


def _time_of_day_multiplier() -> float:
    """Simulate busier daytime, quieter night — based on wall clock hour."""
    hour = time.localtime().tm_hour
    # Peak 9am-6pm, quiet midnight-6am
    if 9 <= hour < 18:
        return random.uniform(1.0, 1.15)
    elif 0 <= hour < 6:
        return random.uniform(0.70, 0.85)
    else:
        return random.uniform(0.90, 1.05)


class TrafficSimulator:

    def __init__(self):
        self._state = "normal"
        self._state_remaining = random.randint(30, 80)
        self._attack: AttackScenario | None = None
        self._attack_pool = ATTACKS.copy()
        random.shuffle(self._attack_pool)
        self._attack_pool_idx = 0
        self._current = {k: v[0] for k, v in BASELINE.items()}
        self._tod_mult = _time_of_day_multiplier()
        self._tod_refresh = 60  # refresh time-of-day multiplier every 60 ticks

    def _next_attack(self) -> AttackScenario:
        """Pick next attack — reshuffles deck when exhausted (no fixed order)."""
        if self._attack_pool_idx >= len(self._attack_pool):
            random.shuffle(self._attack_pool)
            self._attack_pool_idx = 0
        atk = self._attack_pool[self._attack_pool_idx]
        self._attack_pool_idx += 1
        return atk

    def _target_for_feature(self, feature: str) -> tuple[float, float]:
        if self._state in ("attack", "micro") and self._attack and feature in self._attack.overrides:
            return self._attack.overrides[feature]
        return BASELINE[feature]

    def tick(self) -> dict:
        self._state_remaining -= 1
        self._tod_refresh -= 1
        if self._tod_refresh <= 0:
            self._tod_mult = _time_of_day_multiplier()
            self._tod_refresh = 60

        if self._state_remaining <= 0:
            if self._state == "normal":
                # Random chance: skip attack (25%), do micro-blip (10%), or full attack (65%)
                roll = random.random()
                if roll < 0.25:
                    # Stay normal longer
                    self._state = "normal"
                    self._state_remaining = random.randint(20, 80)
                    self._attack = None
                elif roll < 0.35:
                    # Brief micro anomaly
                    self._attack = MICRO_ANOMALY
                    self._state = "micro"
                    self._state_remaining = random.randint(1, 3)
                else:
                    self._attack = self._next_attack()
                    self._state = "attack"
                    self._state_remaining = random.randint(
                        self._attack.min_duration, self._attack.max_duration
                    )

            elif self._state in ("attack", "micro"):
                # 10% chance: chain directly into another attack (no recovery)
                if self._state == "attack" and random.random() < 0.10:
                    self._attack = self._next_attack()
                    self._state = "attack"
                    self._state_remaining = random.randint(
                        self._attack.min_duration, self._attack.max_duration
                    )
                else:
                    self._attack = None
                    self._state = "recovery"
                    self._state_remaining = random.randint(5, 18)

            else:  # recovery
                self._state = "normal"
                self._state_remaining = random.randint(20, 100)

        # Compute alpha — faster ramp-up into attack, slower wind-down
        if self._state == "attack":
            alpha = random.uniform(0.25, 0.45)   # builds up quickly
        elif self._state == "micro":
            alpha = 0.9                            # snaps in immediately
        elif self._state == "recovery":
            alpha = random.uniform(0.08, 0.18)    # slow return
        else:
            alpha = random.uniform(0.05, 0.12)    # gentle normal drift

        result = {}
        for feat, (base_mean, base_std) in BASELINE.items():
            t_mean, t_std = self._target_for_feature(feat)

            # Apply time-of-day only to volume features during normal/recovery
            if self._state in ("normal", "recovery") and feat in ("n_flows", "n_packets", "n_bytes"):
                t_mean = t_mean * self._tod_mult

            blended = _smooth(self._current[feat], t_mean, alpha)
            result[feat] = _gauss(blended, t_std)
            self._current[feat] = blended

        result["attack_scenario"] = self._attack.name if self._attack else "NORMAL"
        result["sim_state"] = self._state
        return result
