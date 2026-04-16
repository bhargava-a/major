"""
Live traffic ingestor — uses the simulator to generate realistic network
feature vectors, feeds each to the LSTM model, and updates Prometheus + SQLite.

Attack scenarios rotate automatically:
  Normal (90s) → Attack (35-50s) → Recovery (20s) → repeat
"""
import asyncio
import logging

from app.model_inference import AnomalyDetector
from app.database import save_anomaly
from app import metrics as m
from app.simulator import TrafficSimulator

logger = logging.getLogger(__name__)


async def run_ingestor(detector: AnomalyDetector):
    """Async loop: generate simulated traffic, run model, push metrics."""
    sim = TrafficSimulator()
    logger.info("Simulator started — Normal(90s) → Attack(35-50s) → Recovery(20s) → repeat")

    while True:
        row = sim.tick()
        state = row.pop("sim_state")
        scenario = row.pop("attack_scenario")

        result = detector.push(row)
        if result is not None:
            # Model output
            m.ANOMALY_SCORE.set(result["anomaly_score"])
            m.ANOMALY_THRESHOLD.set(result["threshold"])
            m.IS_ANOMALY.set(1 if result["is_anomaly"] else 0)
            m.SAMPLES_PROCESSED.inc()
            # Traffic volume
            m.N_FLOWS.set(result["n_flows"])
            m.N_PACKETS.set(result["n_packets"])
            m.N_BYTES.set(result["n_bytes"])
            # Protocol behaviour
            m.TCP_UDP_RATIO_PACKETS.set(result["tcp_udp_ratio_packets"])
            m.TCP_UDP_RATIO_BYTES.set(result["tcp_udp_ratio_bytes"])
            m.DIR_RATIO_PACKETS.set(result["dir_ratio_packets"])
            m.DIR_RATIO_BYTES.set(result["dir_ratio_bytes"])
            # Destination diversity
            m.AVG_DEST_IP.set(result["average_n_dest_ip"])
            m.AVG_DEST_PORTS.set(result["average_n_dest_ports"])
            m.AVG_DEST_ASN.set(result["average_n_dest_asn"])
            # Flow characteristics
            m.AVG_DURATION.set(result["avg_duration"])
            m.AVG_TTL.set(result["avg_ttl"])

            if result["is_anomaly"]:
                m.ANOMALIES_TOTAL.inc()
                save_anomaly(result)
                logger.info(
                    "ANOMALY [%s] score=%.6f threshold=%.6f flows=%.0f",
                    scenario, result["anomaly_score"], result["threshold"], result["n_flows"],
                )
            else:
                logger.debug("NORMAL [%s] score=%.6f", state, result["anomaly_score"])

        await asyncio.sleep(1.0)
