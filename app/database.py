"""SQLite storage for flagged anomalies using SQLAlchemy."""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Session

DB_URL = "sqlite:///./anomalies.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})


class Base(DeclarativeBase):
    pass


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    n_flows = Column(Float)
    n_packets = Column(Float)
    n_bytes = Column(Float)
    severity = Column(String(10))  # "LOW" / "MEDIUM" / "HIGH"


def init_db():
    Base.metadata.create_all(bind=engine)


def save_anomaly(result: dict):
    score = result["anomaly_score"]
    thresh = result["threshold"]
    ratio = score / thresh if thresh > 0 else 1.0
    severity = "HIGH" if ratio > 2.0 else ("MEDIUM" if ratio > 1.5 else "LOW")

    with Session(engine) as session:
        record = Anomaly(
            detected_at=datetime.utcnow(),
            anomaly_score=score,
            threshold=thresh,
            n_flows=result.get("n_flows"),
            n_packets=result.get("n_packets"),
            n_bytes=result.get("n_bytes"),
            severity=severity,
        )
        session.add(record)
        session.commit()


def get_recent_anomalies(limit: int = 50) -> list[dict]:
    with Session(engine) as session:
        rows = (
            session.query(Anomaly)
            .order_by(Anomaly.detected_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "detected_at": r.detected_at.isoformat(),
                "anomaly_score": r.anomaly_score,
                "threshold": r.threshold,
                "n_flows": r.n_flows,
                "n_packets": r.n_packets,
                "n_bytes": r.n_bytes,
                "severity": r.severity,
            }
            for r in rows
        ]
