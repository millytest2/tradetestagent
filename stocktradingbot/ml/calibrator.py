"""XGBoost probability calibrator for stock direction prediction.

Trains on historical trade outcomes with recency weighting (30-day half-life).
Features mirror the same technical + sentiment signals used at trade time.
Falls back to rule-based prior when < 30 settled trades exist.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import settings
from core.models import PredictionFeatures

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "rsi_14",
    "macd_signal",
    "bb_position",
    "volume_ratio",
    "price_change_1d",
    "price_change_5d",
    "price_change_20d",
    "distance_from_52w_high",
    "short_interest_ratio",
    "compound_sentiment",
    "post_count",
    "avg_engagement",
    "trend_score",
    "whale_bid_imbalance",
]

MIN_SAMPLES = 30


def _rule_based_prob(features: PredictionFeatures) -> float:
    prob = 0.5
    if features.rsi_14 < 30:
        prob += 0.10
    elif features.rsi_14 > 70:
        prob -= 0.10
    prob += min(max(features.macd_signal * 0.02, -0.08), 0.08)
    prob += features.compound_sentiment * 0.06
    if features.bb_position <= 0.1:
        prob += 0.05
    elif features.bb_position >= 0.9:
        prob -= 0.05
    return float(max(0.05, min(0.95, prob)))


class StockCalibrator:
    def __init__(self) -> None:
        self._model = None
        self.is_trained = False
        self._model_path = Path(settings.model_path)
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self._model_path.exists():
            try:
                import joblib
                self._model = joblib.load(self._model_path)
                self.is_trained = True
                logger.info("Loaded XGBoost model from %s", self._model_path)
            except Exception as e:
                logger.warning("Could not load model: %s", e)

    def predict_proba(self, features: PredictionFeatures) -> float:
        if not self.is_trained or self._model is None:
            return _rule_based_prob(features)
        try:
            row = {col: getattr(features, col, 0.0) for col in FEATURE_COLS}
            X = pd.DataFrame([row])[FEATURE_COLS]
            prob = float(self._model.predict_proba(X)[0][1])
            return float(np.clip(prob, 0.05, 0.95))
        except Exception as e:
            logger.warning("XGBoost predict failed: %s — using rule-based fallback", e)
            return _rule_based_prob(features)

    def train(self) -> bool:
        """Retrain on settled trades with recency weighting."""
        try:
            import joblib
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from xgboost import XGBClassifier
            from core.database import SessionLocal, TradeRow
            import json
            from datetime import datetime, timezone

            with SessionLocal() as session:
                rows = (
                    session.query(TradeRow)
                    .filter(TradeRow.outcome.in_(["WIN", "LOSS"]))
                    .all()
                )

            records = []
            for row in rows:
                try:
                    notes = json.loads(row.notes or "{}")
                    features = notes.get("features", {})
                    if not features or len(features) < 8:
                        continue
                    features["outcome"] = 1 if row.outcome == "WIN" else 0
                    features["exit_ts"] = (
                        row.exit_date.replace(tzinfo=timezone.utc).timestamp()
                        if row.exit_date else 0
                    )
                    records.append(features)
                except Exception:
                    continue

            if len(records) < MIN_SAMPLES:
                logger.warning("Not enough training data: %d/%d records", len(records), MIN_SAMPLES)
                return False

            df = pd.DataFrame(records)
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            for c in missing:
                df[c] = 0.0

            X = df[FEATURE_COLS].fillna(0.0)
            y = df["outcome"].astype(int)

            # Recency weighting: 30-day half-life
            now_ts = datetime.now(timezone.utc).timestamp()
            HALF_LIFE = 30.0 * 86400
            decay_rate = np.log(2) / HALF_LIFE
            settled_ts = df.get("exit_ts", pd.Series([now_ts] * len(df)))
            age_seconds = np.maximum(0, now_ts - settled_ts.values)
            sample_weights = np.exp(-decay_rate * age_seconds)

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                )),
            ])

            calibrated = CalibratedClassifierCV(pipeline, cv=3, method="isotonic")
            calibrated.fit(X, y, **{"clf__sample_weight": sample_weights})

            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(calibrated, self._model_path)
            self._model = calibrated
            self.is_trained = True

            logger.info("Retrained on %d records — saved to %s", len(records), self._model_path)
            return True

        except ImportError as e:
            logger.error("XGBoost/sklearn not installed: %s", e)
            return False
        except Exception as e:
            logger.error("Training failed: %s", e)
            return False


calibrator = StockCalibrator()
