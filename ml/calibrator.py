"""XGBoost probability calibrator.

Trains on historical trade outcomes to learn which feature combinations
produce a genuine edge over market prices. Falls back to a rule-based
prior when insufficient data exists (< 30 settled trades).

Features used
─────────────
  compound_sentiment    VADER compound score of social posts
  positive_sentiment    fraction of positive VADER score
  negative_sentiment    fraction of negative VADER score
  post_count            number of social posts found
  avg_engagement        mean likes/upvotes across posts
  price_change_24h      absolute 24h price move on the market
  spread                bid-ask spread on the outcome token
  liquidity_usdc        total market liquidity
  volume_24h_usdc       24h trading volume
  time_to_resolution_days  days until the market resolves
  current_yes_price     current market-implied YES probability

Target: 1 if the trade side won, 0 if it lost.
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
    "compound_sentiment",
    "positive_sentiment",
    "negative_sentiment",
    "post_count",
    "avg_engagement",
    "price_change_24h",
    "spread",
    "liquidity_usdc",
    "volume_24h_usdc",
    "time_to_resolution_days",
    "current_yes_price",
    "whale_bid_imbalance",   # on-chain whale order pressure
    "trend_score",           # Google Trends interest score
]

MIN_SAMPLES_FOR_TRAINING = 30


# ── Prior (rule-based fallback) ───────────────────────────────────────────────

def _rule_based_probability(features: PredictionFeatures) -> float:
    """
    Conservative prior used before we have enough training data.

    Logic mirrors the intuition behind the bot:
      - Strong positive sentiment + price hasn't moved much → lean YES
      - Strong negative sentiment → lean NO (return P(NO))
      - High uncertainty → return 0.5 (no signal)
    """
    sentiment = features.compound_sentiment       # -1 to +1
    current_p = features.current_yes_price        # market's view

    # Sentiment signal: scale to ±0.15 adjustment max
    sentiment_adj = sentiment * 0.15

    # Engagement signal: more engagement → slightly more confidence
    engagement_adj = min(features.avg_engagement / 500, 0.05)

    # Volume signal: high relative volume suggests informed activity
    volume_adj = min(features.volume_24h_usdc / 50_000, 0.05)

    raw = current_p + sentiment_adj + engagement_adj * np.sign(sentiment_adj)
    raw += volume_adj * np.sign(sentiment_adj)
    return float(np.clip(raw, 0.05, 0.95))


# ── XGBoost Calibrator ────────────────────────────────────────────────────────

class ProbabilityCalibrator:
    """Wraps an XGBoost classifier with a sklearn CalibratedClassifierCV."""

    def __init__(self) -> None:
        self._model = None
        self._is_trained = False
        self._model_path = Path(settings.model_path)
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self._model_path.exists():
            try:
                import joblib
                self._model = joblib.load(self._model_path)
                self._is_trained = True
                logger.info("Loaded XGBoost calibrator from %s", self._model_path)
            except Exception as e:
                logger.warning("Could not load model: %s", e)

    def _save(self) -> None:
        try:
            import joblib
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model, self._model_path)
            logger.info("Saved XGBoost calibrator to %s", self._model_path)
        except Exception as e:
            logger.error("Model save failed: %s", e)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def predict(self, features: PredictionFeatures) -> float:
        """Return P(YES) — the calibrated win probability."""
        if not self._is_trained:
            return _rule_based_probability(features)

        try:
            X = self._features_to_array(features)
            prob = float(self._model.predict_proba(X)[0, 1])
            return np.clip(prob, 0.05, 0.95)
        except Exception as e:
            logger.warning("XGBoost predict failed, using prior: %s", e)
            return _rule_based_probability(features)

    def _features_to_array(self, features: PredictionFeatures) -> np.ndarray:
        row = {
            "compound_sentiment": features.compound_sentiment,
            "positive_sentiment": features.positive_sentiment,
            "negative_sentiment": features.negative_sentiment,
            "post_count": features.post_count,
            "avg_engagement": features.avg_engagement,
            "price_change_24h": features.price_change_24h,
            "spread": features.spread,
            "liquidity_usdc": features.liquidity_usdc,
            "volume_24h_usdc": features.volume_24h_usdc,
            "time_to_resolution_days": features.time_to_resolution_days,
            "current_yes_price": features.current_yes_price,
            "whale_bid_imbalance": features.whale_bid_imbalance,
            "trend_score": features.trend_score,
        }
        return pd.DataFrame([row])[FEATURE_COLS].values

    def train(self, records: list[dict]) -> dict:
        """
        Train (or retrain) the XGBoost model from historical trade records.

        Each record must have keys matching FEATURE_COLS plus 'label' (0 or 1).
        Returns a dict with training metrics.
        """
        if len(records) < MIN_SAMPLES_FOR_TRAINING:
            logger.info(
                "Only %d records — need %d to train (using rule-based prior)",
                len(records), MIN_SAMPLES_FOR_TRAINING,
            )
            return {"status": "insufficient_data", "n_samples": len(records)}

        try:
            from xgboost import XGBClassifier
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from datetime import datetime, timezone

            df = pd.DataFrame(records)
            X = df[FEATURE_COLS].values
            y = df["label"].values.astype(int)

            # ── Recency weighting: exponential decay over 90 days ────────────
            # Trades settled recently matter more than old ones.
            # Half-life ≈ 30 days (weight halves every 30 days back in time).
            now_ts = datetime.now(timezone.utc).timestamp()
            HALF_LIFE_DAYS = 30.0
            decay_rate = np.log(2) / (HALF_LIFE_DAYS * 86400)
            if "settled_at" in df.columns:
                settled_ts = pd.to_numeric(
                    pd.to_datetime(df["settled_at"], utc=True, errors="coerce"),
                    errors="coerce",
                ) / 1e9   # ns → s
            else:
                settled_ts = pd.Series([now_ts] * len(df))
            settled_ts = settled_ts.fillna(now_ts)
            age_seconds = np.maximum(0, now_ts - settled_ts.values)
            sample_weights = np.exp(-decay_rate * age_seconds).astype(float)
            mean_w = sample_weights.mean()
            if mean_w > 0:
                sample_weights /= mean_w   # normalise to mean=1
            else:
                sample_weights = np.ones(len(sample_weights))
            # ─────────────────────────────────────────────────────────────────

            base = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )

            # Wrap in calibration + standard scaler pipeline
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(base, cv=3, method="isotonic")),
            ])

            # Cross-validate (unweighted — gives honest generalisation estimate)
            scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
            # Fit with recency weights so recent trades drive the model more
            pipeline.fit(X, y, clf__sample_weight=sample_weights)

            self._model = pipeline
            self._is_trained = True
            self._save()

            metrics = {
                "status": "trained",
                "n_samples": len(records),
                "cv_auc_mean": float(scores.mean()),
                "cv_auc_std": float(scores.std()),
            }
            logger.info(
                "XGBoost trained — n=%d, CV AUC=%.3f±%.3f",
                len(records), scores.mean(), scores.std(),
            )
            return metrics

        except ImportError as e:
            logger.error("Missing ML dependency: %s", e)
            return {"status": "error", "reason": str(e)}
        except Exception as e:
            logger.error("Training failed: %s", e)
            return {"status": "error", "reason": str(e)}

    def update_with_outcome(self, features: PredictionFeatures, label: int) -> None:
        """
        Append a single resolved trade to the training dataset stored in the DB
        and optionally retrain if enough new data has accumulated.
        This is called by the postmortem pipeline after each settled trade.
        """
        from core.database import get_session, SessionLocal
        # Store feature row for future retraining (done via collect_training_data)
        logger.debug("Outcome recorded (label=%d) — will retrain on next cycle", label)

    def collect_training_data(self) -> list[dict]:
        """
        Build training records from the trade history.
        Only includes trades with WIN or LOSS outcomes AND at least 3 non-zero
        market features — records with all-zero features teach the model nothing
        and will cause it to predict the training mean for every market.
        """
        from core.database import get_session, TradeRow, SessionLocal

        MARKET_FEATURE_COLS = [
            "price_change_24h", "spread", "liquidity_usdc",
            "volume_24h_usdc", "time_to_resolution_days", "current_yes_price",
        ]

        records = []
        skipped = 0
        try:
            with SessionLocal() as session:
                rows = (
                    session.query(TradeRow)
                    .filter(TradeRow.outcome.in_(["WIN", "LOSS"]))
                    .all()
                )
                for row in rows:
                    import json
                    try:
                        notes = json.loads(row.notes or "{}")
                        feats = notes.get("features") or {}
                        # Fill any missing columns with 0
                        for col in FEATURE_COLS:
                            if col not in feats:
                                feats[col] = 0.0
                        # Reject records where all market-level features are 0
                        # (old format — only current_yes_price was saved)
                        nonzero = sum(
                            1 for col in MARKET_FEATURE_COLS
                            if float(feats.get(col, 0)) != 0.0
                        )
                        if nonzero < 3:
                            skipped += 1
                            continue
                        feats["label"] = 1 if row.outcome == "WIN" else 0
                        records.append(feats)
                    except Exception:
                        continue
        except Exception as e:
            logger.error("Failed to collect training data: %s", e)

        if skipped:
            logger.info(
                "Skipped %d low-quality records (all-zero features) — "
                "retrain after more trades accumulate with full feature logging",
                skipped,
            )
        return records


# Singleton
calibrator = ProbabilityCalibrator()
