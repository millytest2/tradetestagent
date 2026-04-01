"""SQLite persistence layer using SQLAlchemy."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    desc,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings
from core.models import PostmortemFinding, Trade, TradeOutcome, TradeStatus


# ── ORM Base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class TradeRow(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, index=True)
    company_name = Column(String, default="")
    side = Column(String)
    entry_price = Column(Float)
    bet_usd = Column(Float)
    shares = Column(Float)
    stop_loss_price = Column(Float, default=0.0)
    take_profit_price = Column(Float, default=0.0)
    status = Column(String, default="PLACED")
    outcome = Column(String, default="PENDING")
    pnl_usd = Column(Float, default=0.0)
    alpaca_order_id = Column(String, default="")
    entry_date = Column(DateTime, default=datetime.utcnow)
    exit_date = Column(DateTime, nullable=True)
    notes = Column(Text, default="")


class PostmortemRow(Base):
    __tablename__ = "postmortems"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, index=True)
    agent_name = Column(String)
    finding = Column(Text)
    root_cause = Column(Text)
    recommendation = Column(Text)
    severity = Column(String, default="medium")
    created_at = Column(DateTime, default=datetime.utcnow)


class LessonRow(Base):
    __tablename__ = "lessons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String, index=True)
    lesson = Column(Text)
    source_trade_id = Column(Integer, nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemUpdateRow(Base):
    __tablename__ = "system_updates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    update_type = Column(String)
    description = Column(Text)
    payload_json = Column(Text, default="{}")
    applied = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Engine & Session ──────────────────────────────────────────────────────────

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()


# ── Trade CRUD ────────────────────────────────────────────────────────────────

def save_trade(trade: Trade) -> int:
    with get_session() as session:
        row = TradeRow(
            ticker=trade.ticker,
            company_name=trade.company_name,
            side=trade.side.value,
            entry_price=trade.entry_price,
            bet_usd=trade.bet_usd,
            shares=trade.shares,
            stop_loss_price=trade.stop_loss_price,
            take_profit_price=trade.take_profit_price,
            status=trade.status.value,
            outcome=trade.outcome.value,
            pnl_usd=trade.pnl_usd,
            alpaca_order_id=trade.alpaca_order_id,
            entry_date=trade.entry_date,
            exit_date=trade.exit_date,
            notes=trade.notes,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


def update_trade_outcome(
    trade_id: int,
    outcome: TradeOutcome,
    pnl_usd: float,
    exit_date: Optional[datetime] = None,
) -> None:
    with get_session() as session:
        row = session.get(TradeRow, trade_id)
        if row:
            row.outcome = outcome.value
            row.status = TradeStatus.SETTLED.value
            row.pnl_usd = pnl_usd
            row.exit_date = exit_date or datetime.utcnow()
            session.commit()


def get_losing_trades(limit: int = 10) -> list[TradeRow]:
    with get_session() as session:
        return (
            session.query(TradeRow)
            .filter(TradeRow.outcome == TradeOutcome.LOSS.value)
            .order_by(desc(TradeRow.entry_date))
            .limit(limit)
            .all()
        )


def get_open_positions() -> list[TradeRow]:
    """Return all trades with PENDING outcome (currently open)."""
    with get_session() as session:
        return (
            session.query(TradeRow)
            .filter(TradeRow.outcome == TradeOutcome.PENDING.value)
            .order_by(desc(TradeRow.entry_date))
            .all()
        )


def count_open_positions() -> int:
    with get_session() as session:
        return (
            session.query(TradeRow)
            .filter(TradeRow.outcome == TradeOutcome.PENDING.value)
            .count()
        )


def get_trade_stats() -> dict:
    with get_session() as session:
        total = session.query(TradeRow).count()
        wins = session.query(TradeRow).filter(TradeRow.outcome == "WIN").count()
        losses = session.query(TradeRow).filter(TradeRow.outcome == "LOSS").count()
        pending = session.query(TradeRow).filter(TradeRow.outcome == "PENDING").count()
        pnl_rows = session.query(TradeRow).all()
        total_pnl = sum(r.pnl_usd for r in pnl_rows)
        win_rate = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0
        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": win_rate,
            "total_pnl_usd": total_pnl,
        }


# ── Postmortem CRUD ───────────────────────────────────────────────────────────

def save_postmortem_finding(finding: PostmortemFinding) -> int:
    with get_session() as session:
        row = PostmortemRow(
            trade_id=finding.trade_id,
            agent_name=finding.agent_name,
            finding=finding.finding,
            root_cause=finding.root_cause,
            recommendation=finding.recommendation,
            severity=finding.severity,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


def save_lesson(category: str, lesson: str, trade_id: Optional[int] = None) -> None:
    with get_session() as session:
        row = LessonRow(
            category=category,
            lesson=lesson,
            source_trade_id=trade_id,
        )
        session.add(row)
        session.commit()


def get_active_lessons(category: Optional[str] = None, limit: int = 20) -> list[str]:
    with get_session() as session:
        q = session.query(LessonRow).filter(LessonRow.active == True)
        if category:
            q = q.filter(LessonRow.category == category)
        rows = q.order_by(desc(LessonRow.created_at)).limit(limit).all()
        return [r.lesson for r in rows]


def purge_stale_lessons() -> int:
    """Deactivate generic placeholder lessons. Safe to run repeatedly."""
    stale_phrases = [
        "Review process",
        "Investigate agent failure",
        "review process",
        "investigate agent failure",
    ]
    count = 0
    with get_session() as session:
        rows = session.query(LessonRow).filter(LessonRow.active == True).all()
        for row in rows:
            if any(phrase in (row.lesson or "") for phrase in stale_phrases):
                row.active = False
                count += 1
        session.commit()
    return count


def save_system_update(update_type: str, description: str, payload: dict) -> None:
    with get_session() as session:
        row = SystemUpdateRow(
            update_type=update_type,
            description=description,
            payload_json=json.dumps(payload),
        )
        session.add(row)
        session.commit()
