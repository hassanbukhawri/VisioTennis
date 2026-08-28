"""
Accounts + backend-persisted history storage (Phase 2, 2026-08-07).

SQLite via SQLAlchemy -- both already installed in the backend's Python env
(confirmed before writing this), so this needed zero new dependencies. A
single new file, backend/app.db, deliberately SEPARATE from the existing
backend/logs/*.jsonl instrumentation logs (log_session_metrics/log_rejection
in backend/app.py) -- those are anonymous by explicit prior design and stay
that way permanently; see PROJECT_LOG.md's 2026-08-07 "two decisions
recorded before building" entry for the full reasoning. Do not add a
user_id column to anything under backend/logs/ without revisiting that
decision first.
"""
import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker


def utcnow() -> datetime:
    """SQLite's DateTime column strips tzinfo on read, so a value written as
    tz-aware comes back naive -- comparing that against a fresh tz-aware
    datetime.now(timezone.utc) raises TypeError (found the hard way, see
    PROJECT_LOG.md). Every datetime written to or compared against this DB
    goes through this helper instead, so everything stays naive-but-UTC and
    comparable, consistently."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    # Minimal profile fields -- only what's actually used somewhere in the
    # app today (playing_hand has no UI to set it yet, see PROJECT_LOG.md;
    # the column exists so adding that UI later is additive, not a migration).
    name = Column(String, nullable=True)
    playing_hand = Column(String, nullable=True)  # "left" | "right" | None
    created_at = Column(DateTime, nullable=False)

    # Coaching-tone profile (2026-08-10, backend/auth.py's ProfileUpdateRequest
    # / mobile/app/profile-setup.js) -- feeds rag/prompt.py:build_tone_instruction,
    # threaded into every live coaching-generation prompt (analyze-stroke +
    # the chatbot's tier-2/tier-3 paths, see backend/app.py). All nullable:
    # an account that hasn't set a role yet just gets no tone instruction
    # (existing default behavior, unchanged).
    #
    # NOT the same thing as the pre-existing on-device, anonymous
    # `role`/`setRole` in mobile/lib/historyRepository.js (feedback-form
    # question selection only, deliberately unlinked from any account) --
    # that system is untouched by this one. Two independently-named
    # player/coach toggles now exist in this app for different purposes;
    # flagged, not consolidated, per explicit scope.
    role = Column(String, nullable=True)  # "player" | "coach" | None
    skill_level = Column(String, nullable=True)  # "beginner"|"intermediate"|"advanced" -- role="player" only
    years_coaching = Column(Integer, nullable=True)  # role="coach" only
    certification_level = Column(String, nullable=True)  # role="coach" only, see auth.py CERTIFICATION_LEVELS

    # Coach avatar naming (2026-08-23, mobile/components/StrokeChat.js's CoachBubble) --
    # what the PLAYER named their AI coach persona, unrelated to this account's own
    # role above (a "coach"-role account still has a coach_name for the persona they
    # see in their own StrokeChat, same as a "player"-role account does). Nullable,
    # never required -- falls back to the literal string "Coach" wherever displayed,
    # never left blank and never defaulted from `name` (this account's own name).
    coach_name = Column(String, nullable=True)

    tokens = relationship("AuthToken", back_populates="user", cascade="all, delete-orphan")
    history = relationship("StrokeHistory", back_populates="user", cascade="all, delete-orphan")


class AuthToken(Base):
    __tablename__ = "auth_tokens"

    token = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    user = relationship("User", back_populates="tokens")


class StrokeHistory(Base):
    """One persisted analyze-stroke result. `id` is deliberately the same
    value as the response's own stroke_event.event_id -- that's already a
    unique, random, per-request id, so reusing it as the history row's PK
    avoids minting a second, redundant id for the same thing."""
    __tablename__ = "stroke_history"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)
    result_json = Column(Text, nullable=False)

    user = relationship("User", back_populates="history")


def init_db():
    Base.metadata.create_all(engine)
    _migrate_add_missing_columns()


def _migrate_add_missing_columns():
    """`Base.metadata.create_all` only creates tables that don't exist yet --
    it does NOT alter an existing table's columns, so adding role/skill_level/
    years_coaching/certification_level to the User model (2026-08-10) does
    nothing by itself against an already-created backend/app.db file. This
    repo has no migration tool (Alembic etc.) set up, and deleting/recreating
    the dev DB would destroy real accumulated accounts + stroke history from
    testing -- not acceptable for a one-line schema addition. Instead: a
    small, idempotent ALTER TABLE ADD COLUMN for any column the live table is
    actually missing, run on every startup. Safe to run repeatedly (checks
    PRAGMA table_info first, never re-adds an existing column); new columns
    land as NULL on every pre-existing row, which is exactly the intended
    default (see the User model's docstring -- no role means no tone
    instruction, not an error)."""
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return  # fresh DB, create_all() above already made it with every column
    existing_columns = {col["name"] for col in inspector.get_columns("users")}
    new_columns = {
        "role": "VARCHAR",
        "skill_level": "VARCHAR",
        "years_coaching": "INTEGER",
        "certification_level": "VARCHAR",
        "coach_name": "VARCHAR",
    }
    with engine.connect() as conn:
        for col_name, col_type in new_columns.items():
            if col_name not in existing_columns:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))
        conn.commit()


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
