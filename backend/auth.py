"""
Auth helpers for Phase 2 (accounts + backend-persisted history, 2026-08-07).

Deliberately simple, per the approved plan (PROJECT_LOG.md): email +
password, opaque server-side session tokens -- not JWT. An opaque random
token stored in the auth_tokens table needs no signing-key material to
manage/rotate, and is trivially revocable on logout (delete the row),
unlike a JWT which stays valid until it expires even if "logged out."
Appropriate for a small trial, not a production-scale identity system.

Password hashing uses bcrypt directly (already installed; passlib is not
and was deliberately not added just to wrap the same library). Email
format is checked with a plain regex rather than pydantic's EmailStr,
which would require adding the email_validator package -- not worth a new
dependency for a trial-scale signup form.
"""
import re
import secrets
from datetime import timedelta
from typing import Literal

import bcrypt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, field_validator, model_validator
from sqlalchemy.orm import Session

from backend.db import AuthToken, User, get_db_session, utcnow

# Coaching-tone profile (2026-08-10) -- see backend/db.py's User model
# docstring for how these feed rag/prompt.py:build_tone_instruction.
SKILL_LEVELS = ("beginner", "intermediate", "advanced")
# A small fixed set rather than free text, so the stored value is always one
# of a known handful -- "other" is the deliberate escape hatch for anything
# that doesn't fit the first three, not a sign the list is incomplete.
CERTIFICATION_LEVELS = ("club_coach", "national", "itf_certified", "other")

TOKEN_TTL = timedelta(days=30)  # trial-scale, personal test devices -- see PROJECT_LOG.md
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_bearer_scheme = HTTPBearer(auto_error=False)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def issue_token(db: Session, user: User) -> AuthToken:
    now = utcnow()
    token_row = AuthToken(
        token=secrets.token_urlsafe(32),
        user_id=user.id,
        created_at=now,
        expires_at=now + TOKEN_TTL,
    )
    db.add(token_row)
    db.commit()
    db.refresh(token_row)
    return token_row


def get_current_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: Session = Depends(get_db_session),
) -> AuthToken:
    """FastAPI dependency: resolves the Authorization: Bearer <token> header
    to its AuthToken row (with .user available via relationship), 401s on
    anything wrong. Also lazily deletes an expired token on the request
    that discovers it -- no background cleanup job needed at trial scale."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="missing or invalid Authorization header")
    token_row = db.get(AuthToken, credentials.credentials)
    if token_row is None:
        raise HTTPException(status_code=401, detail="invalid or expired session -- please log in again")
    if token_row.expires_at < utcnow():
        db.delete(token_row)
        db.commit()
        raise HTTPException(status_code=401, detail="invalid or expired session -- please log in again")
    return token_row


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str | None = None

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v):
        v = normalize_email(v)
        if not _EMAIL_RE.match(v):
            raise ValueError("not a valid email address")
        return v

    @field_validator("password")
    @classmethod
    def _valid_password(cls, v):
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _normalize(cls, v):
        return normalize_email(v)


class ProfileUpdateRequest(BaseModel):
    """Body for PATCH /api/v1/auth/profile (mobile/app/profile-setup.js).
    role decides which of the two follow-up fields are required -- a coach
    has no skill_level (coach tone is a flat flag, not scaled by experience,
    see rag/prompt.py:build_tone_instruction's docstring for why
    years_coaching/certification_level are collected but not used to scale
    tone), a player has no years_coaching/certification_level. Enforced here,
    not left to the client, so a malformed profile can never reach the DB."""
    role: Literal["player", "coach"]
    skill_level: Literal["beginner", "intermediate", "advanced"] | None = None
    years_coaching: int | None = None
    certification_level: Literal["club_coach", "national", "itf_certified", "other"] | None = None

    @model_validator(mode="after")
    def _check_consistency(self):
        if self.role == "player":
            if self.skill_level is None:
                raise ValueError("skill_level is required when role is 'player'")
            if self.years_coaching is not None or self.certification_level is not None:
                raise ValueError("years_coaching/certification_level must not be set when role is 'player'")
        else:  # coach
            if self.years_coaching is None or self.certification_level is None:
                raise ValueError("years_coaching and certification_level are required when role is 'coach'")
            if self.years_coaching < 0:
                raise ValueError("years_coaching must not be negative")
            if self.skill_level is not None:
                raise ValueError("skill_level must not be set when role is 'coach'")
        return self


class UserOut(BaseModel):
    id: str
    email: str
    name: str | None
    playing_hand: str | None
    role: str | None
    skill_level: str | None
    years_coaching: int | None
    certification_level: str | None
    coach_name: str | None


class AuthResponse(BaseModel):
    token: str
    user: UserOut


def user_to_out(user: User) -> UserOut:
    return UserOut(
        id=user.id, email=user.email, name=user.name, playing_hand=user.playing_hand,
        role=user.role, skill_level=user.skill_level,
        years_coaching=user.years_coaching, certification_level=user.certification_level,
        coach_name=user.coach_name,
    )


class CoachNameUpdateRequest(BaseModel):
    """Body for PATCH /api/v1/auth/coach-name (mobile Settings' 'Coaching
    profile' card). Deliberately separate from ProfileUpdateRequest above --
    coach_name is a per-user preference unrelated to the role-gated
    coaching-tone fields, and ProfileUpdateRequest requires role to already
    be set, which coach_name must not depend on. Blank/whitespace-only input
    normalizes to None here (server-side, not left to the client) so the
    empty-string case can never be stored as a real name and displayed
    literally instead of falling back to "Coach"."""
    coach_name: str | None = None

    @field_validator("coach_name")
    @classmethod
    def _blank_to_none(cls, v):
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None


def user_tone_profile(user: User) -> dict:
    """The subset of a user's profile rag/prompt.py:build_tone_instruction
    actually reads -- built here (not inline at each call site in
    backend/app.py) so every prompt-construction call site pulls the tone
    profile the same way."""
    return {
        "role": user.role, "skill_level": user.skill_level,
        "years_coaching": user.years_coaching, "certification_level": user.certification_level,
    }
