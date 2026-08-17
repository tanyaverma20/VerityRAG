"""
auth.py — real application authentication.

Design, stated plainly:

  - Passwords are hashed with bcrypt (a real, slow, salted KDF — never
    stored, logged, or returned in plaintext anywhere, including in this
    module's own return values; see db/repository.py's get_user_by_email()
    docstring for the one place the hash is even read back).
  - Sessions are opaque, random 256-bit bearer tokens (`secrets.token_urlsafe`),
    NOT a JWT — chosen deliberately so "logout" is a real, immediate,
    server-side DELETE of the session row (db/repository.py:
    delete_user_session()), not just "the client discarded a token that
    remains valid until it expires," which is what a stateless JWT gives
    you. Only the SHA-256 hash of the token is ever stored in the database
    (db/models.py: UserSession.token_hash) — a database read (backup, dump,
    compromise) never exposes a directly-usable token, mirroring how
    password_hash never stores a reversible password.
  - Every protected endpoint depends on get_current_user(), which resolves
    the token from the `Authorization: Bearer <token>` header, hashes it,
    looks up the (non-expired) session, and returns the real authenticated
    User row — or raises a clean 401 with no internal detail leaked.
    Nothing downstream is ever allowed to trust a client-supplied
    workspace_id as a security principal; every authorization check in
    main.py checks the row's real owner_user_id against
    get_current_user()'s result instead (see main.py's
    _require_workspace_owner()).
"""
from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import database

# How long a session token stays valid after issue. Configurable so a real
# deployment can tighten it; 24h is a reasonable default for a research tool
# that isn't handling financial/health data. Re-authenticating simply issues
# a fresh token — there is no silent auto-refresh.
import os
TOKEN_EXPIRY_HOURS = int(os.getenv("AUTH_TOKEN_EXPIRY_HOURS", "24"))

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8

_bearer_scheme = HTTPBearer(auto_error=False)


class AuthError(Exception):
    """Raised for any auth failure whose message is safe to return to the
    client as-is (never a stack trace, never an internal detail)."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_email(email: str) -> None:
    if not email or not _EMAIL_RE.match(email):
        raise AuthError("Please enter a valid email address.")


def validate_password(password: str) -> None:
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")


def hash_password(password: str) -> str:
    # bcrypt has a real 72-byte input limit; truncate deterministically
    # rather than let bcrypt silently ignore bytes past it or raise on an
    # unusually long (but not necessarily malicious) password.
    raw = password.encode("utf-8")[:72]
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        raw = password.encode("utf-8")[:72]
        return bcrypt.checkpw(raw, password_hash.encode("ascii"))
    except (ValueError, TypeError):
        # A malformed/corrupt stored hash must fail closed, never raise
        # into a 500 that could hint at why.
        return False


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def issue_session_token(user_id: str) -> str:
    """Creates a new session row and returns the RAW token — the only time
    the raw value ever exists outside the client's memory; only its hash is
    persisted."""
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS)).isoformat()
    database.create_user_session(_hash_token(token), user_id, expires_at)
    return token


def register_user(email: str, password: str) -> tuple[dict, str]:
    """Real registration: validates input, rejects a duplicate email with a
    clean 409-shaped error (never a raw IntegrityError/stack trace), hashes
    the password, creates the user, and — ONLY if this is the very first
    user account ever created on this deployment — claims every
    pre-existing ownerless workspace for them (see
    db/repository.py:_bootstrap_claim_orphaned_workspaces() for exactly why
    and its safety bound). Returns (user_dict, raw_session_token)."""
    email = normalize_email(email)
    validate_email(email)
    validate_password(password)

    if database.get_user_by_email(email):
        raise AuthError("An account with this email already exists.")

    is_first_user = database.count_users() == 0

    user_id = secrets.token_hex(12)
    password_hash = hash_password(password)
    database.create_user(user_id, email, password_hash)

    if is_first_user:
        claimed = database.bootstrap_claim_orphaned_workspaces(user_id)
        if claimed:
            print(f"[auth] First account registered ({email}) — claimed {claimed} pre-existing ownerless workspace(s).")

    token = issue_session_token(user_id)
    return database.get_user_by_id(user_id), token


def login_user(email: str, password: str) -> tuple[dict, str]:
    """Real login: constant-shaped failure for both "no such user" and
    "wrong password" (never reveals which one it was — a real information
    leak most login flows get wrong)."""
    email = normalize_email(email)
    row = database.get_user_by_email(email)
    if not row or not verify_password(password, row["password_hash"]):
        raise AuthError("Incorrect email or password.")
    token = issue_session_token(row["user_id"])
    return database.get_user_by_id(row["user_id"]), token


def logout_user(token: str) -> None:
    database.delete_user_session(_hash_token(token))


def get_user_from_token(token: str) -> Optional[dict]:
    return database.get_user_by_token_hash(_hash_token(token))


def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme)) -> dict:
    """The real FastAPI dependency every protected endpoint uses. Returns
    the authenticated user dict ({"user_id", "email", "created_at"} — never
    password_hash) or raises a clean 401 with no internal detail — missing
    header, malformed header, unknown token, and expired token all produce
    the exact same generic message, so a client can't distinguish "this
    token never existed" from "this token expired" (nothing useful for an
    attacker to learn either way)."""
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Authentication required.", headers={"WWW-Authenticate": "Bearer"})
    user = get_user_from_token(creds.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session.", headers={"WWW-Authenticate": "Bearer"})
    return user
