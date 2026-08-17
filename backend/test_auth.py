"""
test_auth.py — real application authentication (auth.py + db/models.py's
User/UserSession + db/repository.py's user/session functions). Runs
entirely against the isolated SQLite fixture (conftest.py) — never
production PostgreSQL.
"""
import time

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import auth
import database


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def test_password_hash_is_never_the_plaintext():
    h = auth.hash_password("correct horse battery staple")
    assert h != "correct horse battery staple"
    assert h.startswith("$2b$") or h.startswith("$2a$")  # a real bcrypt hash


def test_password_hash_is_salted_two_hashes_of_same_password_differ():
    h1 = auth.hash_password("same-password-123")
    h2 = auth.hash_password("same-password-123")
    assert h1 != h2  # different salt each time — a real bcrypt property


def test_verify_password_accepts_correct_rejects_wrong():
    h = auth.hash_password("my-real-password-1")
    assert auth.verify_password("my-real-password-1", h) is True
    assert auth.verify_password("not-the-password", h) is False


def test_verify_password_never_raises_on_a_corrupt_stored_hash():
    assert auth.verify_password("anything", "not-a-real-bcrypt-hash") is False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_register_creates_a_real_user_and_returns_a_usable_token():
    user, token = auth.register_user("alice@example.com", "hunter2hunter2")
    assert user["email"] == "alice@example.com"
    assert "password_hash" not in user  # never leaked back to the caller
    assert len(token) > 20
    resolved = auth.get_user_from_token(token)
    assert resolved["user_id"] == user["user_id"]


def test_register_normalizes_email_case_and_whitespace():
    user, _ = auth.register_user("  Bob@Example.com  ", "hunter2hunter2")
    assert user["email"] == "bob@example.com"


def test_register_rejects_a_duplicate_email():
    auth.register_user("carol@example.com", "hunter2hunter2")
    with pytest.raises(auth.AuthError):
        auth.register_user("carol@example.com", "different-password")


def test_register_rejects_a_duplicate_email_case_insensitively():
    auth.register_user("dave@example.com", "hunter2hunter2")
    with pytest.raises(auth.AuthError):
        auth.register_user("DAVE@EXAMPLE.COM", "different-password")


def test_register_rejects_an_invalid_email():
    with pytest.raises(auth.AuthError):
        auth.register_user("not-an-email", "hunter2hunter2")


def test_register_rejects_a_short_password():
    with pytest.raises(auth.AuthError):
        auth.register_user("shortpw@example.com", "short")


def test_register_never_stores_the_plaintext_password_anywhere():
    auth.register_user("plain@example.com", "a-real-secret-99")
    row = database.get_user_by_email("plain@example.com")
    assert row["password_hash"] != "a-real-secret-99"
    assert "a-real-secret-99" not in row["password_hash"]


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
def test_login_with_correct_credentials_succeeds():
    auth.register_user("erin@example.com", "correctpassword1")
    user, token = auth.login_user("erin@example.com", "correctpassword1")
    assert user["email"] == "erin@example.com"
    assert auth.get_user_from_token(token) is not None


def test_login_with_wrong_password_fails():
    auth.register_user("frank@example.com", "correctpassword1")
    with pytest.raises(auth.AuthError):
        auth.login_user("frank@example.com", "wrongpassword1")


def test_login_with_unknown_email_fails_with_the_same_message_as_wrong_password():
    """A real information-leak check: the app must not let a client learn
    whether an email is registered by comparing error messages."""
    auth.register_user("grace@example.com", "correctpassword1")
    try:
        auth.login_user("grace@example.com", "wrongpassword1")
        assert False, "expected AuthError"
    except auth.AuthError as e:
        wrong_password_message = str(e)
    try:
        auth.login_user("never-registered@example.com", "anypassword1")
        assert False, "expected AuthError"
    except auth.AuthError as e:
        unknown_email_message = str(e)
    assert wrong_password_message == unknown_email_message


def test_login_is_case_insensitive_on_email():
    auth.register_user("henry@example.com", "correctpassword1")
    user, _ = auth.login_user("HENRY@EXAMPLE.COM", "correctpassword1")
    assert user["email"] == "henry@example.com"


# ---------------------------------------------------------------------------
# Tokens / sessions
# ---------------------------------------------------------------------------
def test_each_login_issues_a_distinct_token():
    auth.register_user("ivan@example.com", "correctpassword1")
    _, token_a = auth.login_user("ivan@example.com", "correctpassword1")
    _, token_b = auth.login_user("ivan@example.com", "correctpassword1")
    assert token_a != token_b
    # both remain independently valid — logging in again must not revoke
    # a still-active session on another device.
    assert auth.get_user_from_token(token_a) is not None
    assert auth.get_user_from_token(token_b) is not None


def test_an_unknown_token_resolves_to_no_user():
    assert auth.get_user_from_token("this-token-was-never-issued") is None


def test_a_token_is_never_stored_in_the_database_only_its_hash():
    _, token = auth.register_user("judy@example.com", "correctpassword1")
    # The raw token must not appear anywhere in the session table.
    import database as db
    from db.session import session_scope
    from db.models import UserSession
    with session_scope() as session:
        rows = session.query(UserSession).all()
        for row in rows:
            assert row.token_hash != token
            assert token not in row.token_hash


def test_an_expired_token_is_rejected(monkeypatch):
    _, token = auth.register_user("kate@example.com", "correctpassword1")
    # Force the session's expiry into the past directly at the data layer
    # (real expiry check, not a mocked clock) — mirrors what a genuinely
    # old token looks like after TOKEN_EXPIRY_HOURS has elapsed.
    from db.session import session_scope
    from db.models import UserSession
    token_hash = auth._hash_token(token)
    with session_scope() as session:
        row = session.get(UserSession, token_hash)
        row.expires_at = "2000-01-01T00:00:00+00:00"
    assert auth.get_user_from_token(token) is None


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------
def test_logout_immediately_invalidates_that_session_token():
    _, token = auth.register_user("liam@example.com", "correctpassword1")
    assert auth.get_user_from_token(token) is not None
    auth.logout_user(token)
    assert auth.get_user_from_token(token) is None


def test_logout_does_not_invalidate_a_different_session_for_the_same_user():
    auth.register_user("maria@example.com", "correctpassword1")
    _, token_a = auth.login_user("maria@example.com", "correctpassword1")
    _, token_b = auth.login_user("maria@example.com", "correctpassword1")
    auth.logout_user(token_a)
    assert auth.get_user_from_token(token_a) is None
    assert auth.get_user_from_token(token_b) is not None


def test_logout_of_an_unknown_token_does_not_raise():
    auth.logout_user("never-issued-token")  # must be a safe no-op


# ---------------------------------------------------------------------------
# get_current_user FastAPI dependency
# ---------------------------------------------------------------------------
def test_get_current_user_with_no_credentials_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        auth.get_current_user(None)
    assert exc_info.value.status_code == 401


def test_get_current_user_with_an_invalid_token_raises_401():
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="not-a-real-token")
    with pytest.raises(HTTPException) as exc_info:
        auth.get_current_user(creds)
    assert exc_info.value.status_code == 401


def test_get_current_user_with_a_valid_token_returns_the_real_user():
    user, token = auth.register_user("nina@example.com", "correctpassword1")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    resolved = auth.get_current_user(creds)
    assert resolved["user_id"] == user["user_id"]
    assert resolved["email"] == "nina@example.com"


def test_get_current_user_401_never_leaks_whether_the_token_ever_existed():
    """Same generic message for 'never issued' vs 'expired' — see
    get_current_user()'s docstring for why this matters."""
    creds_unknown = HTTPAuthorizationCredentials(scheme="Bearer", credentials="never-issued")
    try:
        auth.get_current_user(creds_unknown)
        assert False
    except HTTPException as e:
        unknown_detail = e.detail

    _, token = auth.register_user("oscar@example.com", "correctpassword1")
    auth.logout_user(token)  # now a real, but revoked, token
    creds_revoked = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    try:
        auth.get_current_user(creds_revoked)
        assert False
    except HTTPException as e:
        revoked_detail = e.detail

    assert unknown_detail == revoked_detail


# ---------------------------------------------------------------------------
# First-user bootstrap (workspace ownership claim)
# ---------------------------------------------------------------------------
def test_bootstrap_claim_assigns_every_ownerless_workspace_to_the_given_user():
    """Exercises database.bootstrap_claim_orphaned_workspaces() directly —
    the exact function auth.register_user() calls when (and only when)
    count_users() == 0 at registration time. Tested directly rather than
    through the full register_user() "is this really the first user ever"
    gate, since that gate is a real, correct, one-time-per-deployment
    condition that a shared test database (many tests registering many
    users across this whole file) can't reliably reproduce per-test."""
    database.create_workspace("legacy-ws-1", "Pre-auth workspace")
    database.create_workspace("legacy-ws-1b", "Another pre-auth workspace")
    assert database.get_workspace("legacy-ws-1")["owner_user_id"] is None
    assert database.get_workspace("legacy-ws-1b")["owner_user_id"] is None

    user, _ = auth.register_user("bootstrap-target@example.com", "correctpassword1")
    claimed_count = database.bootstrap_claim_orphaned_workspaces(user["user_id"])

    assert claimed_count >= 2  # at least the two just created (others may exist from earlier tests in this file)
    assert database.get_workspace("legacy-ws-1")["owner_user_id"] == user["user_id"]
    assert database.get_workspace("legacy-ws-1b")["owner_user_id"] == user["user_id"]


def test_bootstrap_claim_never_reassigns_a_workspace_that_already_has_an_owner():
    user_a, _ = auth.register_user("real-owner-2@example.com", "correctpassword1")
    database.create_workspace("owned-ws-2", "Already owned", owner_user_id=user_a["user_id"])

    user_b, _ = auth.register_user("someone-else-2@example.com", "correctpassword1")
    database.bootstrap_claim_orphaned_workspaces(user_b["user_id"])

    assert database.get_workspace("owned-ws-2")["owner_user_id"] == user_a["user_id"]


def test_register_user_only_runs_the_bootstrap_claim_when_it_is_truly_the_first_account():
    """The real end-to-end wiring: register_user() must NOT auto-claim
    ownerless workspaces for the second (or later) account on a
    deployment — only ever for the very first one."""
    database.create_workspace("legacy-ws-2", "Pre-existing, should stay unclaimed")
    # Ensure at least one user already exists so the next registration is
    # provably not "the first user ever" in this database.
    auth.register_user("already-here@example.com", "correctpassword1")

    late_user, _ = auth.register_user("late-comer@example.com", "correctpassword1")

    assert database.get_workspace("legacy-ws-2")["owner_user_id"] is None
    assert database.get_workspace("legacy-ws-2")["owner_user_id"] != late_user["user_id"]
