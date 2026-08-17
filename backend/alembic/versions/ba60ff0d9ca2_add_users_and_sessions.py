"""add users, user_sessions, and workspaces.owner_user_id

Revision ID: ba60ff0d9ca2
Revises: b749c33c96ec
Create Date: 2026-08-13

Real application authentication (see auth.py, db/models.py). Adds:
  - users: user_id, email (unique), password_hash (bcrypt — never
    plaintext), created_at.
  - user_sessions: opaque bearer-token sessions, keyed by the SHA-256
    hash of the token (never the raw token itself), with a real
    expires_at so a session genuinely stops being valid, and real
    server-side logout (DELETE the row) rather than relying on a
    stateless JWT's client-side-only "discard the token."
  - workspaces.owner_user_id: nullable (existing workspaces created
    before authentication existed have no owner until claimed — see
    db/repository.py's bootstrap_claim_orphaned_workspaces(), invoked
    once, automatically, the moment the very first user account is ever
    registered on a deployment) but required going forward for every
    newly created workspace (enforced in main.py, not at the schema
    level, so a pre-existing NULL-owner row is never itself an error).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ba60ff0d9ca2'
down_revision: Union[str, Sequence[str], None] = 'b749c33c96ec'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'users',
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('password_hash', sa.String(), nullable=False),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('user_id'),
        sa.UniqueConstraint('email', name='uq_users_email'),
    )
    op.create_table(
        'user_sessions',
        sa.Column('token_hash', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('expires_at', sa.String(), nullable=False),
        sa.Column('last_used_at', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('token_hash'),
    )
    op.create_index('ix_user_sessions_user_id', 'user_sessions', ['user_id'], unique=False)
    op.create_index('ix_user_sessions_expires_at', 'user_sessions', ['expires_at'], unique=False)

    op.add_column('workspaces', sa.Column('owner_user_id', sa.String(), nullable=True))
    op.create_index('ix_workspaces_owner_user_id', 'workspaces', ['owner_user_id'], unique=False)
    op.create_foreign_key(
        'fk_workspaces_owner_user_id', 'workspaces', 'users',
        ['owner_user_id'], ['user_id'], ondelete='CASCADE',
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('fk_workspaces_owner_user_id', 'workspaces', type_='foreignkey')
    op.drop_index('ix_workspaces_owner_user_id', table_name='workspaces')
    op.drop_column('workspaces', 'owner_user_id')

    op.drop_index('ix_user_sessions_expires_at', table_name='user_sessions')
    op.drop_index('ix_user_sessions_user_id', table_name='user_sessions')
    op.drop_table('user_sessions')

    op.drop_table('users')
