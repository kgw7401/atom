"""initial schema

Revision ID: f74ee7deae92
Revises:
Create Date: 2026-03-07 17:59:22.626589

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = 'f74ee7deae92'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('session_templates',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('name', sa.String(length=100), nullable=False),
    sa.Column('level', sa.String(length=20), nullable=False),
    sa.Column('topic', sa.String(length=200), nullable=False),
    sa.Column('segments_json', sqlite.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('audio_chunks',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('text', sa.String(length=100), nullable=False),
    sa.Column('variant', sa.Integer(), nullable=False),
    sa.Column('audio_path', sa.String(length=500), nullable=False),
    sa.Column('duration_ms', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('text', 'variant', name='uq_audio_chunk_text_variant')
    )
    op.create_table('drill_plans',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('template_id', sa.String(length=36), nullable=True),
    sa.Column('llm_model', sa.String(length=100), nullable=False),
    sa.Column('session_config_json', sqlite.JSON(), nullable=False),
    sa.Column('plan_json', sqlite.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('session_logs',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('drill_plan_id', sa.String(length=36), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('total_duration_sec', sa.Float(), nullable=False),
    sa.Column('rounds_completed', sa.Integer(), nullable=False),
    sa.Column('rounds_total', sa.Integer(), nullable=False),
    sa.Column('segments_delivered', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('user_profiles',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('experience_level', sa.String(length=20), nullable=False),
    sa.Column('goal', sa.Text(), nullable=False),
    sa.Column('total_sessions', sa.Integer(), nullable=False),
    sa.Column('total_training_minutes', sa.Float(), nullable=False),
    sa.Column('last_session_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('combo_exposure_json', sqlite.JSON(), nullable=False),
    sa.Column('template_preference_json', sqlite.JSON(), nullable=False),
    sa.Column('session_frequency', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('user_profiles')
    op.drop_table('session_logs')
    op.drop_table('drill_plans')
    op.drop_table('audio_chunks')
    op.drop_table('session_templates')
