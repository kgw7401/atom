"""add video analysis tables

Revision ID: 2e1a34f3d1be
Revises: f74ee7deae92
Create Date: 2026-03-22 21:16:50.359749

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2e1a34f3d1be'
down_revision: Union[str, Sequence[str], None] = 'f74ee7deae92'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'video_uploads',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('session_log_id', sa.String(36), nullable=True),
        sa.Column('drill_plan_id', sa.String(36), nullable=False),
        sa.Column('video_path', sa.String(500), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='uploaded'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        'analysis_results',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('video_upload_id', sa.String(36), nullable=False),
        sa.Column('predicted_events', sa.JSON(), nullable=True),
        sa.Column('expected_events', sa.JSON(), nullable=True),
        sa.Column('comparison_json', sa.JSON(), nullable=True),
        sa.Column('feedback_text', sa.Text(), server_default=''),
        sa.Column('accuracy_score', sa.Float(), server_default='0.0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        'session_recommendations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('analysis_result_id', sa.String(36), nullable=False),
        sa.Column('recommended_level', sa.String(20), nullable=False),
        sa.Column('recommended_topic', sa.String(200), server_default=''),
        sa.Column('narrative_text', sa.Text(), server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )

    with op.batch_alter_table('user_profiles', schema=None) as batch_op:
        batch_op.add_column(sa.Column('performance_summary_json', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('session_recommendations')
    op.drop_table('analysis_results')
    op.drop_table('video_uploads')

    with op.batch_alter_table('user_profiles', schema=None) as batch_op:
        batch_op.drop_column('performance_summary_json')
