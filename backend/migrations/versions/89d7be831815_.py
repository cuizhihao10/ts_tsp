"""empty message

Revision ID: 89d7be831815
Revises: adc85d759271
Create Date: 2024-01-27 14:58:20.719708

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '89d7be831815'
down_revision = 'adc85d759271'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('classificationfile', schema=None) as batch_op:
        batch_op.alter_column('classification_result',
               existing_type=mysql.VARCHAR(length=500),
               type_=sa.String(length=1000),
               existing_nullable=False)

    with op.batch_alter_table('segmentationfile', schema=None) as batch_op:
        batch_op.add_column(sa.Column('segmentation_result', sa.String(length=1000), nullable=False))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('segmentationfile', schema=None) as batch_op:
        batch_op.drop_column('segmentation_result')

    with op.batch_alter_table('classificationfile', schema=None) as batch_op:
        batch_op.alter_column('classification_result',
               existing_type=sa.String(length=1000),
               type_=mysql.VARCHAR(length=500),
               existing_nullable=False)

    # ### end Alembic commands ###
