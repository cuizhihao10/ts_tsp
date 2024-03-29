"""empty message

Revision ID: 0ef0d14abb29
Revises: 82d8050e8337
Create Date: 2023-11-08 16:53:28.438397

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '0ef0d14abb29'
down_revision = '82d8050e8337'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('classificationfile', schema=None) as batch_op:
        batch_op.add_column(sa.Column('author_id', sa.Integer(), nullable=True))
        batch_op.drop_constraint('classificationfile_ibfk_1', type_='foreignkey')
        batch_op.create_foreign_key(None, 'user', ['author_id'], ['id'])
        batch_op.drop_column('uploadfile_id')

    with op.batch_alter_table('segmentationfile', schema=None) as batch_op:
        batch_op.add_column(sa.Column('original_images_sequence', sa.String(length=100), nullable=False))
        batch_op.add_column(sa.Column('original_images_path', sa.String(length=100), nullable=False))
        batch_op.add_column(sa.Column('author_id', sa.Integer(), nullable=True))
        batch_op.drop_constraint('segmentationfile_ibfk_1', type_='foreignkey')
        batch_op.create_foreign_key(None, 'user', ['author_id'], ['id'])
        batch_op.drop_column('uploadfile_id')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('segmentationfile', schema=None) as batch_op:
        batch_op.add_column(sa.Column('uploadfile_id', mysql.INTEGER(), autoincrement=False, nullable=True))
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.create_foreign_key('segmentationfile_ibfk_1', 'uploadfile', ['uploadfile_id'], ['id'])
        batch_op.drop_column('author_id')
        batch_op.drop_column('original_images_path')
        batch_op.drop_column('original_images_sequence')

    with op.batch_alter_table('classificationfile', schema=None) as batch_op:
        batch_op.add_column(sa.Column('uploadfile_id', mysql.INTEGER(), autoincrement=False, nullable=True))
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.create_foreign_key('classificationfile_ibfk_1', 'uploadfile', ['uploadfile_id'], ['id'])
        batch_op.drop_column('author_id')

    # ### end Alembic commands ###
