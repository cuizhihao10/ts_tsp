"""empty message

Revision ID: 95e8d5c15127
Revises: 
Create Date: 2023-10-12 16:25:31.261844

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '95e8d5c15127'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('username', sa.String(length=100), nullable=False),
    sa.Column('password', sa.String(length=100), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('username')
    )
    op.create_table('uploadfile',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('filename', sa.String(length=100), nullable=False),
    sa.Column('type', sa.String(length=100), nullable=False),
    sa.Column('size', sa.String(length=100), nullable=False),
    sa.Column('path', sa.String(length=100), nullable=False),
    sa.Column('upload_time', sa.DateTime(), nullable=True),
    sa.Column('author_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['author_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('filename')
    )
    op.create_table('classificationfile',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('filename', sa.String(length=100), nullable=False),
    sa.Column('type', sa.String(length=100), nullable=False),
    sa.Column('size', sa.String(length=100), nullable=False),
    sa.Column('path', sa.String(length=100), nullable=False),
    sa.Column('classification_time', sa.DateTime(), nullable=True),
    sa.Column('classification_result', sa.String(length=50), nullable=False),
    sa.Column('uploadfile_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['uploadfile_id'], ['uploadfile.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('segmentationfile',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('filename', sa.String(length=100), nullable=False),
    sa.Column('type', sa.String(length=100), nullable=False),
    sa.Column('size', sa.String(length=100), nullable=False),
    sa.Column('path', sa.String(length=100), nullable=False),
    sa.Column('segmentation_time', sa.DateTime(), nullable=True),
    sa.Column('uploadfile_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['uploadfile_id'], ['uploadfile.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('segmentationfile')
    op.drop_table('classificationfile')
    op.drop_table('uploadfile')
    op.drop_table('user')
    # ### end Alembic commands ###