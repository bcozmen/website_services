import os

from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./spatial_data.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ClickedPoint(Base):
    __tablename__ = "clicked_points"

    id = Column(Integer, primary_key=True, index=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)

# Create tables
Base.metadata.create_all(bind=engine)