from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, ForeignKey
from sqlalchemy.sql import func
from app.database import Base

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_id = Column(String, nullable=False)
    input_data = Column(JSON, nullable=False)
    predictions = Column(JSON, nullable=False)
    confidence_intervals = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
