from collections.abc import AsyncGenerator
import uuid
from fastapi import Depends
from sqlalchemy import create_engine,String,Column
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine,async_sessionmaker
from sqlalchemy.orm import relationship,DeclarativeBase
from app.core.config import settings
from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase

class Base(DeclarativeBase):
    pass

class User(SQLAlchemyBaseUserTableUUID, Base):
    __tablename__ = "users"

engine = create_async_engine(settings.DATABASE_URL)
async_session_maker = async_sessionmaker(engine,expire_on_commit=False)

async def create_db_and_table():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_async_session() -> AsyncGenerator[AsyncSession,None]:
    async with async_session_maker() as session:
        yield session

async def get_user_db(session : AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session,User)




