import time
import uuid
from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, Integer, JSON, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, synonym


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


JSON_DATA = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


class User(Base):
    __tablename__ = 'user_account'
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(Text, unique=True)
    password: Mapped[str] = mapped_column(Text)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    disabled: Mapped[int] = mapped_column(Integer, default=0)
    time_created: Mapped[int] = mapped_column(BigInteger, default=now_ms)


class LoginSession(Base):
    __tablename__ = 'login_session'
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, ForeignKey('user_account.id', ondelete='CASCADE'), index=True)
    csrf: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[int] = mapped_column(BigInteger)


class Session(Base):
    __tablename__ = 'session'
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, ForeignKey('user_account.id', ondelete='CASCADE'), unique=True)
    assistant_id: Mapped[str] = mapped_column(Text, default=new_id)
    revision: Mapped[int] = mapped_column(BigInteger, default=0)
    user_seq: Mapped[int] = mapped_column(BigInteger, default=0)
    next_message_sequence: Mapped[int] = mapped_column(BigInteger, default=0)
    next_part_sequence: Mapped[int] = mapped_column(BigInteger, default=0)
    next_display_order: Mapped[int] = mapped_column(BigInteger, default=0)
    reset_state: Mapped[dict | None] = mapped_column(JSON_DATA)
    last_reset: Mapped[dict | None] = mapped_column(JSON_DATA)
    runtime_data: Mapped[dict] = mapped_column(JSON_DATA, default=dict)
    time_created: Mapped[int] = mapped_column(BigInteger, default=now_ms)
    time_updated: Mapped[int] = mapped_column(BigInteger, default=now_ms)


class Message(Base):
    __tablename__ = 'message'
    __table_args__ = (
        UniqueConstraint('session_id', 'sequence'),
        UniqueConstraint('session_id', 'display_order'),)
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    session_id: Mapped[str] = mapped_column(Text, ForeignKey('session.id', ondelete='CASCADE'))
    sequence: Mapped[int] = mapped_column(BigInteger)
    display_order: Mapped[int] = mapped_column(BigInteger)
    role: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    attachments: Mapped[list] = mapped_column(JSON_DATA, default=list)
    time_created: Mapped[int] = mapped_column(BigInteger, default=now_ms)
    time_updated: Mapped[int] = mapped_column(BigInteger, default=now_ms)


class Part(Base):
    __tablename__ = 'part'
    __table_args__ = (
        UniqueConstraint('session_id', 'sequence'),
        Index('part_session_type_sequence_idx', 'session_id', 'type', 'sequence'),
        Index('part_session_type_context_idx', 'session_id', 'type', 'context_id', 'sequence'),)
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    id = synonym('key')
    session_id: Mapped[str] = mapped_column(Text, ForeignKey('session.id', ondelete='CASCADE'))
    sequence: Mapped[int] = mapped_column(BigInteger)
    type: Mapped[str] = mapped_column(Text)
    context_id: Mapped[str | None] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(JSON_DATA)
    time_created: Mapped[int] = mapped_column(BigInteger, default=now_ms)
    time_updated: Mapped[int] = mapped_column(BigInteger, default=now_ms)


class SharedState(Base):
    __tablename__ = 'shared_state'
    __table_args__ = (Index('shared_state_task_idx', 'session_id', 'task_id'),)
    session_id: Mapped[str] = mapped_column(Text, ForeignKey('session.id', ondelete='CASCADE'), primary_key=True)
    context_id: Mapped[str] = mapped_column(Text, primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    time_updated: Mapped[int] = mapped_column(BigInteger, default=now_ms)


class AgentMessage(Base):
    __tablename__ = 'agent_message'
    __table_args__ = (
        UniqueConstraint('session_id', 'sequence'),
        Index('agent_message_inbox_idx', 'session_id', 'context_id', 'delivery', 'sequence'),
        Index('agent_message_task_idx', 'session_id', 'task_id', 'kind', 'sequence'),)
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    session_id: Mapped[str] = mapped_column(Text, ForeignKey('session.id', ondelete='CASCADE'))
    sequence: Mapped[int] = mapped_column(BigInteger)
    context_id: Mapped[str | None] = mapped_column(Text)
    task_id: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str | None] = mapped_column(Text)
    delivery: Mapped[str] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(JSON_DATA)
    time_created: Mapped[int] = mapped_column(BigInteger, default=now_ms)
    time_updated: Mapped[int] = mapped_column(BigInteger, default=now_ms)


class LocalSetting(Base):
    __tablename__ = 'local_setting'
    __table_args__ = (Index('local_setting_scope_idx', 'scope', 'scope_id'),
                      Index('local_setting_namespace_key_idx', 'namespace', 'key'))
    scope: Mapped[str] = mapped_column(Text, primary_key=True)
    scope_id: Mapped[str] = mapped_column(Text, primary_key=True)
    namespace: Mapped[str] = mapped_column(Text, primary_key=True)
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    time_created: Mapped[int] = mapped_column(BigInteger, default=now_ms)
    time_updated: Mapped[int] = mapped_column(BigInteger, default=now_ms)


class TurnUsage(Base):
    __tablename__ = 'turn_usage'
    __table_args__ = (CheckConstraint("status in ('running', 'completed', 'error', 'cancelled')"),
                      Index('turn_usage_started_idx', 'started_at'))
    session_id: Mapped[str] = mapped_column(Text, ForeignKey('session.id', ondelete='CASCADE'), primary_key=True)
    turn_id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_message_id: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    started_at: Mapped[int] = mapped_column(BigInteger)
    completed_at: Mapped[int | None] = mapped_column(BigInteger)
    duration_ms: Mapped[int | None] = mapped_column(BigInteger)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default=text('0'))
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default=text('0'))
    cache_creation_input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default=text('0'))
    cache_read_input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default=text('0'))
