"""Department ORM model."""

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Department(Base):
    """An academic department of the college."""

    __tablename__ = "departments"

    department_id: Mapped[int] = mapped_column(primary_key=True)
    department_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    department_code: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Department id={self.department_id} code={self.department_code!r}>"
