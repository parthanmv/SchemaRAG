"""Course ORM model."""

from sqlalchemy import CheckConstraint, ForeignKey, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Course(Base):
    """A course offered by a department."""

    __tablename__ = "courses"
    __table_args__ = (
        CheckConstraint("credits BETWEEN 1 AND 6", name="ck_courses_credits"),
    )

    course_id: Mapped[int] = mapped_column(primary_key=True)
    course_code: Mapped[str] = mapped_column(String(12), nullable=False, unique=True)
    course_name: Mapped[str] = mapped_column(Text, nullable=False)
    credits: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.department_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Course id={self.course_id} code={self.course_code!r}>"
