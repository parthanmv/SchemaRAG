"""Student ORM model."""

from sqlalchemy import CheckConstraint, ForeignKey, Integer, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Student(Base):
    """A student enrolled in the college."""

    __tablename__ = "students"
    __table_args__ = (
        CheckConstraint("semester >= 1 AND semester <= 8", name="ck_students_semester"),
        CheckConstraint(
            "admission_year >= 2000 AND admission_year <= 2100",
            name="ck_students_admission_year",
        ),
    )

    student_id: Mapped[int] = mapped_column(primary_key=True)
    roll_number: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.department_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    semester: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    admission_year: Mapped[int] = mapped_column(Integer, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Student id={self.student_id} roll={self.roll_number!r}>"
