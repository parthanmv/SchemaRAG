"""Attendance ORM model."""

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Attendance(Base):
    """Aggregate attendance for a student in a course for a term."""

    __tablename__ = "attendance"
    __table_args__ = (
        UniqueConstraint(
            "student_id",
            "course_id",
            "academic_year",
            "semester",
            name="uq_attendance_student_course_term",
        ),
        CheckConstraint(
            "classes_held >= 0 AND classes_attended >= 0 "
            "AND classes_attended <= classes_held",
            name="ck_attendance_counts",
        ),
        CheckConstraint(
            "attendance_percentage >= 0 AND attendance_percentage <= 100",
            name="ck_attendance_percentage",
        ),
    )

    attendance_id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.student_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    course_id: Mapped[int] = mapped_column(
        ForeignKey("courses.course_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    classes_held: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    classes_attended: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    attendance_percentage: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    academic_year: Mapped[int] = mapped_column(Integer, nullable=False)
    semester: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Attendance id={self.attendance_id} student={self.student_id} course={self.course_id}>"
