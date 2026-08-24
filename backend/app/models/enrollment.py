"""Enrollment ORM model (students <-> courses join table)."""

from sqlalchemy import CheckConstraint, ForeignKey, Integer, SmallInteger, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Enrollment(Base):
    """A student's enrollment in a course for a specific academic year/semester."""

    __tablename__ = "enrollments"
    __table_args__ = (
        UniqueConstraint(
            "student_id",
            "course_id",
            "academic_year",
            "semester",
            name="uq_enrollments_student_course_term",
        ),
        CheckConstraint("semester >= 1 AND semester <= 8", name="ck_enrollments_semester"),
    )

    enrollment_id: Mapped[int] = mapped_column(primary_key=True)
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
    academic_year: Mapped[int] = mapped_column(Integer, nullable=False)
    semester: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Enrollment id={self.enrollment_id} student={self.student_id} course={self.course_id}>"
