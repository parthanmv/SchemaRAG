"""Mark ORM model."""

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Mark(Base):
    """Marks obtained by a student in one exam of a course."""

    __tablename__ = "marks"
    __table_args__ = (
        UniqueConstraint(
            "student_id",
            "course_id",
            "exam_type",
            "academic_year",
            "semester",
            name="uq_marks_student_course_exam_term",
        ),
        CheckConstraint("marks >= 0 AND marks <= 100", name="ck_marks_range"),
        CheckConstraint("semester >= 1 AND semester <= 8", name="ck_marks_semester"),
    )

    mark_id: Mapped[int] = mapped_column(primary_key=True)
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
    exam_type: Mapped[str] = mapped_column(String(20), nullable=False)
    marks: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    academic_year: Mapped[int] = mapped_column(Integer, nullable=False)
    semester: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Mark id={self.mark_id} student={self.student_id} course={self.course_id}>"
