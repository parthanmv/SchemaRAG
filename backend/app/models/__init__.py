"""ORM models package. Importing this module registers every table with Base.metadata."""

from app.models.attendance import Attendance
from app.models.course import Course
from app.models.department import Department
from app.models.enrollment import Enrollment
from app.models.mark import Mark
from app.models.student import Student

__all__ = ["Attendance", "Course", "Department", "Enrollment", "Mark", "Student"]
