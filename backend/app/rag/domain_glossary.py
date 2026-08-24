"""Curated domain documentation (prose only).

This module provides *descriptions* that make generated schema documents
readable. It deliberately contains no structural information: table names,
column names, types, keys, and constraints always come from PostgreSQL
reflection. Every key below is cross-checked against the extracted schema by
``app.rag.validation``; stale entries fail validation loudly instead of being
silently ignored.
"""

from __future__ import annotations

# Table-level prose. Keys must match a real table name in PostgreSQL.
TABLE_DESCRIPTIONS: dict[str, str] = {
    "departments": "Stores the academic departments of the college.",
    "students": "Stores student information, including the department each student belongs to.",
    "courses": "Stores the courses offered by departments and their credit weights.",
    "enrollments": "Records which students are enrolled in which courses for an academic year and semester.",
    "marks": "Stores the marks a student obtained for one exam of a course in a given term.",
    "attendance": "Stores aggregated class attendance for a student in a course during a term.",
}

# Column-level prose. Keys must be ``table.column`` pairs that exist in PostgreSQL.
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "students.roll_number": "Unique roll number identifying the student within their admission cohort.",
    "students.email": "Unique institutional email address of the student.",
    "courses.course_code": "Short unique code for the course (for example CSE101).",
    "marks.exam_type": "Kind of exam, such as quiz, midterm, or final.",
    "attendance.attendance_percentage": "Percentage of held classes the student attended.",
}
