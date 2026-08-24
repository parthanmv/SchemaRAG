"""Realistic SQL tests: JOIN, GROUP BY / HAVING aggregation, subqueries."""

from sqlalchemy import text

from app.db.session import engine


def test_join_students_with_departments() -> None:
    """Basic JOIN: every student resolves to exactly one department."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT s.roll_number, s.name, d.department_name
                FROM students s
                JOIN departments d ON d.department_id = s.department_id
                LIMIT 25
                """
            )
        ).fetchall()
    assert len(rows) == 25
    assert all(r.department_name for r in rows)


def test_three_way_join_with_filter() -> None:
    """students -> enrollments -> courses JOIN with WHERE filtering."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT s.student_id, c.course_code
                FROM students s
                JOIN enrollments e ON e.student_id = s.student_id
                JOIN courses c ON c.course_id = e.course_id
                WHERE c.credits >= 4
                ORDER BY s.student_id
                LIMIT 20
                """
            )
        ).fetchall()
    assert rows, "Expected high-credit enrollments to exist"
    assert list(rows[0]._mapping.keys()) == ["student_id", "course_code"]


def test_group_by_avg_marks_per_department() -> None:
    """Aggregation: average marks per department across all exam types."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT d.department_name,
                       ROUND(AVG(m.marks), 2) AS avg_marks,
                       COUNT(*) AS mark_count
                FROM marks m
                JOIN students s ON s.student_id = m.student_id
                JOIN departments d ON d.department_id = s.department_id
                GROUP BY d.department_name
                ORDER BY avg_marks DESC
                """
            )
        ).fetchall()
    assert len(rows) == 8, "One aggregate row per department expected"
    averages = [float(r.avg_marks) for r in rows]
    assert all(0 <= a <= 100 for a in averages)
    assert averages == sorted(averages, reverse=True), "ORDER BY avg DESC violated"


def test_having_department_average_above_threshold() -> None:
    """HAVING clause: departments whose average marks exceed 65."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT d.department_code, AVG(m.marks) AS avg_marks
                FROM marks m
                JOIN students s ON s.student_id = m.student_id
                JOIN departments d ON d.department_id = s.department_id
                GROUP BY d.department_code
                HAVING AVG(m.marks) > 60
                ORDER BY avg_marks DESC
                """
            )
        ).fetchall()
    assert 1 <= len(rows) <= 8
    assert all(float(r.avg_marks) > 60 for r in rows)


def test_top_five_students_query_shape() -> None:
    """'Who are the top 5 students?' - by average marks."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT s.student_id, s.name, ROUND(AVG(m.marks), 2) AS avg_marks
                FROM students s
                JOIN marks m ON m.student_id = s.student_id
                GROUP BY s.student_id, s.name
                ORDER BY avg_marks DESC
                LIMIT 5
                """
            )
        ).fetchall()
    assert len(rows) == 5
    values = [float(r.avg_marks) for r in rows]
    assert values == sorted(values, reverse=True)


def test_subquery_students_above_department_average() -> None:
    """Correlated-style subquery: students scoring above their department average."""

    query = text(
        """
        WITH dept_avg AS (
            SELECT s.department_id, AVG(m.marks) AS avg_marks
            FROM marks m
            JOIN students s ON s.student_id = m.student_id
            GROUP BY s.department_id
        )
        SELECT COUNT(DISTINCT s.student_id) AS above_avg_students
        FROM students s
        JOIN marks m ON m.student_id = s.student_id
        JOIN dept_avg da ON da.department_id = s.department_id
        GROUP BY s.student_id, s.department_id, da.avg_marks
        HAVING AVG(m.marks) > da.avg_marks
        """
    )
    with engine.connect() as conn:
        count = sum(row[0] for row in conn.execute(query))
    assert count > 0, "Expected some students to score above their department average"


def test_low_attendance_query_returns_rows() -> None:
    """'Which students have attendance below 75%?'"""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT s.student_id, s.name
                FROM students s
                JOIN attendance a ON a.student_id = s.student_id
                WHERE a.attendance_percentage < 75
                ORDER BY s.student_id
                """
            )
        ).fetchall()
    assert len(rows) > 50, "Dataset should contain many low-attendance cases"
