"""Phase 5 tests: AST-based SQL security validator (no database required).

Covers the required safe/reject matrix from the Phase 5 spec, fail-closed
behaviour, comment rejection, CTE policy, and the normalized-SQL output.
"""

import pytest

from app.rag.sql_security import SecurityReport, SQLSecurityValidator, validate_sql_security


@pytest.fixture()
def validator() -> SQLSecurityValidator:
    return SQLSecurityValidator()


# ---------------------------------------------------------------------------
# SAFE statements - must be allowed
# ---------------------------------------------------------------------------
SAFE_CASES = [
    # 1
    ("SELECT * FROM students", "plain select"),
    # 2
    ("SELECT name FROM students WHERE semester = 5", "where filter"),
    # 3
    (
        "SELECT s.name, d.department_name FROM students s "
        "JOIN departments d ON d.department_id = s.department_id",
        "join",
    ),
    # 4
    (
        "SELECT department_id, AVG(marks) AS avg_marks FROM marks GROUP BY department_id",
        "group by + aggregate",
    ),
    # 5
    (
        "SELECT department_id, AVG(marks) FROM marks GROUP BY department_id HAVING AVG(marks) > 75",
        "having",
    ),
    # 6
    (
        "SELECT name FROM students ORDER BY admission_year DESC LIMIT 10 OFFSET 5",
        "order by / limit / offset",
    ),
    # 7
    (
        "WITH dept_avg AS (SELECT department_id, AVG(marks) AS am FROM marks "
        "GROUP BY department_id) SELECT * FROM dept_avg WHERE am > 75",
        "CTE select",
    ),
    # 8
    (
        "SELECT name FROM students WHERE student_id IN "
        "(SELECT student_id FROM marks WHERE marks > 90)",
        "subquery",
    ),
    ("SELECT DISTINCT exam_type FROM marks", "distinct"),
    ("SELECT COUNT(*) FROM enrollments", "count"),
    (
        "SELECT CASE WHEN attendance_percentage < 75 THEN 'low' ELSE 'ok' END "
        "AS status FROM attendance",
        "case expression",
    ),
    (
        "SELECT ROUND(AVG(marks), 2) FROM marks WHERE exam_type = 'final'",
        "scalar functions",
    ),
    (
        "SELECT EXTRACT(YEAR FROM CURRENT_DATE) AS yr",
        "extract / current date",
    ),
    (
        "SELECT UPPER(name) || '-' || roll_number AS label FROM students",
        "string operators",
    ),
    (
        "SELECT m.course_id, SUM(m.marks) FROM marks m "
        "LEFT JOIN students s ON s.student_id = m.student_id GROUP BY m.course_id",
        "left join",
    ),
    ("SELECT * FROM public.students", "explicit public schema qualifier"),
]


@pytest.mark.parametrize("sql", [c[0] for c in SAFE_CASES], ids=[c[1] for c in SAFE_CASES])
def test_safe_sql_is_allowed(validator: SQLSecurityValidator, sql: str):
    report = validator.validate(sql)
    assert report.allowed is True, f"safe SQL rejected: {report.issues}"
    assert report.issues == ()
    assert report.normalized_sql is not None


# ---------------------------------------------------------------------------
# REJECTED statements - must never be allowed
# ---------------------------------------------------------------------------
REJECT_CASES = [
    ("INSERT INTO students (name) VALUES ('x')", "insert"),
    ("UPDATE students SET name = 'x' WHERE student_id = 1", "update"),
    ("DELETE FROM students", "delete"),
    ("DROP TABLE students", "drop"),
    ("ALTER TABLE students ADD COLUMN hack text", "alter"),
    ("TRUNCATE TABLE students", "truncate"),
    ("CREATE TABLE evil (id int)", "create"),
    ("GRANT SELECT ON students TO PUBLIC", "grant"),
    ("REVOKE SELECT ON students FROM PUBLIC", "revoke"),
    (
        "MERGE INTO students s USING departments d ON s.department_id = d.department_id "
        "WHEN MATCHED THEN UPDATE SET name = 'x'",
        "merge",
    ),
    ("COPY students TO STDOUT", "copy"),
    ("CALL some_procedure()", "call"),
    ("DO $$ BEGIN UPDATE students SET name = 'x'; END $$;", "do block"),
    ("EXECUTE prep_stmt", "execute"),
    ("VACUUM ANALYZE students", "vacuum command"),
    ("BEGIN", "begin transaction"),
    ("COMMIT", "commit"),
    ("ROLLBACK", "rollback"),
    ("SET statement_timeout = 0", "session set"),
    ("SELECT * FROM pg_catalog.pg_tables", "pg_catalog access"),
    ("SELECT * FROM information_schema.tables", "information_schema access"),
    ("SELECT table_name FROM information_schema.columns", "information_schema columns"),
    ("SELECT * FROM pg_tables", "pg_* relation"),
    ("SELECT pg_sleep(10)", "pg_sleep"),
    ("SELECT pg_catalog.pg_sleep(10)", "schema-qualified function"),
    ("SELECT pg_read_file('/etc/passwd')", "server file read"),
    ("SELECT lo_import('/etc/passwd')", "large object import"),
    ("SELECT lo_export(12345, '/tmp/x')", "large object export"),
    ("SELECT dblink('host=x dbname=y', 'SELECT 1')", "dblink"),
    ("SELECT nextval('students_student_id_seq')", "sequence mutation"),
    ("SELECT set_config('role', 'postgres', false)", "config mutation"),
    ("SELECT * FROM sales.orders", "non-public schema"),
    ("SELECT * FROM other_schema.t", "arbitrary other schema"),
    ("SELECT * FROM mydb.public.students", "cross-database three-part name"),
]


@pytest.mark.parametrize(
    "sql", [c[0] for c in REJECT_CASES], ids=[c[1] for c in REJECT_CASES]
)
def test_dangerous_sql_is_rejected(validator: SQLSecurityValidator, sql: str):
    report = validator.validate(sql)
    assert report.allowed is False, f"dangerous SQL allowed: {sql}"
    assert report.issues
    assert report.normalized_sql is None  # nothing usable comes out of a rejection


# ---------------------------------------------------------------------------
# Structural attacks
# ---------------------------------------------------------------------------
def test_multiple_statements_rejected(validator: SQLSecurityValidator):
    report = validator.validate("SELECT 1; DELETE FROM students;")
    assert report.allowed is False


def test_malformed_sql_rejected_fail_closed(validator: SQLSecurityValidator):
    report = validator.validate("SELECT FROM WHERE nonsense )(")
    assert report.allowed is False
    assert any("does not parse" in i for i in report.issues)


def test_empty_sql_rejected(validator: SQLSecurityValidator):
    assert validator.validate("").allowed is False
    assert validator.validate("   ").allowed is False


def test_comment_line_bypass_rejected(validator: SQLSecurityValidator):
    report = validator.validate("SELECT * FROM students -- DROP TABLE students")
    assert report.allowed is False


def test_comment_block_bypass_rejected(validator: SQLSecurityValidator):
    report = validator.validate("/* hidden */ SELECT * FROM students")
    assert report.allowed is False


def test_select_into_rejected(validator: SQLSecurityValidator):
    """SELECT INTO creates a table - must not pass validation."""
    report = validator.validate("SELECT * INTO stolen_copy FROM students")
    assert report.allowed is False


def test_for_update_locking_rejected(validator: SQLSecurityValidator):
    report = validator.validate("SELECT * FROM students WHERE student_id = 1 FOR UPDATE")
    assert report.allowed is False


def test_data_modifying_cte_rejected(validator: SQLSecurityValidator):
    """PostgreSQL allows INSERT/UPDATE/DELETE inside CTEs - the walk must catch it."""
    insert_cte = (
        "WITH x AS (INSERT INTO students (name) VALUES ('a') RETURNING *) "
        "SELECT * FROM x"
    )
    delete_cte = (
        "WITH x AS (DELETE FROM marks RETURNING *) SELECT * FROM x"
    )
    update_cte = (
        "WITH x AS (UPDATE students SET name='a' RETURNING *) SELECT * FROM x"
    )
    for sql in (insert_cte, delete_cte, update_cte):
        report = validator.validate(sql)
        assert report.allowed is False, sql
        assert report.normalized_sql is None


def test_union_of_selects_allowed_but_union_with_delete_rejected(
    validator: SQLSecurityValidator,
):
    ok = validator.validate("SELECT name FROM students UNION SELECT department_name FROM departments")
    assert ok.allowed is True

    bad = validator.validate("SELECT name FROM students UNION DELETE FROM marks")
    assert bad.allowed is False


def test_validator_is_stateless_and_repeatable(validator: SQLSecurityValidator):
    first = validator.validate("DELETE FROM students")
    second = validator.validate("SELECT * FROM students")
    third = validator.validate("DELETE FROM students")
    assert first == third
    assert second.allowed is True


def test_convenience_wrapper_matches_instance():
    report: SecurityReport = validate_sql_security("SELECT 1 FROM departments")
    assert report.allowed is True
