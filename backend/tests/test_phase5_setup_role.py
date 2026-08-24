"""Phase 5 tests: safe SQL composition in setup_readonly_role.

Guards the bug where ``CREATE ROLE ... PASSWORD %s`` was sent as a
server-side bind ($1), which PostgreSQL DDL rejects. The fix composes the
statement client-side via psycopg.sql (Identifier + Literal), so these tests
pin down:

* statements are psycopg.sql Composables (never %-formatted / f-string SQL)
* rendered SQL contains no bind placeholders and no raw concatenation
* passwords containing quotes/backslashes are escaped correctly
* role name, flags, grants and .env keys are preserved
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import pytest

from app.scripts.setup_readonly_role import (
    ENV_PATH,
    PROJECT_TABLES,
    ROLE_NAME,
    _alter_role_password_stmt,
    _connect_kwargs,
    _create_role_stmt,
    _ensure_role,
    _generate_password,
    _grant_readonly,
    _write_env,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "app" / "scripts" / "setup_readonly_role.py"

TRICKY_PASSWORD = "p@ss'wo\"rd\\x;--"


@pytest.fixture(scope="module")
def pg_context():
    """A live connection used ONLY to render sql composables (no execution)."""
    try:
        import psycopg

        from app.core.config import get_settings

        s = get_settings()
        conn = psycopg.connect(
            host=s.db_host, port=s.db_port, dbname=s.db_name,
            user=s.db_user, password=s.db_password, connect_timeout=4,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"PostgreSQL unavailable for rendering context: {type(exc).__name__}")
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Statement composition - the actual bug fix
# ---------------------------------------------------------------------------
def test_create_role_stmt_is_composable(pg_context):
    from psycopg import sql

    stmt = _create_role_stmt(ROLE_NAME, TRICKY_PASSWORD)
    assert isinstance(stmt, sql.Composed)
    rendered = stmt.as_string(pg_context)
    # No server-side binds may survive composition.
    assert "$1" not in rendered
    assert "%s" not in rendered


def test_create_role_stmt_escapes_password(pg_context):
    rendered = _create_role_stmt(ROLE_NAME, TRICKY_PASSWORD).as_string(pg_context)
    # psycopg3 may render as plain '...' or E'...' escape form; both double
    # internal single quotes so the literal can never terminate early.
    plain = "'p@ss''wo\"rd\\x;--'"
    escape = "E'p@ss''wo\"rd\\\\x;--'"
    assert plain in rendered or escape in rendered, rendered
    # Never spliced into the statement unquoted.
    assert re.search(rf"PASSWORD\s+{re.escape(TRICKY_PASSWORD)}(?!['])", rendered) is None


def test_create_role_stmt_preserves_role_and_flags(pg_context):
    rendered = _create_role_stmt(ROLE_NAME, "whatever").as_string(pg_context)
    assert f'"schemarag_reader"' in rendered  # Identifier is safely quoted
    for flag in ("LOGIN", "NOSUPERUSER", "NOCREATEDB", "NOCREATEROLE"):
        assert flag in rendered
    assert rendered.index("NOSUPERUSER") < rendered.index("NOCREATEDB") < rendered.index("NOCREATEROLE")


def test_alter_role_stmt_escapes_password(pg_context):
    rendered = _alter_role_password_stmt(ROLE_NAME, "it's").as_string(pg_context)
    assert rendered == 'ALTER ROLE "schemarag_reader" LOGIN PASSWORD \'it\'\'s\''
    assert "$1" not in rendered and "%s" not in rendered


def test_statements_never_use_raw_concatenation():
    """Structural guard: password must reach SQL only through sql.Literal."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "sql.Literal(password)" in source
    assert not re.search(r"PASSWORD\s+%s", source), "bind placeholder in DDL"
    # No f-string / %-formatted CREATE/ALTER ROLE statements may exist.
    assert not re.search(r"f[\"'][^\"']*(?:CREATE|ALTER)\s+ROLE", source)
    assert not re.search(r"execute\(f?['\"].*(?:CREATE|ALTER)\s+ROLE", source)


# ---------------------------------------------------------------------------
# Preserved behaviour
# ---------------------------------------------------------------------------
def test_role_name_unchanged():
    assert ROLE_NAME == "schemarag_reader"


def test_project_tables_unchanged():
    assert PROJECT_TABLES == (
        "students", "departments", "courses", "enrollments", "marks", "attendance",
    )


def test_env_keys_written_are_exactly_exec_db():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"EXEC_DB_USER"' in source and '"EXEC_DB_PASSWORD"' in source
    # No other credential-bearing keys are introduced by the writer.
    written_keys = set(re.findall(r'"([A-Z_]+)":\s', source))
    assert written_keys == {"EXEC_DB_USER", "EXEC_DB_PASSWORD"}


def test_verification_refusal_still_checked():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "DELETE FROM students" in source
    assert "InsufficientPrivilege" in source
    assert "ReadOnlySqlTransaction" in source


# ---------------------------------------------------------------------------
# Password generator (preserved behaviour)
# ---------------------------------------------------------------------------
def test_generated_password_meets_complexity():
    pw = _generate_password()
    assert len(pw) == 24
    assert any(c.islower() for c in pw)
    assert any(c.isupper() for c in pw)
    assert any(c.isdigit() for c in pw)
    allowed = set(string.ascii_letters + string.digits + "!@#$%^&*_-")
    assert set(pw) <= allowed


def test_generated_passwords_unique():
    assert len({_generate_password() for _ in range(20)}) == 20


# ---------------------------------------------------------------------------
# Regression: exact generated password flows consistently everywhere (bug #2)
# ---------------------------------------------------------------------------
class _SettingsStub:
    db_host = "localhost"
    db_port = 5432
    db_name = "college_db"


class _RecordingCursor:
    def __init__(self, exists=False):
        self.statements = []
        self._exists = exists

    def execute(self, query, params=None):
        self.statements.append((query, params))

    def fetchone(self):
        return (1,) if self._exists else None


def test_connect_kwargs_carry_the_exact_password():
    """libpq keyword/value DSNs do not percent-decode - kwargs must be used."""
    pw = "p@ss#word&x%"
    kwargs = _connect_kwargs(_SettingsStub(), ROLE_NAME, pw, read_only=True)
    assert kwargs["password"] == pw  # byte-identical, no quote_plus mangling
    assert kwargs["user"] == ROLE_NAME
    assert kwargs["dbname"] == "college_db"
    assert "default_transaction_read_only=on" in kwargs["options"]


def test_source_never_interpolates_passwords_into_dsn_strings():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "quote_plus" not in source, "URL-quoting corrupts keyword/value DSNs"
    assert not re.search(r"password=\{", source), "interpolated DSN password"
    assert not re.search(r"user=\{", source), "interpolated DSN user"
    assert "_connect_kwargs(settings, ROLE_NAME, password" in source


def test_verification_reuses_the_generated_password(pg_context):
    pw = _generate_password()
    create = _create_role_stmt(ROLE_NAME, pw).as_string(pg_context)
    alter = _alter_role_password_stmt(ROLE_NAME, pw).as_string(pg_context)
    kwargs = _connect_kwargs(_SettingsStub(), ROLE_NAME, pw)
    # The literal stored by the DDL is exactly what the connection will send.
    stored = re.search(r"(?:E)?'([^']*)'", create).group(1)
    assert stored == pw == kwargs["password"]
    assert re.search(r"(?:E)?'([^']*)'", alter).group(1) == pw


def test_ensure_role_create_branch_uses_generated_password(pg_context):
    pw = _generate_password()
    cur = _RecordingCursor(exists=False)
    _ensure_role(cur, ROLE_NAME, pw)
    assert "SELECT 1 FROM pg_roles" in str(cur.statements[0][0])
    rendered = cur.statements[1][0].as_string(pg_context)
    assert rendered.startswith(f'CREATE ROLE "{ROLE_NAME}"')
    assert f"'{pw}'" in rendered  # generator alphabet needs no escaping here
    assert "$1" not in rendered and "%s" not in rendered


def test_ensure_role_existing_role_resets_password(pg_context):
    pw = _generate_password()
    cur = _RecordingCursor(exists=True)
    _ensure_role(cur, ROLE_NAME, pw)
    rendered = cur.statements[1][0].as_string(pg_context)
    assert rendered.startswith(f'ALTER ROLE "{ROLE_NAME}" LOGIN PASSWORD')
    assert f"'{pw}'" in rendered


def test_grant_readonly_issues_exactly_the_intended_grants(pg_context):
    cur = _RecordingCursor()
    _grant_readonly(cur, ROLE_NAME, "college_db")
    rendered = [entry[0].as_string(pg_context) for entry in cur.statements]
    assert rendered == [
        'GRANT CONNECT ON DATABASE "college_db" TO "schemarag_reader"',
        'GRANT USAGE ON SCHEMA public TO "schemarag_reader"',
        *[
            f'GRANT SELECT ON public."{t}" TO "schemarag_reader"'
            for t in PROJECT_TABLES
        ],
    ]


# ---------------------------------------------------------------------------
# Regression: .env writer replaces instead of appending duplicates (bug #3)
# ---------------------------------------------------------------------------
SAMPLE_ENV = (
    "# SchemaRAG configuration\n"
    "DB_HOST=localhost\n"
    "DB_PASSWORD=db-secret\n"
    "EXEC_DB_USER=stale_user\n"
    "GEMINI_API_KEY=gemini-secret\n"
    "EXEC_DB_PASSWORD=stale_pw_one\n"
    "LLM_PROVIDER=gemini\n"
    "EXEC_DB_USER=stale_user_dup\n"
    "EXEC_DB_PASSWORD=stale_pw_two\n"
)


def test_write_env_replaces_entries_and_collapses_duplicates(tmp_path):
    env = tmp_path / ".env"
    env.write_text(SAMPLE_ENV, encoding="utf-8")

    _write_env("schemarag_reader", "FRESH_pw", env)

    text = env.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert sum(1 for ln in lines if ln.startswith("EXEC_DB_USER=")) == 1
    assert sum(1 for ln in lines if ln.startswith("EXEC_DB_PASSWORD=")) == 1
    assert "EXEC_DB_USER=schemarag_reader" in lines
    assert "EXEC_DB_PASSWORD=FRESH_pw" in lines
    # stale values gone
    assert "stale_user" not in text and "stale_pw" not in text
    # unrelated entries + comments preserved, original order kept
    assert lines[0] == "# SchemaRAG configuration"
    assert "DB_HOST=localhost" in lines and "DB_PASSWORD=db-secret" in lines
    assert "GEMINI_API_KEY=gemini-secret" in lines and "LLM_PROVIDER=gemini" in lines
    assert text.index("EXEC_DB_USER=") < text.index("GEMINI_API_KEY")


def test_write_env_appends_once_when_keys_missing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\nB=2\n", encoding="utf-8")

    _write_env("u", "p", env)

    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == ["A=1", "B=2"]
    assert lines.count("EXEC_DB_USER=u") == 1
    assert lines.count("EXEC_DB_PASSWORD=p") == 1


def test_write_env_creates_file_when_absent(tmp_path):
    env = tmp_path / ".env"
    _write_env("u", "p", env)
    content = env.read_text(encoding="utf-8")
    assert "EXEC_DB_USER=u" in content and "EXEC_DB_PASSWORD=p" in content
