"""One-time helper: create the dedicated read-only execution role.

Creates ``schemarag_reader`` (LOGIN, no superuser/createrole/createdb) and
grants exactly:
    * CONNECT on the database
    * USAGE on schema ``public``
    * SELECT on the six project tables

and nothing else. The generated password is written straight into the
repo-root .env (EXEC_DB_USER / EXEC_DB_PASSWORD, replaced in place so exactly
one entry of each exists) and never printed.

Run from backend/ with admin credentials (the default ``postgres`` role):

    set ADMIN_DB_USER=postgres            (Windows cmd)
    set ADMIN_DB_PASSWORD=...
    python -m app.scripts.setup_readonly_role

If ADMIN_DB_USER/PASSWORD are not provided you will be prompted via getpass.

Security notes:
* Role/password DDL composes client-side via psycopg.sql (Identifier +
  Literal) - PostgreSQL DDL rejects server-side bind placeholders for the
  PASSWORD clause, and raw concatenation would be unsafe.
* Every connection is opened with individual keyword arguments (never an
  interpolated DSN string): libpq keyword/value connstrings do NOT
  percent-decode, so URL-quoting passwords there silently corrupts any
  password containing e.g. '@' or '#'.
"""

from __future__ import annotations

import getpass
import secrets
import string
import sys
from pathlib import Path

import psycopg
from psycopg import sql

from app.core.config import get_settings

PROJECT_TABLES = (
    "students",
    "departments",
    "courses",
    "enrollments",
    "marks",
    "attendance",
)
ROLE_NAME = "schemarag_reader"
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


def _generate_password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*_-"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(length))
        if (
            any(c.islower() for c in pw)
            and any(c.isupper() for c in pw)
            and any(c.isdigit() for c in pw)
        ):
            return pw


def _create_role_stmt(role: str, password: str):
    """CREATE ROLE with the password as a safely escaped SQL literal."""
    return sql.SQL(
        "CREATE ROLE {role} LOGIN PASSWORD {pw} NOSUPERUSER NOCREATEDB NOCREATEROLE"
    ).format(role=sql.Identifier(role), pw=sql.Literal(password))


def _alter_role_password_stmt(role: str, password: str):
    """ALTER ROLE ... PASSWORD using the same safe literal composition."""
    return sql.SQL("ALTER ROLE {role} LOGIN PASSWORD {pw}").format(
        role=sql.Identifier(role), pw=sql.Literal(password)
    )


def _connect_kwargs(
    settings, user: str, password: str, *, read_only: bool = False
) -> dict:
    """Connection parameters for psycopg.connect(**kwargs).

    Individual parameters let the driver escape each value correctly - unlike
    a hand-built keyword/value DSN, where percent-encoded characters would be
    taken literally and change the effective password.
    """
    kwargs = dict(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=user,
        password=password,
        connect_timeout=5,
    )
    if read_only:
        kwargs["options"] = "-c default_transaction_read_only=on"
    return kwargs


def _admin_connection() -> psycopg.Connection:
    settings = get_settings()
    import os

    user = os.environ.get("ADMIN_DB_USER") or input("Admin DB user [postgres]: ").strip() or "postgres"
    password = os.environ.get("ADMIN_DB_PASSWORD") or getpass.getpass(
        f"Password for {user}@{settings.db_host}:{settings.db_port}: "
    )
    return psycopg.connect(autocommit=True, **_connect_kwargs(settings, user, password))


def _ensure_role(cur, role: str, password: str) -> None:
    """Create the role or reset its password to exactly ``password``."""
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
    if cur.fetchone():
        print(f"Role {role} already exists; resetting password and grants.")
        cur.execute(_alter_role_password_stmt(role, password))
    else:
        cur.execute(_create_role_stmt(role, password))


def _grant_readonly(cur, role: str, dbname: str) -> None:
    cur.execute(
        sql.SQL("GRANT CONNECT ON DATABASE {db} TO {role}").format(
            db=sql.Identifier(dbname), role=sql.Identifier(role)
        )
    )
    cur.execute(
        sql.SQL("GRANT USAGE ON SCHEMA public TO {role}").format(
            role=sql.Identifier(role)
        )
    )
    for table in PROJECT_TABLES:
        cur.execute(
            sql.SQL("GRANT SELECT ON public.{t} TO {role}").format(
                t=sql.Identifier(table), role=sql.Identifier(role)
            )
        )


def _write_env(user: str, password: str, env_path: Path = ENV_PATH) -> None:
    """Write EXEC_DB_USER / EXEC_DB_PASSWORD, replacing existing entries.

    Exactly one line per key survives; unrelated lines (and their order,
    comments included) are preserved byte-for-byte.
    """
    updates = {"EXEC_DB_USER": user, "EXEC_DB_PASSWORD": password}
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else None
        if key in updates:
            if key in seen:
                continue  # collapse pre-existing duplicates
            seen.add(key)
            out.append(f"{key}={updates[key]}")
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")

    env_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")


def main() -> int:
    settings = get_settings()
    password = _generate_password()

    try:
        with _admin_connection() as conn, conn.cursor() as cur:
            _ensure_role(cur, ROLE_NAME, password)
            _grant_readonly(cur, ROLE_NAME, settings.db_name)
    except psycopg.OperationalError as exc:
        print(f"Could not connect as administrator: {type(exc).__name__}")
        return 1

    _write_env(ROLE_NAME, password)

    # Verify: connect AS the new role with THE SAME generated password.
    verify_kwargs = _connect_kwargs(settings, ROLE_NAME, password, read_only=True)
    try:
        vconn = psycopg.connect(**verify_kwargs)
    except psycopg.OperationalError:
        print("Verification failed: could not authenticate schemarag_reader "
              "with the freshly set password.")
        return 1
    with vconn, vconn.cursor() as vcur:
        vcur.execute("SELECT COUNT(*) FROM students")
        count = vcur.fetchone()[0]
        for denied in ("DELETE FROM students",):
            try:
                vconn.rollback()
                vcur.execute(denied)
                print(f"WARNING: write was NOT refused ({denied}) - grants too broad!")
                return 1
            except (
                psycopg.errors.InsufficientPrivilege,
                psycopg.errors.ReadOnlySqlTransaction,
            ):
                # Either refusal proves the write path is closed: missing
                # grants (42501) or the session read-only guard (25006).
                pass
    print(f"OK: {ROLE_NAME} can SELECT (students rows: {count}) and writes are refused.")
    print(f"Credentials written to {ENV_PATH} (never printed to console output above).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
