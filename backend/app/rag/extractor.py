"""Metadata extraction service.

Discovers the *actual* PostgreSQL schema via SQLAlchemy's reflection
(``sqlalchemy.inspect``) — nothing about tables, columns, keys, or
constraints is hardcoded here. PostgreSQL is the source of truth.
"""

from __future__ import annotations

from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector

from app.db.session import engine as default_engine
from app.rag.models import (
    CheckConstraintInfo,
    ColumnInfo,
    ForeignKeyInfo,
    SchemaMetadata,
    TableInfo,
    UniqueConstraintInfo,
)

DEFAULT_SCHEMA = "public"


def _normalize_type(sqlalchemy_type: object) -> str:
    """Render an SQLAlchemy type as a stable lowercase PostgreSQL-ish string."""
    return str(sqlalchemy_type).strip().lower()


def _sort_key_fk(fk: ForeignKeyInfo) -> tuple:
    return (fk.columns, fk.referred_table, fk.referred_columns)


class MetadataExtractor:
    """Extracts a :class:`SchemaMetadata` snapshot from a live database."""

    def __init__(self, engine: Engine | None = None, schema: str = DEFAULT_SCHEMA) -> None:
        self._engine = engine if engine is not None else default_engine
        self._schema = schema

    def extract(self) -> SchemaMetadata:
        """Reflect the schema and return a deterministic metadata snapshot."""
        inspector: Inspector = sqlalchemy_inspect(self._engine)
        table_names = sorted(inspector.get_table_names(schema=self._schema))
        tables = tuple(self._extract_table(inspector, name) for name in table_names)
        return SchemaMetadata(schema_name=self._schema, tables=tables)

    # ------------------------------------------------------------------
    # Per-table extraction
    # ------------------------------------------------------------------
    def _extract_table(self, inspector: Inspector, table_name: str) -> TableInfo:
        pk = inspector.get_pk_constraint(table_name, schema=self._schema)
        pk_columns: tuple[str, ...] = tuple(pk.get("constrained_columns") or ())
        comment = inspector.get_table_comment(table_name, schema=self._schema).get("text")

        columns = tuple(
            ColumnInfo(
                name=col["name"],
                data_type=_normalize_type(col["type"]),
                nullable=bool(col.get("nullable", True)),
                default=repr(col["default"]) if col.get("default") is not None else None,
                primary_key=col["name"] in pk_columns,
            )
            for col in inspector.get_columns(table_name, schema=self._schema)
        )

        foreign_keys = sorted(
            (self._extract_foreign_key(table_name, fk) for fk in
             inspector.get_foreign_keys(table_name, schema=self._schema)),
            key=_sort_key_fk,
        )

        unique_constraints = self._extract_unique_constraints(
            inspector, table_name, set(pk_columns)
        )
        check_constraints = sorted(
            (
                CheckConstraintInfo(
                    name=ck.get("name"),
                    expression=" ".join(ck["sqltext"].split()),
                )
                for ck in inspector.get_check_constraints(table_name, schema=self._schema)
            ),
            key=lambda c: (c.name or "", c.expression),
        )

        return TableInfo(
            name=table_name,
            columns=columns,
            primary_keys=pk_columns,
            foreign_keys=tuple(foreign_keys),
            unique_constraints=tuple(unique_constraints),
            check_constraints=tuple(check_constraints),
            comment=comment,
        )

    def _extract_foreign_key(self, table_name: str, fk: dict) -> ForeignKeyInfo:
        options = fk.get("options") or {}
        return ForeignKeyInfo(
            constraint_name=fk.get("name"),
            columns=tuple(fk["constrained_columns"]),
            referred_table=fk["referred_table"],
            referred_columns=tuple(fk["referred_columns"]),
            on_delete=options.get("ondelete"),
        )

    def _extract_unique_constraints(
        self, inspector: Inspector, table_name: str, pk_columns: set[str]
    ) -> list[UniqueConstraintInfo]:
        """Merge UNIQUE constraints and standalone unique indexes into one list.

        PostgreSQL exposes single-column ``UNIQUE`` declared via ORM
        ``unique=True`` as unique *indexes* backed by constraints; composite
        ``UniqueConstraint(...)`` declarations appear as real constraints.
        Both are uniqueness facts worth documenting; PK-implied uniqueness is
        skipped because primary keys are documented separately.
        """
        entries: list[UniqueConstraintInfo] = []
        constraint_index_names: set[str] = set()

        for uc in inspector.get_unique_constraints(table_name, schema=self._schema):
            cols = tuple(uc["column_names"])
            if set(cols) == pk_columns:
                continue
            entries.append(
                UniqueConstraintInfo(name=uc.get("name"), columns=cols)
            )
            if uc.get("name"):
                constraint_index_names.add(uc["name"])

        for index in inspector.get_indexes(table_name, schema=self._schema):
            if not index.get("unique"):
                continue
            if index.get("duplicates_constraint"):
                continue  # already captured above as a constraint
            cols = tuple(index["column_names"])
            if set(cols) == pk_columns:
                continue
            entries.append(
                UniqueConstraintInfo(
                    name=index.get("name"), columns=cols, enforced_as_index=True
                )
            )

        entries.sort(key=lambda u: (u.name or "", u.columns))
        return entries


def extract_schema_metadata(
    engine: Engine | None = None, schema: str = DEFAULT_SCHEMA
) -> SchemaMetadata:
    """Convenience function: build an extractor and return the snapshot."""
    return MetadataExtractor(engine=engine, schema=schema).extract()
