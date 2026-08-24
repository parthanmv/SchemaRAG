"""Typed models for extracted schema metadata and generated knowledge documents.

These Pydantic models are the single in-memory contract shared by the
metadata extractor, the knowledge generator, the validator, and Phase 3
(embeddings/FAISS) consumers.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    """Knowledge document categories produced for RAG ingestion."""

    SCHEMA = "schema"
    RELATIONSHIP = "relationship"
    CONSTRAINT = "constraint"
    BUSINESS_RULE = "business_rule"
    QUERY_EXAMPLE = "query_example"


class DocumentSource(str, Enum):
    """Provenance of a knowledge document's information."""

    POSTGRESQL_METADATA = "postgresql_metadata"
    DOMAIN_RULES = "domain_rules"
    CURATED_EXAMPLES = "curated_examples"


class ColumnInfo(BaseModel):
    """A single column of a table, as reported by PostgreSQL."""

    model_config = ConfigDict(frozen=True)

    name: str
    data_type: str
    nullable: bool
    default: str | None = None
    primary_key: bool = False


class ForeignKeyInfo(BaseModel):
    """A discovered foreign-key relationship between two tables."""

    model_config = ConfigDict(frozen=True)

    constraint_name: str | None
    columns: tuple[str, ...]
    referred_table: str
    referred_columns: tuple[str, ...]
    on_delete: str | None = None

    def relationship_key(self, table_name: str) -> str:
        """Stable identifier like ``students.department_id -> departments.department_id``."""
        src = ", ".join(f"{table_name}.{col}" for col in self.columns)
        dst = ", ".join(f"{self.referred_table}.{col}" for col in self.referred_columns)
        return f"{src} -> {dst}"


class UniqueConstraintInfo(BaseModel):
    """A uniqueness guarantee (a real UNIQUE constraint or a unique index)."""

    model_config = ConfigDict(frozen=True)

    name: str | None
    columns: tuple[str, ...]
    enforced_as_index: bool = False

    @property
    def kind(self) -> Literal["constraint", "unique index"]:
        return "unique index" if self.enforced_as_index else "constraint"


class CheckConstraintInfo(BaseModel):
    """A CHECK constraint with its raw SQL expression from PostgreSQL."""

    model_config = ConfigDict(frozen=True)

    name: str | None
    expression: str


class TableInfo(BaseModel):
    """Full extracted definition of one database table."""

    model_config = ConfigDict(frozen=True)

    name: str
    columns: tuple[ColumnInfo, ...]
    primary_keys: tuple[str, ...] = Field(default_factory=tuple)
    foreign_keys: tuple[ForeignKeyInfo, ...] = Field(default_factory=tuple)
    unique_constraints: tuple[UniqueConstraintInfo, ...] = Field(default_factory=tuple)
    check_constraints: tuple[CheckConstraintInfo, ...] = Field(default_factory=tuple)
    comment: str | None = None

    def column(self, name: str) -> ColumnInfo:
        """Return the named column or raise ``KeyError``."""
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"Table {self.name!r} has no column {name!r}")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(col.name for col in self.columns)


class SchemaMetadata(BaseModel):
    """Complete metadata snapshot of the inspected PostgreSQL schema."""

    model_config = ConfigDict(frozen=True)

    schema_name: str = "public"
    tables: tuple[TableInfo, ...]

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(table.name for table in self.tables)

    def get_table(self, name: str) -> TableInfo:
        """Return the named table or raise ``KeyError``."""
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(f"No table named {name!r} in extracted metadata")

    def relationships(self) -> tuple[ForeignKeyInfo, ...]:
        """All foreign keys across all tables, flattened."""
        return tuple(fk for table in self.tables for fk in table.foreign_keys)


class KnowledgeDocument(BaseModel):
    """A RAG-ready knowledge document with retrieval metadata.

    ``content`` is the text intended for embedding; the remaining fields are
    structured metadata consumed by retrieval filtering and provenance checks.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_type: DocumentType
    title: str
    content: str
    tables: tuple[str, ...] = Field(default_factory=tuple)
    source: DocumentSource
    extra: dict[str, str] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    """One problem found while validating metadata or documents."""

    severity: Literal["error", "warning"]
    message: str
