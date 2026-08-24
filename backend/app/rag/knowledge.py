"""Knowledge generation service.

Converts an extracted :class:`SchemaMetadata` snapshot into RAG-ready
:class:`KnowledgeDocument` objects of five types:

* ``schema``         – one document per table (grounded in PostgreSQL)
* ``relationship``   – one document per discovered foreign key
* ``constraint``     – one document per CHECK / UNIQUE constraint
* ``business_rule``  – curated domain rules (explicitly *not* DB constraints)
* ``query_example``  – curated natural-language question examples

All ordering is deterministic so repeated runs produce identical output.
"""

from __future__ import annotations

import re

from app.rag import domain_glossary
from app.rag.business_rules import BUSINESS_RULES
from app.rag.models import (
    DocumentSource,
    DocumentType,
    ForeignKeyInfo,
    KnowledgeDocument,
    SchemaMetadata,
    TableInfo,
)
from app.rag.query_examples import QUERY_EXAMPLES

# Matches PostgreSQL casts such as ``::numeric`` or ``::varchar(20)``.
# Deliberately excludes whitespace so multi-word expressions survive intact.
_CAST_RE = re.compile(r"::[a-z_][a-z0-9_]*(?:\([^)]*\))?", re.IGNORECASE)


def _readable_expression(expression: str) -> str:
    """Lightly normalise a CHECK expression without changing its meaning."""
    cleaned = _CAST_RE.sub("", expression)
    return " ".join(cleaned.split())


def _column_flags(table: TableInfo) -> dict[str, list[str]]:
    """Per-column descriptive flags derived purely from extracted metadata."""
    flags: dict[str, list[str]] = {col.name: [] for col in table.columns}
    unique_columns = {
        uc.columns[0] for uc in table.unique_constraints if len(uc.columns) == 1
    }
    fk_by_column: dict[str, ForeignKeyInfo] = {}
    for fk in table.foreign_keys:
        for col in fk.columns:
            fk_by_column[col] = fk

    for col in table.columns:
        out = flags[col.name]
        if col.primary_key:
            out.append("primary key")
        if not col.nullable and not col.primary_key:
            out.append("not null")
        if col.name in unique_columns and not col.primary_key:
            out.append("unique")
        fk = fk_by_column.get(col.name)
        if fk is not None:
            target = ", ".join(
                f"{fk.referred_table}.{c}" for c in fk.referred_columns
            )
            out.append(f"references {target}")
    return flags


class KnowledgeGenerator:
    """Builds deterministic knowledge documents from extracted metadata."""

    def generate(self, metadata: SchemaMetadata) -> tuple[KnowledgeDocument, ...]:
        """Return every document, sorted by ``document_id``."""
        documents: list[KnowledgeDocument] = []
        for table in metadata.tables:
            documents.extend(self._documents_for_table(table))
        documents.extend(self._business_rule_documents(metadata))
        documents.extend(self._query_example_documents())
        return tuple(sorted(documents, key=lambda d: d.document_id))

    # ------------------------------------------------------------------
    # Schema / relationship / constraint documents (per table)
    # ------------------------------------------------------------------
    def _documents_for_table(self, table: TableInfo) -> list[KnowledgeDocument]:
        return [
            self._schema_document(table),
            *(self._relationship_document(table, fk) for fk in table.foreign_keys),
            *(
                self._check_constraint_document(table, ck)
                for ck in table.check_constraints
            ),
            *(
                self._unique_constraint_document(table, uc)
                for uc in table.unique_constraints
            ),
        ]

    def _schema_document(self, table: TableInfo) -> KnowledgeDocument:
        intro = domain_glossary.TABLE_DESCRIPTIONS.get(
            table.name, f"The {table.name} table is defined in the college database."
        )
        flags = _column_flags(table)

        lines = [f"{intro}", "", "Columns:"]
        for col in table.columns:
            parts = [f"{col.name}: {col.data_type}", *flags[col.name]]
            line = "- " + ", ".join(parts)
            description = domain_glossary.COLUMN_DESCRIPTIONS.get(
                f"{table.name}.{col.name}"
            )
            if description:
                line += f" ({description})"
            lines.append(line)

        if not table.primary_keys:
            lines.append("")
            lines.append("This table has no declared primary key.")
        if table.comment:
            lines.append("")
            lines.append(f"Database comment: {table.comment}")

        return KnowledgeDocument(
            document_id=f"schema_{table.name}",
            document_type=DocumentType.SCHEMA,
            title=f"{table.name} table",
            content="\n".join(lines),
            tables=(table.name,),
            source=DocumentSource.POSTGRESQL_METADATA,
            extra={"table": table.name},
        )

    def _relationship_document(
        self, table: TableInfo, fk: ForeignKeyInfo
    ) -> KnowledgeDocument:
        src = ", ".join(f"{table.name}.{col}" for col in fk.columns)
        dst = ", ".join(f"{fk.referred_table}.{col}" for col in fk.referred_columns)
        on_delete = f" ON DELETE {fk.on_delete}" if fk.on_delete else ""
        name = fk.constraint_name or "unnamed_foreign_key"
        content = (
            f"Each row of the {table.name} table references one row of the "
            f"{fk.referred_table} table: {src} references {dst}. "
            f"This is a many-to-one relationship from {table.name} to "
            f"{fk.referred_table}, enforced by the foreign key constraint "
            f"{name}{on_delete}. Queries joining these tables can connect "
            f"{table.name}.{fk.columns[0]} with {fk.referred_table}."
            f"{fk.referred_columns[0]}."
        )
        return KnowledgeDocument(
            document_id=f"relationship_{name}",
            document_type=DocumentType.RELATIONSHIP,
            title=f"Relationship: {src} -> {dst}",
            content=content,
            tables=(table.name, fk.referred_table),
            source=DocumentSource.POSTGRESQL_METADATA,
            extra={
                "from_table": table.name,
                "to_table": fk.referred_table,
                "constraint_name": name,
                "on_delete": fk.on_delete or "",
                "relationship_key": fk.relationship_key(table.name),
            },
        )

    def _check_constraint_document(self, table: TableInfo, ck) -> KnowledgeDocument:
        name = ck.name or "unnamed_check"
        readable = _readable_expression(ck.expression)
        content = (
            f"The PostgreSQL CHECK constraint {name} on the {table.name} table "
            f"enforces: {readable}. Any row of {table.name} violating this "
            f"expression is rejected by the database."
        )
        return KnowledgeDocument(
            document_id=f"constraint_check_{name}",
            document_type=DocumentType.CONSTRAINT,
            title=f"Check constraint {name} on {table.name}",
            content=content,
            tables=(table.name,),
            source=DocumentSource.POSTGRESQL_METADATA,
            extra={
                "table": table.name,
                "constraint_name": name,
                "constraint_kind": "check",
                "expression": ck.expression,
            },
        )

    def _unique_constraint_document(self, table: TableInfo, uc) -> KnowledgeDocument:
        name = uc.name or "unnamed_unique"
        cols = ", ".join(uc.columns)
        content = (
            f"The combination of {cols} must be unique across rows of the "
            f"{table.name} table (enforced as a unique {uc.kind} "
            f"{name} by PostgreSQL)."
        )
        return KnowledgeDocument(
            document_id=f"constraint_unique_{name}",
            document_type=DocumentType.CONSTRAINT,
            title=f"Unique {uc.kind} {name} on {table.name}",
            content=content,
            tables=(table.name,),
            source=DocumentSource.POSTGRESQL_METADATA,
            extra={
                "table": table.name,
                "constraint_name": name,
                "constraint_kind": "unique",
                "columns": cols,
            },
        )

    # ------------------------------------------------------------------
    # Business rules / query examples (curated sources)
    # ------------------------------------------------------------------
    def _business_rule_documents(
        self, metadata: SchemaMetadata
    ) -> list[KnowledgeDocument]:
        known_checks = {
            ck.name
            for table in metadata.tables
            for ck in table.check_constraints
        }
        documents: list[KnowledgeDocument] = []
        for rule in BUSINESS_RULES:
            enforced_line = (
                "This rule IS enforced by PostgreSQL via CHECK constraint "
                f"{rule.backed_by_check}."
                if rule.backed_by_check and rule.backed_by_check in known_checks
                else "This rule is NOT enforced by any PostgreSQL constraint; "
                "it is interpretation guidance only."
            )
            content = (
                f"Business rule (domain source): {rule.statement} "
                f"{enforced_line}"
            )
            documents.append(
                KnowledgeDocument(
                    document_id=f"business_rule_{rule.rule_id}",
                    document_type=DocumentType.BUSINESS_RULE,
                    title=f"Business rule: {rule.rule_id}",
                    content=content,
                    tables=tuple(sorted(rule.related_tables)),
                    source=DocumentSource.DOMAIN_RULES,
                    extra={
                        "rule_id": rule.rule_id,
                        "statement": rule.statement,
                        "enforced_by_postgresql": (
                            "yes" if rule.backed_by_check in known_checks else "no"
                        ),
                        "backing_check_constraint": rule.backed_by_check or "",
                    },
                )
            )
        return documents

    def _query_example_documents(self) -> list[KnowledgeDocument]:
        documents: list[KnowledgeDocument] = []
        for example in QUERY_EXAMPLES:
            content = (
                f"Example question: {example.question}\n"
                f"Relevant tables: {', '.join(example.relevant_tables)}\n"
                f"SQL concepts: {', '.join(example.concepts)}"
            )
            documents.append(
                KnowledgeDocument(
                    document_id=f"query_example_{example.example_id}",
                    document_type=DocumentType.QUERY_EXAMPLE,
                    title=f"Query example: {example.question}",
                    content=content,
                    tables=tuple(sorted(example.relevant_tables)),
                    source=DocumentSource.CURATED_EXAMPLES,
                    extra={
                        "example_id": example.example_id,
                        "question": example.question,
                        "concepts": ", ".join(example.concepts),
                    },
                )
            )
        return documents
