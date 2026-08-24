"""Curated business rules for the college domain.

These rules are *interpretation guidance* for future RAG/Text-to-SQL stages.
They are intentionally kept separate from database constraints: each rule
states explicitly whether PostgreSQL enforces it. Nothing here may claim to be
a database constraint unless the corresponding CHECK constraint actually
exists in PostgreSQL (the validator cross-checks ``backed_by_check`` names).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BusinessRule(BaseModel):
    """One domain rule with its enforcement status."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    statement: str
    related_tables: tuple[str, ...] = Field(default_factory=tuple)
    # Name of the PostgreSQL CHECK constraint that enforces this rule, if any.
    backed_by_check: str | None = None


BUSINESS_RULES: tuple[BusinessRule, ...] = (
    BusinessRule(
        rule_id="marks_scale",
        statement="Marks are interpreted on a 0-100 scale.",
        related_tables=("marks",),
        backed_by_check="ck_marks_range",
    ),
    BusinessRule(
        rule_id="pass_mark",
        statement="A mark below 40 can be treated as failing.",
        related_tables=("marks",),
        backed_by_check=None,
    ),
    BusinessRule(
        rule_id="attendance_threshold",
        statement=(
            "Attendance below 75 percent can be treated as insufficient attendance."
        ),
        related_tables=("attendance",),
        backed_by_check=None,
    ),
)
