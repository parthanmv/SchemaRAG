"""Deterministic seed script for the SchemaRAG college database.

Run from the ``backend`` directory:

    python -m app.scripts.seed_database

The script drops and recreates all tables from the ORM models, generates a
fully deterministic, relationally consistent dataset, bulk-inserts it, resets
identity sequences, and verifies row-count targets.

Every run with the same seed produces byte-identical data (verified via a
SHA-256 digest printed at the end).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import insert, text

from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.models import (
    Attendance,
    Course,
    Department,
    Enrollment,
    Mark,
    Student,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("seed")

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
SEED = 42
N_DEPARTMENTS = 8
N_STUDENTS = 1_000
N_COURSES = 50
MIN_ENROLLMENTS = 5_000
EXAM_TYPES: tuple[str, ...] = ("quiz", "midterm", "final")

# Academic context of the generated data.
CURRENT_YEAR = 2026          # used to derive each student's current semester
TERM_ACADEMIC_YEAR = 2025    # academic year (2025-26) of enrolment/marks/attendance


FIRST_NAMES = (
    "Aarav", "Aditi", "Akash", "Amit", "Ananya", "Aniket", "Anjali", "Arjun",
    "Bhavya", "Deepak", "Diya", "Esha", "Gaurav", "Harsh", "Ishaan", "Ishita",
    "Kabir", "Karan", "Kiran", "Kritika", "Manish", "Meera", "Mohit", "Neha",
    "Nikhil", "Nisha", "Pooja", "Pranav", "Priya", "Rahul", "Riya", "Rohan",
    "Sahil", "Sanjana", "Shreya", "Simran", "Sneha", "Soham", "Tanvi", "Uday",
    "Varun", "Vibha", "Vikram", "Vinay", "Yash", "Aisha", "Devansh", "Nandini",
)

LAST_NAMES = (
    "Agarwal", "Bhatt", "Chauhan", "Desai", "Deshmukh", "Dubey", "Ghosh",
    "Gupta", "Iyer", "Jain", "Joshi", "Kapoor", "Khan", "Kulkarni", "Malhotra",
    "Mehta", "Menon", "Mishra", "Nair", "Naik", "Pandey", "Patel", "Pillai",
    "Rao", "Reddy", "Saxena", "Sharma", "Shetty", "Singh", "Sinha", "Trivedi",
    "Verma",
)


@dataclass
class Dataset:
    """In-memory representation of the full generated dataset."""

    departments: list[dict] = field(default_factory=list)
    courses: list[dict] = field(default_factory=list)
    students: list[dict] = field(default_factory=list)
    enrollments: list[dict] = field(default_factory=list)
    marks: list[dict] = field(default_factory=list)
    attendance: list[dict] = field(default_factory=list)

    def tables(self) -> dict[str, list[dict]]:
        return {
            "departments": self.departments,
            "courses": self.courses,
            "students": self.students,
            "enrollments": self.enrollments,
            "marks": self.marks,
            "attendance": self.attendance,
        }


DEPARTMENTS: tuple[tuple[str, str], ...] = (
    ("Computer Science and Engineering", "CSE"),
    ("Electronics and Communication Engineering", "ECE"),
    ("Electrical Engineering", "EE"),
    ("Mechanical Engineering", "ME"),
    ("Civil Engineering", "CE"),
    ("Information Technology", "IT"),
    ("Chemical Engineering", "CHE"),
    ("Mathematics", "MATH"),
)

COURSE_TITLES: dict[str, tuple[str, ...]] = {
    "CSE": (
        "Introduction to Programming", "Data Structures", "Design and Analysis of Algorithms",
        "Database Management Systems", "Operating Systems", "Computer Networks",
        "Software Engineering", "Machine Learning Fundamentals",
    ),
    "ECE": (
        "Digital Logic Design", "Signals and Systems", "Analog Electronics",
        "Microprocessors and Microcontrollers", "Communication Systems", "VLSI Design",
        "Embedded Systems",
    ),
    "EE": (
        "Network Theory", "Electrical Machines", "Power System Analysis",
        "Control Systems", "Power Electronics", "Electric Drives",
    ),
    "ME": (
        "Engineering Mechanics", "Thermodynamics", "Fluid Mechanics",
        "Manufacturing Processes", "Machine Design", "Heat Transfer",
    ),
    "CE": (
        "Structural Analysis", "Geotechnical Engineering", "Transportation Engineering",
        "Environmental Engineering", "Surveying", "Concrete Technology",
    ),
    "IT": (
        "Web Technologies", "Object Oriented Analysis and Design", "Computer Graphics",
        "Information Security", "Cloud Computing", "Data Warehousing and Mining",
    ),
    "CHE": (
        "Chemical Process Principles", "Mass Transfer Operations", "Heat Transfer Operations",
        "Chemical Reaction Engineering", "Process Dynamics and Control", "Plant Design",
    ),
    "MATH": (
        "Linear Algebra", "Real Analysis", "Probability and Statistics",
        "Discrete Mathematics", "Numerical Methods", "Operations Research",
    ),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _manual_binomial(rng: random.Random, n: int, p: float) -> int:
    """Binomial sampler built from Bernoulli trials (portable & deterministic)."""
    return sum(1 for _ in range(n) if rng.random() < p)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def generate_departments(ds: Dataset, rng: random.Random) -> None:
    del rng  # departments are fixed; parameter kept for a uniform generator API
    for idx, (name, code) in enumerate(DEPARTMENTS[:N_DEPARTMENTS], start=1):
        ds.departments.append(
            {"department_id": idx, "department_name": name, "department_code": code}
        )


def generate_courses(ds: Dataset, rng: random.Random) -> None:
    # Distribute N_COURSES across departments: 7 for the first two, 6 for the rest.
    per_dept = [7, 7] + [6] * (N_DEPARTMENTS - 2)
    course_id = 0
    for dept, count in zip(ds.departments, per_dept):
        code = dept["department_code"]
        titles = COURSE_TITLES[code][:count]
        if len(titles) < count:  # defensive: pool must cover allocation
            raise ValueError(f"Not enough course titles configured for {code}")
        for offset, title in enumerate(titles):
            course_id += 1
            ds.courses.append(
                {
                    "course_id": course_id,
                    "course_code": f"{code}{100 + offset}",
                    "course_name": title,
                    "credits": rng.choices([2, 3, 4, 5], weights=[1, 4, 4, 1])[0],
                    "department_id": dept["department_id"],
                }
            )


def generate_students(ds: Dataset, rng: random.Random) -> None:
    dept_ids = [d["department_id"] for d in ds.departments]
    dept_codes = {d["department_id"]: d["department_code"] for d in ds.departments}
    # Weighted admission years: recent cohorts are larger.
    years = list(range(2019, CURRENT_YEAR + 1))
    year_weights = [2, 3, 5, 10, 15, 20, 25, 20]

    roll_counters: dict[tuple[str, int], int] = {}
    for student_id in range(1, N_STUDENTS + 1):
        department_id = rng.choice(dept_ids)
        admission_year = rng.choices(years, weights=year_weights)[0]
        base_semester = (CURRENT_YEAR - admission_year) * 2 - 1
        semester = _clamp(base_semester + rng.choice([0, 1]), 1, 8)

        key = (dept_codes[department_id], admission_year)
        roll_counters[key] = roll_counters.get(key, 0) + 1
        roll_number = f"{key[0]}{admission_year}{roll_counters[key]:03d}"

        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        ds.students.append(
            {
                "student_id": student_id,
                "roll_number": roll_number,
                "name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}.{student_id}@college.edu",
                "department_id": department_id,
                "semester": semester,
                "admission_year": admission_year,
            }
        )


def generate_enrollments(ds: Dataset, rng: random.Random) -> None:
    """Enrol every student in 4-6 courses, mostly from their own department."""
    dept_ids = [d["department_id"] for d in ds.departments]
    courses_by_dept: dict[int, list[dict]] = {}
    for course in ds.courses:
        courses_by_dept.setdefault(course["department_id"], []).append(course)

    seen_pairs: set[tuple[int, int]] = set()
    for student in ds.students:
        target = rng.choices([4, 5, 6], weights=[15, 55, 30])[0]
        home_pool = courses_by_dept[student["department_id"]]
        enrolled: list[dict] = []
        # Occasional open elective from another department.
        if rng.random() < 0.12:
            other = rng.choice([d for d in dept_ids if d != student["department_id"]])
            elective = rng.choice(courses_by_dept[other])
            enrolled.append(elective)
            seen_pairs.add((student["student_id"], elective["course_id"]))
        remaining_home = [c for c in home_pool
                          if (student["student_id"], c["course_id"]) not in seen_pairs]
        needed = min(target - len(enrolled), len(remaining_home))
        enrolled.extend(rng.sample(remaining_home, needed))

        for course in enrolled:
            pair = (student["student_id"], course["course_id"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            ds.enrollments.append(
                {
                    "enrollment_id": len(ds.enrollments) + 1,
                    "student_id": student["student_id"],
                    "course_id": course["course_id"],
                    "academic_year": TERM_ACADEMIC_YEAR,
                    "semester": student["semester"],
                }
            )

    # Top up to guarantee the MIN_ENROLLMENTS target deterministically.
    if len(ds.enrollments) < MIN_ENROLLMENTS:
        for student in ds.students:
            if len(ds.enrollments) >= MIN_ENROLLMENTS:
                break
            home_pool = courses_by_dept[student["department_id"]]
            taken = {e["course_id"] for e in ds.enrollments
                     if e["student_id"] == student["student_id"]}
            available = [c for c in home_pool if c["course_id"] not in taken]
            if available:
                course = rng.choice(available)
                ds.enrollments.append(
                    {
                        "enrollment_id": len(ds.enrollments) + 1,
                        "student_id": student["student_id"],
                        "course_id": course["course_id"],
                        "academic_year": TERM_ACADEMIC_YEAR,
                        "semester": student["semester"],
                    }
                )


def generate_marks_and_attendance(ds: Dataset, rng: random.Random) -> None:
    """Generate marks + attendance driven by stable per-student/per-course factors.

    - ``ability``      : latent skill of a student (correlates their marks).
    - ``dept_offset``  : department-level difficulty/ease shift.
    - ``difficulty``   : course-level difficulty shift.
    - ``diligence``    : latent class-attendance habit of a student.
    """
    ability = {s["student_id"]: rng.uniform(45, 92) for s in ds.students}
    diligence = {s["student_id"]: rng.uniform(0.58, 0.99) for s in ds.students}
    dept_offset = {d["department_id"]: rng.uniform(-6, 6) for d in ds.departments}
    difficulty = {c["course_id"]: rng.uniform(-8, 8) for c in ds.courses}

    for enrollment in ds.enrollments:
        sid = enrollment["student_id"]
        cid = enrollment["course_id"]
        base = ability[sid] + dept_offset[_course_dept(ds, cid)] + difficulty[cid]

        for exam_type in EXAM_TYPES:
            noise = rng.gauss(0, 7) - (3 if exam_type == "quiz" else 0)
            ds.marks.append(
                {
                    "mark_id": len(ds.marks) + 1,
                    "student_id": sid,
                    "course_id": cid,
                    "exam_type": exam_type,
                    "marks": round(_clamp(base + noise, 0, 100), 2),
                    "academic_year": enrollment["academic_year"],
                    "semester": enrollment["semester"],
                }
            )

        held = rng.randint(30, 44)
        attended_prob = _clamp(diligence[sid] + rng.gauss(0, 0.05), 0.0, 1.0)
        attended = _manual_binomial(rng, held, attended_prob)
        # Decimal + ROUND_HALF_UP matches PostgreSQL's ROUND(numeric) exactly,
        # so the stored percentage always equals the recomputed value.
        percentage = (
            (Decimal(attended) * 100 / Decimal(held))
            .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            if held
            else Decimal("0.00")
        )
        ds.attendance.append(
            {
                "attendance_id": len(ds.attendance) + 1,
                "student_id": sid,
                "course_id": cid,
                "classes_held": held,
                "classes_attended": attended,
                "attendance_percentage": percentage,
                "academic_year": enrollment["academic_year"],
                "semester": enrollment["semester"],
            }
        )


_COURSE_DEPT_CACHE: dict[int, int] = {}


def _course_dept(ds: Dataset, course_id: int) -> int:
    """Map course_id -> department_id (cached)."""
    if not _COURSE_DEPT_CACHE:
        for course in ds.courses:
            _COURSE_DEPT_CACHE[course["course_id"]] = course["department_id"]
    return _COURSE_DEPT_CACHE[course_id]


def generate_dataset(seed: int = SEED) -> Dataset:
    """Produce the complete deterministic dataset."""
    global _COURSE_DEPT_CACHE
    _COURSE_DEPT_CACHE = {}

    rng = random.Random(seed)
    ds = Dataset()
    generate_departments(ds, rng)
    generate_courses(ds, rng)
    generate_students(ds, rng)
    generate_enrollments(ds, rng)
    generate_marks_and_attendance(ds, rng)
    return ds


def dataset_digest(ds: Dataset) -> str:
    """SHA-256 digest over the canonical JSON encoding of every table."""
    hasher = hashlib.sha256()
    for name, rows in ds.tables().items():
        hasher.update(name.encode())
        hasher.update(
            json.dumps(
                rows, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        )
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------
def init_schema() -> None:
    """Drop and recreate all tables from the ORM metadata."""
    logger.info("Dropping and recreating schema ...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


_TABLE_MODEL_ORDER: tuple[type, ...] = (
    Department, Course, Student, Enrollment, Mark, Attendance,
)


def load_dataset(ds: Dataset) -> None:
    """Bulk-insert the dataset (single transaction, explicit PKs)."""
    with SessionLocal() as session, session.begin():
        for model in _TABLE_MODEL_ORDER:
            rows = getattr(ds, model.__tablename__)
            if rows:
                session.execute(insert(model), rows)
        # Explicit PKs bypass the identity sequences; realign them so future
        # INSERTs cannot collide. Table/column names come from ORM metadata,
        # never from user input, so f-string interpolation is safe here.
        for model in _TABLE_MODEL_ORDER:
            table = model.__tablename__
            pk_col = model.__mapper__.primary_key[0].name
            session.execute(
                text(
                    f"SELECT setval("
                    f"pg_get_serial_sequence('{table}', '{pk_col}'), "
                    f"COALESCE((SELECT MAX({pk_col}) FROM {table}), 1))"
                )
            )
    logger.info("Dataset loaded.")


def verify_counts() -> dict[str, int]:
    """Read back row counts and enforce Phase 1 minimum targets."""
    counts: dict[str, int] = {}
    with engine.connect() as conn:
        for model in _TABLE_MODEL_ORDER:
            counts[model.__tablename__] = conn.execute(
                text(f"SELECT COUNT(*) FROM {model.__tablename__}")
            ).scalar_one()

    requirements = {
        "departments": N_DEPARTMENTS,
        "students": N_STUDENTS,
        "courses": N_COURSES,
        "enrollments": MIN_ENROLLMENTS,
        "marks": MIN_ENROLLMENTS,
        "attendance": MIN_ENROLLMENTS,
    }
    failures = [
        f"{tbl}: {counts[tbl]} < required {req}"
        for tbl, req in requirements.items()
        if counts[tbl] < req
    ]
    if failures:
        raise RuntimeError("Row-count verification failed: " + "; ".join(failures))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the SchemaRAG college database.")
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--skip-schema-reset",
        action="store_true",
        help="Insert into existing tables instead of dropping/recreating them.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate and validate data without touching the DB."
    )
    args = parser.parse_args(argv)

    ds = generate_dataset(args.seed)
    logger.info(
        "Generated: %d departments, %d courses, %d students, %d enrollments, "
        "%d marks, %d attendance records.",
        len(ds.departments), len(ds.courses), len(ds.students),
        len(ds.enrollments), len(ds.marks), len(ds.attendance),
    )
    logger.info("Dataset digest (seed=%d): %s", args.seed, dataset_digest(ds))

    if args.dry_run:
        logger.info("Dry run requested - nothing written to the database.")
        return 0

    if not args.skip_schema_reset:
        init_schema()
    load_dataset(ds)
    counts = verify_counts()
    logger.info("Row counts: %s", counts)
    logger.info("Seed completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
