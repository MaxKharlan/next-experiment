"""Transactional SQLite snapshots for the single-process API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column, MetaData, String, Table, create_engine, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import make_url


class SQLiteInvestigations:
    def __init__(self, database_url: str):
        url = make_url(database_url)
        if (
            url.get_backend_name() != "sqlite"
            or not url.database
            or url.database == ":memory:"
        ):
            raise ValueError(
                "Investigation persistence requires a file-backed SQLite URL"
            )
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, connect_args={"timeout": 30})
        schema = MetaData()
        self.snapshots = Table(
            "investigation_snapshots",
            schema,
            Column("id", String, primary_key=True),
            Column("metadata", JSON, nullable=False),
            Column("events", JSON, nullable=False),
        )
        schema.create_all(self.engine)

    def save(self, meta: dict[str, Any], events: list[dict[str, Any]]) -> None:
        statement = insert(self.snapshots).values(
            id=meta["id"],
            metadata=meta,
            events=events,
        )
        statement = statement.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "metadata": statement.excluded.metadata,
                "events": statement.excluded.events,
            },
        )
        with self.engine.begin() as connection:
            connection.execute(statement)

    def load(self, inv_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(self.snapshots.c.metadata, self.snapshots.c.events).where(
                    self.snapshots.c.id == inv_id
                )
            ).first()
        if row is None:
            raise KeyError(inv_id)
        return row[0], row[1]

    def ids(self) -> list[str]:
        with self.engine.connect() as connection:
            return list(connection.scalars(select(self.snapshots.c.id)))
