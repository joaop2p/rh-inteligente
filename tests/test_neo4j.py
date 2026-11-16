from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from src.database.neo4j import (
    Neo4jConfigurationError,
    Neo4jDatabase,
    Neo4jConfig,
)


class FakeRecord:
    def data(self) -> dict[str, Any]:
        return {"value": 7}


class FakeTransaction:
    def __init__(self) -> None:
        self.query: str | None = None
        self.parameters: Mapping[str, object] | None = None

    def run(
        self, query: str, parameters: Mapping[str, object]
    ) -> list[FakeRecord]:
        self.query = query
        self.parameters = parameters
        return [FakeRecord()]


class FakeSession:
    def __init__(self, transaction: FakeTransaction) -> None:
        self.transaction = transaction
        self.operation: str | None = None

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute_read(
        self, work: Callable[[FakeTransaction], list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        self.operation = "read"
        return work(self.transaction)

    def execute_write(
        self, work: Callable[[FakeTransaction], list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        self.operation = "write"
        return work(self.transaction)


class FakeDriver:
    def __init__(self) -> None:
        self.transaction = FakeTransaction()
        self.session_instance: FakeSession | None = None
        self.database: str | None = None
        self.closed = False
        self.verified = False

    def session(self, *, database: str) -> FakeSession:
        self.database = database
        self.session_instance = FakeSession(self.transaction)
        return self.session_instance

    def verify_connectivity(self) -> None:
        self.verified = True

    def close(self) -> None:
        self.closed = True


def test_from_env_reads_dotenv_without_exposing_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "NEO4J_URI=bolt://localhost:7687\n"
        "NEO4J_USERNAME=neo4j\n"
        "NEO4J_PASSWORD=secret-test-value\n"
        "NEO4J_DATABASE=hr\n",
        encoding="utf-8",
    )
    for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
        monkeypatch.delenv(key, raising=False)

    driver = FakeDriver()
    import neo4j

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *args, **kwargs: driver)
    database = Neo4jDatabase.from_env(dotenv_path)

    assert database._config.uri == "bolt://localhost:7687"
    assert database._config.database == "hr"
    assert "secret-test-value" not in repr(database._config)
    database.close()
    assert driver.closed


def test_from_env_requires_uri_and_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("NEO4J_URI", "NEO4J_PASSWORD"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(Neo4jConfigurationError, match="NEO4J_URI, NEO4J_PASSWORD"):
        Neo4jDatabase.from_env(tmp_path / "missing.env")


@pytest.mark.parametrize("operation", ["execute_read", "execute_write"])
def test_queries_are_parameterized_and_use_configured_database(
    operation: str,
) -> None:
    driver = FakeDriver()
    database = Neo4jDatabase(
        Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="secret",
            database="hr",
        ),
        driver=driver,
    )

    result = getattr(database, operation)(
        "MATCH (p:Pessoa {funcionario_id: $id}) RETURN p.nome AS value",
        {"id": "E001"},
    )

    assert result == [{"value": 7}]
    assert driver.database == "hr"
    assert driver.session_instance is not None
    assert driver.session_instance.operation == (
        "read" if operation == "execute_read" else "write"
    )
    assert driver.transaction.query == (
        "MATCH (p:Pessoa {funcionario_id: $id}) RETURN p.nome AS value"
    )
    assert driver.transaction.parameters == {"id": "E001"}


def test_context_manager_closes_driver() -> None:
    driver = FakeDriver()
    with Neo4jDatabase(
        Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="secret",
        ),
        driver=driver,
    ) as database:
        database.verify_connectivity()

    assert driver.verified
    assert driver.closed
