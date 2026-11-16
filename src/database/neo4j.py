"""Conexão parametrizada com o Neo4j para consultas do backend."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar


class Neo4jConfigurationError(ValueError):
    """Configuração necessária para conectar ao Neo4j está ausente ou inválida."""


class _Record(Protocol):
    def data(self) -> dict[str, Any]: ...


class _Transaction(Protocol):
    def run(
        self, query: str, parameters: Mapping[str, object]
    ) -> Iterable[_Record]: ...


_Result = TypeVar("_Result")


class _Session(Protocol):
    def __enter__(self) -> _Session: ...
    def __exit__(self, *args: object) -> None: ...

    def execute_read(
        self, work: Callable[[_Transaction], _Result]
    ) -> _Result: ...

    def execute_write(
        self, work: Callable[[_Transaction], _Result]
    ) -> _Result: ...


class _Driver(Protocol):
    def session(self, *, database: str) -> _Session: ...
    def verify_connectivity(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class Neo4jConfig:
    """Credenciais e database do Neo4j; a senha não aparece no repr."""

    uri: str
    username: str
    password: str = field(repr=False)
    database: str = "neo4j"


class Neo4jDatabase:
    """Executa consultas parametrizadas em transações gerenciadas pelo driver."""

    def __init__(self, config: Neo4jConfig, driver: _Driver | None = None) -> None:
        self._config = config
        if driver is None:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                config.uri,
                auth=(config.username, config.password),
            )
        self._driver = driver

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> Neo4jDatabase:
        """Carrega configuração do .env e do ambiente sem sobrescrever o ambiente."""
        from dotenv import load_dotenv

        load_dotenv(
            dotenv_path=dotenv_path or (Path.cwd() / ".env"),
            override=False,
        )
        uri = os.getenv("NEO4J_URI") or ""
        password = os.getenv("NEO4J_PASSWORD") or ""
        missing = [
            name
            for name, value in (("NEO4J_URI", uri), ("NEO4J_PASSWORD", password))
            if not value
        ]
        if missing:
            raise Neo4jConfigurationError(
                "Defina as variáveis obrigatórias: " + ", ".join(missing)
            )

        return cls(
            Neo4jConfig(
                uri=uri,
                username=os.getenv("NEO4J_USERNAME") or "neo4j",
                password=password,
                database=os.getenv("NEO4J_DATABASE") or "neo4j",
            )
        )

    def verify_connectivity(self) -> None:
        """Falha explicitamente se o servidor não estiver acessível."""
        self._driver.verify_connectivity()

    @staticmethod
    def _run(
        transaction: _Transaction,
        query: str,
        parameters: Mapping[str, object] | None,
    ) -> list[dict[str, Any]]:
        return [
            record.data()
            for record in transaction.run(query, dict(parameters or {}))
        ]

    def execute_read(
        self,
        query: str,
        parameters: Mapping[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        """Executa uma consulta de leitura; dados externos devem ir em parameters."""
        with self._driver.session(database=self._config.database) as session:
            return session.execute_read(
                lambda transaction: self._run(transaction, query, parameters)
            )

    def execute_write(
        self,
        query: str,
        parameters: Mapping[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        """Executa uma escrita transacional com parâmetros separados da consulta."""
        with self._driver.session(database=self._config.database) as session:
            return session.execute_write(
                lambda transaction: self._run(transaction, query, parameters)
            )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jDatabase:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()