# Temporal Memory System

This project is a temporal knowledge graph designed to store and reason about facts that change over time. It provides a persistent, queryable memory layer that can understand temporal relationships, resolve conflicts, and perform multi-hop inference.

The system is exposed via a REST API built with FastAPI.

## Core Concepts

- **Temporal Graph**: The central data structure is a graph where relationships (edges) are valid only during specific time intervals. This allows the system to represent how facts change. For example, a person `lives_in` "New York" from time `t1` to `t2`, and then `lives_in` "London" from `t2` onwards.

- **Conflict Resolution**: When a new fact contradicts an existing one (e.g., a person can only live in one city at a time), the system automatically ends the previous fact's time interval and starts a new one for the new fact.

- **Inference**: The system can perform multi-hop queries to find inferred relationships. For example, if "Bob `works_at` Acme Inc." and "Acme Inc. `is_located_in` California", the system can infer that "Bob `is_in` California".

## Architecture

- **Storage**: A SQLite database (`production_memory.db`) provides persistence. The schema is optimized for temporal queries, with indices on time intervals (`valid_from`, `valid_to`).

- **API**: A FastAPI server (`api_server.py`) exposes the memory system's functionality through a set of REST endpoints.

- **Caching**: A simple, in-memory LRU cache (`caching.py`) is used to improve performance for frequently accessed data, such as query results and entity lookups.

## Getting Started

### Prerequisites

- Python 3.12+

### Installation

1.  Clone the repository.
2.  It is recommended to use a virtual environment:
    ```sh
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  Install the dependencies using `uv`:
    ```sh
    pip install uv
    uv pip install -r requirements.txt 
    ```
    *(If you don't have a `requirements.txt`, you can install from `pyproject.toml`)*
    ```sh
    uv pip install -e .
    ```


### Running the API Server

To start the API server, run:

```sh
uvicorn api_server:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

## API Endpoints

- `POST /facts`: Add a new fact to the memory system.
- `POST /query/current`: Query the current value of a relation for an entity.
- `POST /query/history`: Retrieve the complete temporal history of a relation.
- `POST /reason`: Perform multi-hop reasoning to infer new facts.
- `GET /stats`: Get statistics about the memory system.

Refer to the `/docs` endpoint for detailed request and response models.
