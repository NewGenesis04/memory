"""
SQLite Storage Backend for Temporal Memory System
"""

import sqlite3
import json
from typing import List, Optional, Dict, Tuple, Any

from temporal_memory_models import (
    TimeLabel, TemporalNode, TemporalRelation, TemporalInterval, TemporalGraph
)

class SQLiteStorage:
    """
    Handles all database interactions for the temporal memory system,
    persisting the temporal graph to a SQLite database.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initializes the storage backend.
        
        Args:
            db_path: Path to the SQLite database file.
                     Defaults to ":memory:" for an in-memory database.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
    def initialize_schema(self):
        """
        Creates the necessary tables and indices in the database.
        This is a direct implementation of the schema from production_guide.md.
        """
        with self.conn:
            # Enable optimizations
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA synchronous = NORMAL;")
            self.conn.execute("PRAGMA temp_store = MEMORY;")

            # Entities table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                node_type TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                attributes TEXT,
                embedding BLOB
            ) WITHOUT ROWID;
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name COLLATE NOCASE);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(node_type);")

            # Temporal relations table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS temporal_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                valid_from INTEGER NOT NULL,
                valid_to INTEGER,
                confidence REAL NOT NULL DEFAULT 1.0,
                attributes TEXT,
                FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
            ) WITHOUT ROWID;
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source_time ON temporal_relations(source_id, valid_from);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_validity ON temporal_relations(valid_from, valid_to) WHERE valid_to IS NOT NULL;")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_active ON temporal_relations(source_id, relation_type) WHERE valid_to IS NULL;")

            # Metadata table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            """)
            self.conn.execute("INSERT OR IGNORE INTO system_metadata (key, value) VALUES ('current_time', '0');")

    def add_node(self, node: TemporalNode):
        """Adds a new node to the entities table."""
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO entities (id, name, node_type, created_at, attributes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.name,
                    node.node_type,
                    node.created_at.value,
                    json.dumps(node.attributes) if node.attributes else None,
                ),
            )

    def add_relation(self, relation: TemporalRelation):
        """Adds a new relation to the temporal_relations table."""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO temporal_relations (id, source_id, target_id, relation_type, valid_from, valid_to, confidence, attributes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation.id,
                    relation.source_id,
                    relation.target_id,
                    relation.relation_type,
                    relation.interval.valid_from.value,
                    relation.interval.valid_to.value if relation.interval.valid_to else None,
                    relation.confidence,
                    json.dumps(relation.attributes) if relation.attributes else None,
                ),
            )

    def update_relation_interval(self, relation_id: str, valid_to: TimeLabel):
        """Updates the valid_to field of an existing relation."""
        with self.conn:
            self.conn.execute(
                "UPDATE temporal_relations SET valid_to = ? WHERE id = ?",
                (valid_to.value, relation_id),
            )

    def find_entity_by_name(self, name: str) -> Optional[str]:
        """Finds an entity by name and returns its ID."""
        cursor = self.conn.execute("SELECT id FROM entities WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row["id"] if row else None

    def find_conflicting_relations(
        self, relation_type: str, source_id: str, target_id: str, interval: TemporalInterval
    ) -> List[str]:
        """Finds relations that conflict with a new potential fact."""
        query = """
        SELECT id FROM temporal_relations
        WHERE source_id = ?
          AND relation_type = ?
          AND target_id != ?
          AND (valid_to IS NULL OR valid_to >= ?)
          AND valid_from <= ?
        """
        
        # For open-ended intervals
        end_from = interval.valid_to.value if interval.valid_to else interval.valid_from.value

        cursor = self.conn.execute(query, (source_id, relation_type, target_id, interval.valid_from.value, end_from))
        return [row["id"] for row in cursor.fetchall()]

    def find_existing_fact(
        self, subject_id: str, predicate: str, object_id: str, time: TimeLabel
    ) -> Optional[str]:
        """Checks if an identical, active fact already exists."""
        cursor = self.conn.execute(
            """
            SELECT id FROM temporal_relations
            WHERE source_id = ?
              AND relation_type = ?
              AND target_id = ?
              AND valid_from <= ?
              AND (valid_to IS NULL OR valid_to >= ?)
            """,
            (subject_id, predicate, object_id, time.value, time.value),
        )
        row = cursor.fetchone()
        return row["id"] if row else None

    def get_metadata(self, key: str) -> Optional[str]:
        """Retrieves a value from the metadata table."""
        cursor = self.conn.execute("SELECT value FROM system_metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else None

    def set_metadata(self, key: str, value: str):
        """Sets a value in the metadata table."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO system_metadata (key, value) VALUES (?, ?)", (key, value)
            )

    def load_graph_into_memory(self) -> TemporalGraph:
        """
        Loads the entire graph from SQLite into an in-memory TemporalGraph object.
        This is useful for algorithms that need the full graph structure.
        """
        graph = TemporalGraph()
        
        # Load nodes
        for row in self.conn.execute("SELECT * FROM entities"):
            node = TemporalNode(
                id=row["id"],
                name=row["name"],
                node_type=row["node_type"],
                created_at=TimeLabel(value=row["created_at"]),
                attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            )
            graph.add_node(node)
            
        # Load relations
        for row in self.conn.execute("SELECT * FROM temporal_relations"):
            interval = TemporalInterval(
                valid_from=TimeLabel(value=row["valid_from"]),
                valid_to=TimeLabel(value=row["valid_to"]) if row["valid_to"] is not None else None,
            )
            relation = TemporalRelation(
                id=row["id"],
                source_id=row["source_id"],
                target_id=row["target_id"],
                relation_type=row["relation_type"],
                interval=interval,
                confidence=row["confidence"],
                attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            )
            graph.add_relation(relation)
            
        return graph

    def close(self):
        """Closes the database connection."""
        self.conn.close()

__all__ = ["SQLiteStorage"]
