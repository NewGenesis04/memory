"""
Complete Temporal Memory System
"""

import uuid
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel

from temporal_memory_models import (
    TimeLabel, TemporalNode, TemporalRelation, TemporalInterval,
    TemporalGraph, Journey, ForemostJourney
)

from temporal_algorithms import (
    ForemostJourneyAlgorithm,
    TemporalConflictDetector,
    TemporalReasoningEngine
)
from sqlite_storage import SQLiteStorage
from caching import Cache


# ==================== Memory Operations ====================

class MemoryOperation(str, Enum):
    """Memory operations"""
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class ExtractedFact(BaseModel):
    """Fact extracted from conversation"""
    subject: str
    predicate: str
    object: str
    confidence: float
    temporal_marker: Optional[str] = None  # "now", "last year", "2024-01-15"
    
    
class TemporalMemorySystem:
    """
    Complete memory system with SQLite persistence and in-memory caching.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initializes the memory system with a storage backend and caching.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.storage = SQLiteStorage(db_path)
        self.storage.initialize_schema()
        
        # Caching layer
        self.graph_cache = Cache(max_size=10, ttl_seconds=5) # Cache for full graph loads
        self.query_cache = Cache(max_size=100, ttl_seconds=60) # Cache for query results
        self.entity_id_cache = Cache(max_size=1000, ttl_seconds=3600) # Cache for entity name->ID
        
        # Load current time from DB or initialize
        current_time_val = self.storage.get_metadata("current_time")
        self.current_time = TimeLabel(value=int(current_time_val))
    
    def _get_graph(self) -> TemporalGraph:
        """
        Helper to load the graph from storage, with caching.
        The cache key includes the current time to ensure freshness.
        """
        cache_key = f"full_graph_{self.current_time.value}"
        cached_graph = self.graph_cache.get(cache_key)
        if cached_graph:
            return cached_graph
        
        graph = self.storage.load_graph_into_memory()
        self.graph_cache.set(cache_key, graph)
        return graph

    # ==================== Time Management ====================
    
    def advance_time(self) -> TimeLabel:
        """Advance to next discrete time step"""
        new_time_val = self.current_time.value + 1
        self.current_time = TimeLabel(value=new_time_val)
        self.storage.set_metadata("current_time", str(new_time_val))
        # Clear caches on time advance as data may have changed
        self.graph_cache.clear()
        self.query_cache.clear()
        return self.current_time
    
    def get_time_label(self, timestamp: Optional[datetime] = None) -> TimeLabel:
        """
        Returns the current time label.
        Note: The demo's simple time management is kept. A production system
        would have a more robust datetime-to-TimeLabel mapping.
        """
        return self.current_time
    
    # ==================== Entity Management ====================
    
    def get_or_create_entity(
        self,
        entity_name: str,
        entity_type: str = "Concept"
    ) -> str:
        """
        Find or create entity node in the database.
        Returns entity ID.
        """
        # Check cache first
        cached_id = self.entity_id_cache.get(entity_name.lower())
        if cached_id:
            return cached_id

        # If not in cache, check DB
        entity_id = self.storage.find_entity_by_name(entity_name)
        if entity_id:
            self.entity_id_cache.set(entity_name.lower(), entity_id)
            return entity_id
        
        # Create new entity
        entity_id = str(uuid.uuid4())
        node = TemporalNode(
            id=entity_id,
            name=entity_name,
            node_type=entity_type,
            created_at=self.current_time
        )
        self.storage.add_node(node)
        self.entity_id_cache.set(entity_name.lower(), entity_id)
        
        return entity_id
    
    # ==================== Core Memory Operations ====================
    
    def process_fact(
        self,
        fact: ExtractedFact,
        timestamp: Optional[datetime] = None
    ) -> Tuple[MemoryOperation, Optional[str]]:
        """
        Process extracted fact and determine operation, persisting to the DB.
        
        Returns:
            (operation, relation_id or None)
        """
        # Invalidate caches since we are modifying data
        self.graph_cache.clear()
        self.query_cache.clear()

        time = self.get_time_label(timestamp)
        
        subject_id = self.get_or_create_entity(fact.subject)
        object_id = self.get_or_create_entity(fact.object)
        
        new_interval = TemporalInterval(valid_from=time, valid_to=None)
        
        exclusive_relations = {"lives_in", "works_at", "is_married_to", "is_a"}
        conflicts = []
        if fact.predicate in exclusive_relations:
            conflicts = self.storage.find_conflicting_relations(
                fact.predicate, subject_id, object_id, new_interval
            )

        if conflicts:
            return self._handle_conflict(
                fact, subject_id, object_id, time, conflicts
            )
        
        existing = self.storage.find_existing_fact(
            subject_id, fact.predicate, object_id, time
        )
        
        if existing:
            return (MemoryOperation.NOOP, existing)
        
        relation_id = self._add_relation(
            subject_id, fact.predicate, object_id, time, fact.confidence
        )
        
        return (MemoryOperation.ADD, relation_id)
    
    def _handle_conflict(
        self,
        fact: ExtractedFact,
        subject_id: str,
        object_id: str,
        time: TimeLabel,
        conflicts: List[str]
    ) -> Tuple[MemoryOperation, Optional[str]]:
        """Handle conflicting facts by updating temporal intervals in the DB."""
        for conflict_id in conflicts:
            self.storage.update_relation_interval(
                conflict_id, TimeLabel(value=time.value - 1)
            )
        
        relation_id = self._add_relation(
            subject_id, fact.predicate, object_id, time, fact.confidence
        )
        
        return (MemoryOperation.UPDATE, relation_id)

    def _add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        time: TimeLabel,
        confidence: float
    ) -> str:
        """Adds a new temporal relation to the database."""
        relation_id = str(uuid.uuid4())
        
        relation = TemporalRelation(
            id=relation_id,
            source_id=subject_id,
            target_id=object_id,
            relation_type=predicate,
            interval=TemporalInterval(valid_from=time, valid_to=None),
            confidence=confidence
        )
        
        self.storage.add_relation(relation)
        return relation_id
    
    # ==================== Querying ====================
    
    def query_current(
        self,
        entity: str,
        relation: str
    ) -> List[str]:
        """Queries the current state from the database, with caching."""
        cache_key = f"query_current_{entity}_{relation}"
        cached_result = self.query_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        graph = self._get_graph()
        reasoning_engine = TemporalReasoningEngine(graph)
        
        entity_id = self._find_entity_id(entity)
        if not entity_id:
            return []
        
        result_ids = reasoning_engine.get_fact_at_time(
            entity_id, relation, self.current_time
        )
        
        results = [graph.nodes[node_id].name for node_id in result_ids]
        self.query_cache.set(cache_key, results)
        return results
    
    def query_at_time(
        self,
        entity: str,
        relation: str,
        time: TimeLabel
    ) -> List[str]:
        """Queries historical state from the database, with caching."""
        cache_key = f"query_at_time_{entity}_{relation}_{time.value}"
        cached_result = self.query_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        graph = self._get_graph()
        reasoning_engine = TemporalReasoningEngine(graph)

        entity_id = self._find_entity_id(entity)
        if not entity_id:
            return []
        
        results_ids = reasoning_engine.get_fact_at_time(
            entity_id, relation, time
        )
        
        results = [graph.nodes[node_id].name for node_id in results_ids]
        self.query_cache.set(cache_key, results)
        return results

    def query_history(
        self,
        entity: str,
        relation: str
    ) -> List[Dict[str, Any]]:
        """Gets the complete history of a relation from the database, with caching."""
        cache_key = f"query_history_{entity}_{relation}"
        cached_result = self.query_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        graph = self._get_graph()
        reasoning_engine = TemporalReasoningEngine(graph)

        entity_id = self._find_entity_id(entity)
        if not entity_id:
            return []
        
        history = reasoning_engine.get_fact_history(entity_id, relation)
        
        result = []
        for start, end, value_id in history:
            result.append({
                "from": start.value,
                "to": end.value if end else "present",
                "value": graph.nodes[value_id].name
            })
        
        self.query_cache.set(cache_key, result)
        return result

    def query_with_reasoning(
        self,
        entity: str,
        query: str,
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """Performs multi-hop reasoning by loading the graph, with caching."""
        cache_key = f"reasoning_{entity}_{query}_{max_hops}"
        cached_result = self.query_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        graph = self._get_graph()
        reasoning_engine = TemporalReasoningEngine(graph)

        entity_id = self._find_entity_id(entity)
        if not entity_id:
            return {}
        
        inferences = reasoning_engine.infer_at_time(
            entity_id, max_hops, self.current_time
        )
        
        result = {}
        for target_id, journeys in inferences.items():
            target_name = graph.nodes[target_id].name
            
            journey = journeys[0] if journeys else None
            if journey:
                path = self._journey_to_readable_path(journey, graph)
                result[target_name] = {
                    "confidence": self._calculate_journey_confidence(journey, graph),
                    "reasoning_path": path,
                    "hops": len(journey.nodes) - 1
                }
        
        self.query_cache.set(cache_key, result)
        return result
    
    # ==================== Helper Methods ====================
    
    def _find_entity_id(self, entity_name: str) -> Optional[str]:
        """Find entity ID by name from storage, with caching."""
        cached_id = self.entity_id_cache.get(entity_name.lower())
        if cached_id:
            return cached_id
        
        entity_id = self.storage.find_entity_by_name(entity_name)
        if entity_id:
            self.entity_id_cache.set(entity_name.lower(), entity_id)
        return entity_id
    
    def _journey_to_readable_path(self, journey: Journey, graph: TemporalGraph) -> List[str]:
        """Convert journey to readable path."""
        path = []
        for i in range(len(journey.nodes) - 1):
            from_node = graph.nodes[journey.nodes[i]].name
            to_node = graph.nodes[journey.nodes[i + 1]].name
            
            edges = graph.get_instance_at(journey.time_labels[i])
            relation = "→"
            for edge in edges:
                if (edge.source_id == journey.nodes[i] and
                    edge.target_id == journey.nodes[i + 1]):
                    relation = edge.relation_type
                    break
            
            path.append(f"{from_node} --[{relation}]--> {to_node}")
        
        return path
    
    def _calculate_journey_confidence(self, journey: Journey, graph: TemporalGraph) -> float:
        """Calculate confidence of inferred fact based on journey."""
        confidence = 1.0
        for i, time_label in enumerate(journey.time_labels):
            edges = graph.get_instance_at(time_label)
            for edge in edges:
                if (edge.source_id == journey.nodes[i] and
                    edge.target_id == journey.nodes[i + 1]):
                    confidence *= edge.confidence
                    break
        return confidence
    
    # ==================== Statistics ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics from the database."""
        graph = self._get_graph()
        return {
            "num_entities": len(graph.nodes),
            "num_relations": len(graph.relations),
            "temporal_age": graph.age,
            "current_time": self.current_time.value,
            "time_range": {
                "min": graph.lambda_min.value,
                "max": graph.lambda_max.value
            }
        }

    def close(self):
        """Closes the storage connection."""
        self.storage.close()

# ==================== Export ====================

__all__ = [
    'TemporalMemorySystem',
    'ExtractedFact',
    'MemoryOperation',
]