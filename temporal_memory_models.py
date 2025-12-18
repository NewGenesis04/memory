"""
Temporal Graph Memory System - Data Models
Based on "An Introduction to Temporal Graphs: An Algorithmic Perspective"

Key concepts from paper:
- Journey: path with strictly increasing time-labels
- Temporal diameter: max time to reach any node from any time-node
- Out-disjoint journeys: non-conflicting temporal paths
- Foremost journey: earliest-arriving path from time t
"""

from typing import List, Optional, Dict, Set, Tuple, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import json


# ==================== Temporal Graph Core Models ====================

class TimeLabel(BaseModel):
    """
    A discrete time moment (from paper Section 2)
    Can represent: seconds, days, or discrete steps
    """
    value: int  # Discrete time value
    timestamp: Optional[datetime] = None  # Human-readable time
    
    def __lt__(self, other: 'TimeLabel') -> bool:
        return self.value < other.value
    
    def __le__(self, other: 'TimeLabel') -> bool:
        return self.value <= other.value
    
    def __eq__(self, other: 'TimeLabel') -> bool:
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)


class TemporalNode(BaseModel):
    """
    Node in temporal graph (from paper Section 2)
    Represents an entity that exists across time
    """
    id: str
    name: str
    node_type: str  # Person, Location, Concept, etc.
    attributes: Dict[str, Any] = Field(default_factory=dict)
    created_at: TimeLabel
    
    class Config:
        frozen = True  # Immutable for use as dict key


class TemporalEdge(BaseModel):
    """
    Edge with time-label (from paper Section 2)
    λ : E → 2^ℕ (labeling function assigning natural numbers to edges)
    """
    source_id: str
    target_id: str
    relation_type: str
    time_label: TimeLabel  # When this edge is available
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.time_label.value))


class TemporalInterval(BaseModel):
    """
    Time interval for edge validity
    Used for efficient storage: instead of storing edge at each time,
    store once with [valid_from, valid_to] interval
    """
    valid_from: TimeLabel
    valid_to: Optional[TimeLabel] = None  # None means "still valid"
    
    def contains(self, time: TimeLabel) -> bool:
        """Check if time is within this interval"""
        if time < self.valid_from:
            return False
        if self.valid_to is None:
            return True
        return time <= self.valid_to
    
    def overlaps(self, other: 'TemporalInterval') -> bool:
        """Check if two intervals overlap (for conflict detection)"""
        # If either is open-ended, check start times
        if self.valid_to is None or other.valid_to is None:
            return not (self.valid_from > other.valid_to if other.valid_to else False or
                       other.valid_from > self.valid_to if self.valid_to else False)
        
        # Both have end times
        return not (self.valid_from > other.valid_to or other.valid_from > self.valid_to)


class TemporalRelation(BaseModel):
    """
    Relation with temporal validity interval
    More efficient than storing TemporalEdge for each time point
    """
    id: str
    source_id: str
    target_id: str
    relation_type: str
    interval: TemporalInterval
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    
    def is_valid_at(self, time: TimeLabel) -> bool:
        """Check if this relation is valid at given time"""
        return self.interval.contains(time)
    
    def to_temporal_edge(self, time: TimeLabel) -> Optional[TemporalEdge]:
        """Convert to temporal edge if valid at given time"""
        if not self.is_valid_at(time):
            return None
        
        return TemporalEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            relation_type=self.relation_type,
            time_label=time,
            confidence=self.confidence,
            attributes=self.attributes
        )


# ==================== Journey Models (Paper Section 2.1) ====================

class TemporalWalk(BaseModel):
    """
    Alternating sequence of nodes and times (from paper Section 2.1)
    (u₁, t₁, u₂, t₂, ..., uₖ₋₁, tₖ₋₁, uₖ)
    """
    nodes: List[str]  # Node IDs
    time_labels: List[TimeLabel]  # Edge labels (length = len(nodes) - 1)
    
    @property
    def departure_time(self) -> TimeLabel:
        """First time label"""
        return self.time_labels[0] if self.time_labels else None
    
    @property
    def arrival_time(self) -> TimeLabel:
        """Last time label"""
        return self.time_labels[-1] if self.time_labels else None
    
    @property
    def duration(self) -> int:
        """
        Duration (temporal length) from paper Section 2.1:
        duration = tₖ₋₁ - t₁ + 1
        """
        if not self.time_labels:
            return 0
        return self.arrival_time.value - self.departure_time.value + 1
    
    def is_time_respecting(self) -> bool:
        """
        Check if walk is time-respecting (journey)
        From paper: "strictly increasing edge-labels"
        """
        for i in range(len(self.time_labels) - 1):
            if self.time_labels[i] >= self.time_labels[i + 1]:
                return False
        return True
    
    def has_duplicate_nodes(self) -> bool:
        """Check if any node is visited more than once"""
        return len(self.nodes) != len(set(self.nodes))


class Journey(TemporalWalk):
    """
    Time-respecting path (from paper Section 2.1)
    A journey is a temporal walk with:
    1. Strictly increasing time labels
    2. Pairwise distinct nodes
    """
    
    def __init__(self, **data):
        super().__init__(**data)
        # Validate journey properties
        if not self.is_time_respecting():
            raise ValueError("Journey must have strictly increasing time labels")
        if self.has_duplicate_nodes():
            raise ValueError("Journey must have pairwise distinct nodes")


class ForemostJourney(Journey):
    """
    Foremost journey from time t (from paper Section 2.1)
    "A u-v journey J is called foremost from time t ∈ ℕ if it
    departs after time t and its arrival time is minimized"
    """
    from_time: TimeLabel  # The time t from which this is foremost
    
    @classmethod
    def from_journey(cls, journey: Journey, from_time: TimeLabel) -> 'ForemostJourney':
        """Create foremost journey from regular journey"""
        return cls(
            nodes=journey.nodes,
            time_labels=journey.time_labels,
            from_time=from_time
        )


# ==================== Temporal Graph (Paper Section 2) ====================

class TemporalGraph(BaseModel):
    """
    Complete temporal graph D = (V, A)
    Where:
    - V is set of nodes
    - A ⊆ V² × ℕ is set of time-edges
    
    From paper Section 2: "A temporal graph may be viewed as a 
    sequence of static graphs (G₁, G₂, ..., Gλₘₐₓ)"
    """
    nodes: Dict[str, TemporalNode] = Field(default_factory=dict)
    relations: Dict[str, TemporalRelation] = Field(default_factory=dict)
    
    # Temporal properties (from paper Section 2)
    lambda_min: TimeLabel = Field(default=TimeLabel(value=0))  # Minimum label
    lambda_max: TimeLabel = Field(default=TimeLabel(value=0))  # Maximum label
    
    @property
    def age(self) -> int:
        """
        Age (lifetime) of temporal graph (from paper Section 2):
        α(λ) = λₘₐₓ - λₘᵢₙ + 1
        """
        return self.lambda_max.value - self.lambda_min.value + 1
    
    def add_node(self, node: TemporalNode) -> None:
        """Add node to graph"""
        self.nodes[node.id] = node
        
        # Update temporal bounds
        if node.created_at > self.lambda_max:
            self.lambda_max = node.created_at
    
    def add_relation(self, relation: TemporalRelation) -> None:
        """Add temporal relation to graph"""
        self.relations[relation.id] = relation
        
        # Update temporal bounds
        if relation.interval.valid_from > self.lambda_max:
            self.lambda_max = relation.interval.valid_from
        
        if relation.interval.valid_to and relation.interval.valid_to > self.lambda_max:
            self.lambda_max = relation.interval.valid_to
    
    def get_instance_at(self, time: TimeLabel) -> List[TemporalEdge]:
        """
        Get the tth instance of temporal graph (from paper Section 2):
        D(t) = (V, A(t)) where A(t) = {e : (e,t) ∈ A}
        
        Returns all edges available at time t
        """
        edges = []
        for relation in self.relations.values():
            edge = relation.to_temporal_edge(time)
            if edge:
                edges.append(edge)
        return edges
    
    def get_neighbors_at_time(
        self, 
        node_id: str, 
        time: TimeLabel,
        direction: str = "outgoing"
    ) -> List[Tuple[str, TemporalEdge]]:
        """
        Get neighbors of node at specific time
        
        Args:
            node_id: Source node
            time: Time to check
            direction: "outgoing", "incoming", or "both"
        
        Returns:
            List of (neighbor_id, edge) tuples
        """
        neighbors = []
        
        for relation in self.relations.values():
            if not relation.is_valid_at(time):
                continue
            
            if direction in ("outgoing", "both") and relation.source_id == node_id:
                edge = relation.to_temporal_edge(time)
                if edge:
                    neighbors.append((relation.target_id, edge))
            
            if direction in ("incoming", "both") and relation.target_id == node_id:
                edge = relation.to_temporal_edge(time)
                if edge:
                    neighbors.append((relation.source_id, edge))
        
        return neighbors
    
    def get_relations_in_interval(
        self,
        start: TimeLabel,
        end: Optional[TimeLabel] = None
    ) -> List[TemporalRelation]:
        """
        Get all relations that were valid at any point in interval [start, end]
        """
        query_interval = TemporalInterval(valid_from=start, valid_to=end)
        
        return [
            rel for rel in self.relations.values()
            if rel.interval.overlaps(query_interval)
        ]


# ==================== Disjoint Journey Models (Paper Section 3) ====================

class JourneyDisjointness(str, Enum):
    """Types of journey disjointness (from paper Section 3)"""
    NODE_DISJOINT = "node_disjoint"  # No shared nodes (except s and z)
    EDGE_DISJOINT = "edge_disjoint"  # No shared edges
    OUT_DISJOINT = "out_disjoint"    # Never leave same node at same time


class OutDisjointJourneys(BaseModel):
    """
    Set of out-disjoint journeys (from paper Section 3)
    "Two journeys are called out-disjoint if they never leave
    from the same node at the same time"
    
    This is key for Menger's temporal analogue
    """
    journeys: List[Journey]
    source_id: str
    target_id: str
    
    def are_out_disjoint(self) -> bool:
        """
        Verify all journeys are pairwise out-disjoint
        """
        # Track (node, departure_time) pairs
        departures: Set[Tuple[str, int]] = set()
        
        for journey in self.journeys:
            for i, node_id in enumerate(journey.nodes[:-1]):  # All but last node
                time = journey.time_labels[i].value
                
                if (node_id, time) in departures:
                    return False
                departures.add((node_id, time))
        
        return True


# ==================== Temporal Distance (Paper Section 2.1) ====================

class TemporalDistance(BaseModel):
    """
    Temporal distance from (u,t) to v (from paper Section 2.1)
    "The temporal distance from a node u at time t to a node v is
    defined as the duration of a foremost u-v journey from time t"
    """
    from_node: str
    to_node: str
    from_time: TimeLabel
    distance: int  # Duration of foremost journey
    journey: Optional[ForemostJourney] = None


class TemporalDiameter(BaseModel):
    """
    Temporal (dynamic) diameter (from paper Section 2.1)
    "A temporal graph D has temporal diameter d if d is the minimum
    integer for which the temporal distance from every time-node (u,t)
    to every node v is at most d"
    """
    diameter: int
    worst_case_pair: Optional[Tuple[str, TimeLabel, str]] = None


# ==================== Export ====================

__all__ = [
    'TimeLabel',
    'TemporalNode',
    'TemporalEdge',
    'TemporalInterval',
    'TemporalRelation',
    'TemporalWalk',
    'Journey',
    'ForemostJourney',
    'TemporalGraph',
    'JourneyDisjointness',
    'OutDisjointJourneys',
    'TemporalDistance',
    'TemporalDiameter',
]