"""
Temporal Graph Algorithms
Implements key algorithms from the paper for memory system

Key algorithms implemented:
1. Foremost Journey Computation (Section 2.1) - O(nα³ + |λ|)
2. Out-Disjoint Journey Finding (Section 3) - For Menger's analogue
3. Temporal Diameter Calculation (Section 2.1)
4. Conflict Detection via Journey Analysis
"""

from typing import List, Optional, Dict, Set, Tuple
from collections import defaultdict, deque
import heapq

from temporal_memory_models import (
    TimeLabel, TemporalNode, TemporalEdge, TemporalGraph,
    Journey, ForemostJourney, TemporalDistance, OutDisjointJourneys,
    TemporalInterval
)


class ForemostJourneyAlgorithm:
    r"""
    Computes foremost journeys (Section 2.1 of paper)
    
    From paper:
    "There is an algorithm that, given a source node s \in V and a time t_start,
    computes for all w \in V \ {s} a foremost s-w journey from time t_start.
    Running time: O(nα³(λ) + |λ|)"
    
    This is essentially temporal BFS with path length replaced by arrival time
    """
    
    def __init__(self, graph: TemporalGraph):
        self.graph = graph
    
    def compute_foremost_journeys(
        self,
        source_id: str,
        start_time: TimeLabel
    ) -> Dict[str, ForemostJourney]:
        """
        Compute foremost journey from source to all reachable nodes
        
        Algorithm (from paper Section 2.1):
        1. Start from source at time start_time
        2. For each time t, pick all already-reached nodes
        3. Inspect all edges incident to that node at time t
        4. If edge leads to unreached node, mark it as reached
        5. This greedy approach finds foremost journeys
        
        Returns:
            Dict mapping node_id -> ForemostJourney
        """
        # Track earliest arrival time at each node
        arrival_times: Dict[str, TimeLabel] = {source_id: start_time}
        
        # Track the journey to each node
        journeys: Dict[str, List[Tuple[str, TimeLabel]]] = {
            source_id: []  # Empty journey to source
        }
        
        # Process times in order (BFS by time)
        for t in range(start_time.value, self.graph.lambda_max.value + 1):
            current_time = TimeLabel(value=t)
            
            # Get all nodes reached so far
            reached_nodes = list(arrival_times.keys())
            
            for node_id in reached_nodes:
                # Can only use this node if we've already arrived
                if arrival_times[node_id] > current_time:
                    continue
                
                # Check all outgoing edges at this time
                neighbors = self.graph.get_neighbors_at_time(
                    node_id, 
                    current_time,
                    direction="outgoing"
                )
                
                for neighbor_id, edge in neighbors:
                    # If not yet reached, or we found earlier arrival
                    if (neighbor_id not in arrival_times or 
                        current_time < arrival_times[neighbor_id]):
                        
                        # Update arrival time
                        arrival_times[neighbor_id] = current_time
                        
                        # Build journey: parent's journey + this edge
                        parent_journey = journeys[node_id]
                        journeys[neighbor_id] = parent_journey + [
                            (node_id, current_time)
                        ]
        
        # Convert to ForemostJourney objects
        result = {}
        for node_id, journey_edges in journeys.items():
            if node_id == source_id:
                continue  # Skip source itself
            
            # Build journey
            nodes = [source_id]
            time_labels = []
            
            for from_node, time in journey_edges:
                if from_node not in nodes:
                    nodes.append(from_node)
                time_labels.append(time)
            
            # Add final node
            if node_id not in nodes:
                nodes.append(node_id)
            
            if time_labels:  # Only if there's actually a journey
                result[node_id] = ForemostJourney(
                    nodes=nodes,
                    time_labels=time_labels,
                    from_time=start_time
                )
        
        return result
    
    def compute_temporal_distance(
        self,
        from_node: str,
        to_node: str,
        from_time: TimeLabel
    ) -> Optional[TemporalDistance]:
        """
        Compute temporal distance between two nodes
        
        From paper Section 2.1:
        "The temporal distance from a node u at time t to a node v is
        defined as the duration of a foremost u-v journey from time t"
        """
        journeys = self.compute_foremost_journeys(from_node, from_time)
        
        if to_node not in journeys:
            return None  # Not reachable
        
        journey = journeys[to_node]
        
        return TemporalDistance(
            from_node=from_node,
            to_node=to_node,
            from_time=from_time,
            distance=journey.duration,
            journey=journey
        )


class TemporalDiameterCalculator:
    """
    Calculate temporal diameter (Section 2.1)
    
    From paper:
    "A temporal graph D = (V,A) has temporal diameter d if d is the
    minimum integer for which the temporal distance from every time-node
    (u,t) ∈ V × {0,1,...,α-d} to every node v ∈ V is at most d"
    """
    
    def __init__(self, graph: TemporalGraph):
        self.graph = graph
        self.foremost_algo = ForemostJourneyAlgorithm(graph)
    
    def compute_diameter(self) -> int:
        """
        Compute temporal diameter
        
        This is expensive: O(n² × α) where n = nodes, α = age
        For production: cache or approximate
        """
        max_distance = 0
        worst_case = None
        
        # For every time-node (u, t)
        for node_id in self.graph.nodes.keys():
            # Only check times up to α - d (we'll compute d iteratively)
            for t in range(self.graph.lambda_min.value, 
                          self.graph.lambda_max.value + 1):
                time = TimeLabel(value=t)
                
                # Compute foremost journeys from this time-node
                journeys = self.foremost_algo.compute_foremost_journeys(
                    node_id, 
                    time
                )
                
                # Find maximum distance
                for target_id, journey in journeys.items():
                    if journey.duration > max_distance:
                        max_distance = journey.duration
                        worst_case = (node_id, time, target_id)
        
        return max_distance


class OutDisjointJourneyFinder:
    """
    Find out-disjoint journeys (Section 3)
    
    From paper:
    "Two journeys are called out-disjoint if they never leave
    from the same node at the same time"
    
    This is used for Menger's temporal analogue
    """
    
    def __init__(self, graph: TemporalGraph):
        self.graph = graph
    
    def find_max_out_disjoint_journeys(
        self,
        source_id: str,
        target_id: str,
        from_time: TimeLabel
    ) -> OutDisjointJourneys:
        """
        Find maximum number of out-disjoint journeys from source to target
        
        From paper Section 3:
        "The maximum number of out-disjoint journeys from s to z is equal
        to the minimum number of node departure times needed to separate s from z"
        
        Algorithm:
        1. Find all possible journeys from s to z
        2. Greedily select journeys that are out-disjoint
        3. Track (node, time) departures to ensure disjointness
        """
        # Find all journeys (simplified: using repeated foremost journey)
        all_journeys = self._find_all_journeys(source_id, target_id, from_time)
        
        # Greedily select out-disjoint journeys
        selected_journeys = []
        used_departures: Set[Tuple[str, int]] = set()
        
        for journey in all_journeys:
            # Check if this journey conflicts with any selected journey
            conflicts = False
            
            for i, node_id in enumerate(journey.nodes[:-1]):
                time = journey.time_labels[i].value
                
                if (node_id, time) in used_departures:
                    conflicts = True
                    break
            
            if not conflicts:
                # Add this journey
                selected_journeys.append(journey)
                
                # Mark its departures as used
                for i, node_id in enumerate(journey.nodes[:-1]):
                    time = journey.time_labels[i].value
                    used_departures.add((node_id, time))
        
        return OutDisjointJourneys(
            journeys=selected_journeys,
            source_id=source_id,
            target_id=target_id
        )
    
    def _find_all_journeys(
        self,
        source_id: str,
        target_id: str,
        from_time: TimeLabel,
        max_journeys: int = 10
    ) -> List[Journey]:
        """
        Find multiple journeys (simplified version)
        In production: use k-shortest paths algorithm
        """
        journeys = []
        
        # Try different starting times to find diverse journeys
        for t in range(from_time.value, min(from_time.value + 20, 
                                            self.graph.lambda_max.value + 1)):
            time = TimeLabel(value=t)
            
            algo = ForemostJourneyAlgorithm(self.graph)
            foremost = algo.compute_foremost_journeys(source_id, time)
            
            if target_id in foremost:
                journey = foremost[target_id]
                
                # Check if this is different from existing journeys
                is_new = True
                for existing in journeys:
                    if self._journeys_similar(journey, existing):
                        is_new = False
                        break
                
                if is_new:
                    journeys.append(journey)
                
                if len(journeys) >= max_journeys:
                    break
        
        return journeys
    
    def _journeys_similar(self, j1: Journey, j2: Journey) -> bool:
        """Check if two journeys are too similar"""
        # Simple heuristic: same nodes = similar
        return set(j1.nodes) == set(j2.nodes)


class TemporalConflictDetector:
    """
    Detect conflicts using temporal graph analysis
    
    Key insight: If two facts create conflicting journeys,
    they cannot both be true at the same time
    """
    
    def __init__(self, graph: TemporalGraph):
        self.graph = graph
    
    def detect_conflicts(
        self,
        new_relation_type: str,
        source_id: str,
        target_id: str,
        time_interval: 'TemporalInterval'
    ) -> List[str]:
        """
        Detect conflicting relations
        
        A conflict exists if:
        1. Same source node
        2. Same relation type (e.g., both "lives_in")
        3. Different targets
        4. Overlapping time intervals
        
        Returns:
            List of conflicting relation IDs
        """
        conflicts = []
        
        # Define mutually exclusive relations
        exclusive_relations = {
            "lives_in",    # Can't live in two places simultaneously
            "works_at",    # Typically one primary job
            "is_married_to",
            "is_a"         # Type relations
        }
        
        if new_relation_type not in exclusive_relations:
            return []  # This relation type allows multiple targets
        
        # Check existing relations
        for rel_id, relation in self.graph.relations.items():
            if (relation.source_id == source_id and
                relation.relation_type == new_relation_type and
                relation.target_id != target_id and
                relation.interval.overlaps(time_interval)):
                
                conflicts.append(rel_id)
        
        return conflicts
    
    def would_create_cycle(
        self,
        source_id: str,
        target_id: str,
        current_time: TimeLabel
    ) -> bool:
        """
        Check if adding edge would create temporal cycle
        
        Uses foremost journey algorithm to check if path already exists
        from target back to source
        """
        algo = ForemostJourneyAlgorithm(self.graph)
        journeys = algo.compute_foremost_journeys(target_id, current_time)
        
        return source_id in journeys


class TemporalReasoningEngine:
    """
    High-level reasoning using temporal graph algorithms
    """
    
    def __init__(self, graph: TemporalGraph):
        self.graph = graph
        self.foremost_algo = ForemostJourneyAlgorithm(graph)
        self.conflict_detector = TemporalConflictDetector(graph)
    
    def get_fact_at_time(
        self,
        entity: str,
        relation: str,
        time: TimeLabel
    ) -> List[str]:
        """
        Get all objects related to entity via relation at specific time
        
        Example:
            get_fact_at_time("User", "lives_in", TimeLabel(100))
            -> ["San Francisco"]
        """
        results = []
        
        for relation_obj in self.graph.relations.values():
            if (relation_obj.source_id == entity and
                relation_obj.relation_type == relation and
                relation_obj.is_valid_at(time)):
                
                results.append(relation_obj.target_id)
        
        return results
    
    def get_fact_history(
        self,
        entity: str,
        relation: str
    ) -> List[Tuple[TimeLabel, Optional[TimeLabel], str]]:
        """
        Get complete history of a relation
        
        Returns:
            List of (start_time, end_time, value) tuples
        """
        history = []
        
        for relation_obj in self.graph.relations.values():
            if (relation_obj.source_id == entity and
                relation_obj.relation_type == relation):
                
                history.append((
                    relation_obj.interval.valid_from,
                    relation_obj.interval.valid_to,
                    relation_obj.target_id
                ))
        
        # Sort by start time
        history.sort(key=lambda x: x[0].value)
        
        return history
    
    def infer_at_time(
        self,
        entity: str,
        max_hops: int,
        time: TimeLabel
    ) -> Dict[str, List[Journey]]:
        """
        Infer all facts reachable within max_hops at given time
        
        Uses foremost journey algorithm to find all reachable nodes
        """
        journeys = self.foremost_algo.compute_foremost_journeys(
            entity,
            time
        )
        
        # Filter by max hops
        filtered = {}
        for target, journey in journeys.items():
            if len(journey.nodes) - 1 <= max_hops:  # -1 because nodes include source
                filtered[target] = [journey]
        
        return filtered
    
    def explain_inference(
        self,
        entity: str,
        inferred_fact: str,
        time: TimeLabel
    ) -> Optional[Journey]:
        """
        Explain how we inferred a fact by showing the journey
        
        Returns the journey (reasoning path) from entity to inferred_fact
        """
        journeys = self.foremost_algo.compute_foremost_journeys(
            entity,
            time
        )
        
        return journeys.get(inferred_fact)


# ==================== Export ====================

__all__ = [
    'ForemostJourneyAlgorithm',
    'TemporalDiameterCalculator',
    'OutDisjointJourneyFinder',
    'TemporalConflictDetector',
    'TemporalReasoningEngine',
]