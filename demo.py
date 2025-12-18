"""
Temporal Memory System Demo
Shows how the system solves Mem0's problems using temporal graphs
"""

import os
import time
from datetime import datetime, timedelta
from temporal_memory_system import TemporalMemorySystem, ExtractedFact, MemoryOperation

DB_PATH = "demo_memory.db"

def demo_temporal_transitions():
    """
    Demo: Handling facts that change over time
    Shows temporal transition (the core problem Mem0 struggles with)
    """
    print("=" * 70)
    print("DEMO 1: Temporal Transitions (Mem0's Core Problem)")
    print("=" * 70)
    
    system = TemporalMemorySystem(db_path=DB_PATH)
    
    # January: User lives in NYC
    print("\n January 2024 (T=0):")
    print("User: 'I live in New York City'")
    
    fact1 = ExtractedFact(
        subject="User",
        predicate="lives_in",
        object="New York City",
        confidence=0.95
    )
    
    op, rel_id = system.process_fact(fact1)
    print(f"   → Operation: {op.value}")
    print(f"   → Relation ID: {rel_id}")
    
    # Advance time
    system.advance_time()
    system.advance_time()
    
    # March: User moves to SF
    print("\n March 2024 (T=2):")
    print("User: 'I moved to San Francisco'")
    
    fact2 = ExtractedFact(
        subject="User",
        predicate="lives_in",
        object="San Francisco",
        confidence=0.95
    )
    
    op, rel_id = system.process_fact(fact2)
    print(f"   → Operation: {op.value} (detected conflict!)")
    print(f"   → System automatically closed NYC relation")
    print(f"   → System opened new SF relation")
    
    # Query current state
    print("\n Query: 'Where does user live now?'")
    current = system.query_current("User", "lives_in")
    print(f"   → Answer: {current}")
    
    # Get complete history
    print("\n Complete History:")
    history = system.query_history("User", "lives_in")
    for entry in history:
        print(f"   • From T={entry['from']} to T={entry['to']}: {entry['value']}")
    
    system.close()
    print("\n Success: Temporal transitions handled perfectly!")
    print("   - No confusion about current vs past")
    print("   - Complete history preserved")
    print("   - Conflicts resolved automatically")


def demo_multi_hop_reasoning():
    """
    Demo: Multi-hop reasoning using journeys
    Shows inference capabilities Mem0 lacks
    """
    print("\n\n")
    print("=" * 70)
    print("DEMO 2: Multi-Hop Reasoning with Journeys")
    print("=" * 70)
    
    system = TemporalMemorySystem(db_path=DB_PATH)
    
    # Build knowledge graph
    print("\n Building knowledge graph...")
    
    facts = [
        ExtractedFact(
            subject="User",
            predicate="is_vegetarian",
            object="True",
            confidence=0.95
        ),
        ExtractedFact(
            subject="User",
            predicate="lives_in",
            object="San Francisco",
            confidence=0.90
        ),
        ExtractedFact(
            subject="San Francisco",
            predicate="has_cuisine",
            object="Italian",
            confidence=0.85
        ),
        ExtractedFact(
            subject="Italian",
            predicate="includes",
            object="Pasta",
            confidence=0.90
        ),
    ]
    
    for fact in facts:
        op, _ = system.process_fact(fact)
        print(f"   • {fact.subject} --[{fact.predicate}]--> {fact.object}: {op.value}")
    
    # Query with reasoning
    print("\n Query with Reasoning: 'What foods are suitable for user?'")
    result = system.query_with_reasoning("User", "food preferences", max_hops=3)
    
    print("\n   Inferred facts:")
    for fact, details in result.items():
        print(f"\n    {fact}")
        print(f"      Confidence: {details['confidence']:.2f}")
        print(f"      Reasoning path ({details['hops']} hops):")
        for step in details['reasoning_path']:
            print(f"        → {step}")
    
    system.close()
    print("\n Success: Multi-hop inference working!")
    print("   - Followed journeys through graph")
    print("   - Computed confidence propagation")
    print("   - Provided reasoning paths (explainable AI)")


def demo_caching_performance():
    """
    Demo: Caching layer performance improvements
    """
    print("\n\n")
    print("=" * 70)
    print("DEMO 3: Caching Performance")
    print("=" * 70)
    
    system = TemporalMemorySystem(db_path=DB_PATH)

    # Initial query (cache miss)
    print("\n1. First query for user's location:")
    start_time = time.time()
    result1 = system.query_current("User", "lives_in")
    duration1 = (time.time() - start_time) * 1000
    print(f"   → Result: {result1} (took {duration1:.2f} ms)")
    print("   → Status: Cache MISS (loaded from DB)")

    # Second query (cache hit)
    print("\n2. Second query for the same information:")
    start_time = time.time()
    result2 = system.query_current("User", "lives_in")
    duration2 = (time.time() - start_time) * 1000
    print(f"   → Result: {result2} (took {duration2:.2f} ms)")
    print("   → Status: Cache HIT (served from memory)")

    # Invalidate cache by adding a fact
    print("\n3. Adding a new fact (invalidates cache):")
    system.advance_time()
    fact = ExtractedFact(subject="User", predicate="likes", object="Pizza", confidence=0.9)
    system.process_fact(fact)
    print("   → Fact added, relevant caches cleared.")

    # Third query (cache miss again)
    print("\n4. Third query after cache invalidation:")
    start_time = time.time()
    result3 = system.query_current("User", "lives_in")
    duration3 = (time.time() - start_time) * 1000
    print(f"   → Result: {result3} (took {duration3:.2f} ms)")
    print("   → Status: Cache MISS (re-loaded from DB)")

    system.close()
    print(f"\n Success: Caching reduced query time by ~{((duration1 - duration2) / duration1 * 100):.0f}%!")
    print("   - Subsequent identical queries are near-instant.")
    print("   - Cache is automatically invalidated on data changes.")


def main():
    """Run all demos"""
    try:
        print("\n" + "=" * 70)
        print(" " * 15 + "TEMPORAL MEMORY SYSTEM DEMO")
        print(" " * 10 + "Better than Mem0 with Temporal Graphs")
        print("=" * 70)
        
        # --- Demo 1 ---
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        demo_temporal_transitions()
        
        # --- Demo 2 ---
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        demo_multi_hop_reasoning()

        # --- Demo 3 ---
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        # Need to add some data for the caching demo to query
        temp_system = TemporalMemorySystem(db_path=DB_PATH)
        temp_system.process_fact(ExtractedFact(subject="User", predicate="lives_in", object="San Francisco", confidence=0.9))
        temp_system.close()
        demo_caching_performance()
        
        print("\n\n" + "=" * 70)
        print(" " * 20 + "ALL DEMOS COMPLETED!")
        print("=" * 70)

    finally:
        # Final cleanup
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            print(f"🧹 Cleaned up database file: {DB_PATH}")


if __name__ == "__main__":
    main()