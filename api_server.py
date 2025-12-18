"""
FastAPI Server to expose the Temporal Memory System
Based on the production_guide.md
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from temporal_memory_system import TemporalMemorySystem, ExtractedFact, MemoryOperation

# ==================== Configuration ====================

DB_PATH = "production_memory.db"

# ==================== FastAPI App & Memory System Initialization ====================

app = FastAPI(
    title="Temporal Memory API",
    version="1.0.0",
    description="API for interacting with a persistent, temporal knowledge graph.",
)

# In a real app, you might manage multiple memory systems (e.g., per user)
# For this example, we use a single, global instance.
memory_system = TemporalMemorySystem(db_path=DB_PATH)

@app.on_event("startup")
async def startup_event():
    print("🚀 Temporal Memory API starting up...")
    print(f"💾 Loading memory from database: {DB_PATH}")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Shutting down API and closing database connection...")
    memory_system.close()
    print("👋 Goodbye!")

# ==================== API Models ====================

class FactRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0

class QueryRequest(BaseModel):
    entity: str
    relation: str

class ReasoningRequest(BaseModel):
    entity: str
    query: str
    max_hops: int = 2

class FactResponse(BaseModel):
    operation: str
    relation_id: Optional[str]
    message: str

# ==================== API Endpoints ====================

@app.get("/", tags=["General"])
async def read_root():
    """Root endpoint with basic system status."""
    return {
        "message": "Temporal Memory API is running",
        "stats": memory_system.get_statistics()
    }

@app.post("/facts", response_model=FactResponse, tags=["Facts"])
async def add_fact(fact_request: FactRequest):
    """
    Adds a new fact to the memory system at the current time.
    The system will automatically handle conflicts and temporal transitions.
    """
    try:
        fact = ExtractedFact(**fact_request.dict())
        op, rel_id = memory_system.process_fact(fact)
        
        # Advance time after each fact to simulate progression
        memory_system.advance_time()

        return FactResponse(
            operation=op.value,
            relation_id=rel_id,
            message=f"Fact processed with operation: {op.value}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/current", tags=["Querying"])
async def query_current(request: QueryRequest):
    """
    Queries the current value(s) of a specific relation for an entity.
    """
    results = memory_system.query_current(request.entity, request.relation)
    return {"entity": request.entity, "relation": request.relation, "current_values": results}

@app.post("/query/history", tags=["Querying"])
async def query_history(request: QueryRequest):
    """

    Retrieves the complete temporal history of a relation for an entity.
    """
    history = memory_system.query_history(request.entity, request.relation)
    return {"entity": request.entity, "relation": request.relation, "history": history}

@app.post("/reason", tags=["Reasoning"])
async def reason_with_inference(request: ReasoningRequest):
    """
    Performs multi-hop reasoning to infer new facts from the knowledge graph.
    """
    results = memory_system.query_with_reasoning(
        request.entity, request.query, request.max_hops
    )
    return {"entity": request.entity, "inferred_facts": results}

@app.get("/stats", tags=["General"])
async def get_stats():
    """
    Returns current statistics about the memory system.
    """
    return memory_system.get_statistics()

# ==================== Main ====================

if __name__ == "__main__":
    print("Starting API server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
