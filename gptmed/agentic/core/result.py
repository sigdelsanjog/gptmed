"""
Result wrapper for agent outputs.
Standardizes communication between agents and orchestrator.
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, List
from enum import Enum
from datetime import datetime
import json


class ExecutionStatus(Enum):
    """Execution status of an agent."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class AgentResult:
    """
    Standard result wrapper for all agent executions.
    
    This enables:
    - Consistent communication between agents
    - Result validation and type checking
    - Metadata tracking (timing, confidence, etc.)
    - Easy serialization (JSON, logging)
    - Result chaining (output of one agent → input of next)
    """
    
    # Core fields
    agent_name: str
    status: ExecutionStatus
    result: Any
    
    # Quality metrics
    confidence_score: float = 0.0  # 0.0 to 1.0
    execution_time_ms: float = 0.0
    
    # Tracking
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_id: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage (for result chaining)
    source_agents: List[str] = field(default_factory=list)
    dependencies: Dict[str, "AgentResult"] = field(default_factory=dict)
    
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status in [ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT]
    
    def get_result(self) -> Any:
        """Get the result if successful, None otherwise."""
        return self.result if self.is_success() else None
    
    def add_metadata(self, key: str, value: Any) -> "AgentResult":
        """Add metadata and return self for chaining."""
        self.metadata[key] = value
        return self
    
    def add_source_agent(self, agent_name: str) -> "AgentResult":
        """Add source agent to lineage."""
        if agent_name not in self.source_agents:
            self.source_agents.append(agent_name)
        return self
    
    def add_dependency(self, agent_name: str, result: "AgentResult") -> "AgentResult":
        """Add dependency for tracing multi-agent workflows."""
        self.dependencies[agent_name] = result
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        data = asdict(self)
        data['status'] = self.status.value
        
        # Handle dependencies in dict conversion
        data['dependencies'] = {
            k: v.to_dict() for k, v in self.dependencies.items()
        }
        
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_summary(self) -> str:
        """Get human-readable summary."""
        status_emoji = {
            ExecutionStatus.COMPLETED: "✅",
            ExecutionStatus.FAILED: "❌",
            ExecutionStatus.TIMEOUT: "⏱️",
            ExecutionStatus.RUNNING: "🔄",
            ExecutionStatus.PENDING: "⏳",
            ExecutionStatus.SKIPPED: "⊘"
        }
        
        emoji = status_emoji.get(self.status, "❓")
        
        summary = f"{emoji} {self.agent_name}: {self.status.value} ({self.execution_time_ms:.1f}ms)"
        
        if self.confidence_score > 0:
            summary += f" [confidence: {self.confidence_score:.2%}]"
        
        if self.error_message:
            summary += f"\n   Error: {self.error_message}"
        
        return summary
    
    def __str__(self) -> str:
        """String representation."""
        return self.to_summary()
