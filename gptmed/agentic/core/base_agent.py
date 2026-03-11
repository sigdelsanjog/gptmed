"""
Base Agent class - Abstract interface for all agents.
Defines the contract that all agents must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time
from datetime import datetime

from .result import AgentResult, ExecutionStatus
from .logger import AgentLogger
from .exceptions import AgentExecutionError


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Design Principles:
    1. Single Responsibility: Each agent does ONE thing well
    2. Fault Isolation: Agent failures don't crash the system
    3. Observable: All operations are logged and tracked
    4. Chainable: Output feeds to next agent's input
    5. Configurable: Behavior controlled via config, not code
    
    Architecture Notes:
    - Agents are stateless processing units
    - State is stored in memory/context, not in agent
    - Agents communicate via AgentResult objects
    - Each agent has a unique name for identification
    - Confidence scores indicate processing quality
    """
    
    def __init__(self, name: str, description: str, enabled: bool = True):
        """
        Initialize base agent.
        
        Args:
            name: Unique identifier (e.g., "PrescriptionAnalyzer")
            description: Human-readable description of agent's purpose
            enabled: Whether this agent should execute
        """
        self.name = name
        self.description = description
        self.enabled = enabled
        self.logger = AgentLogger
        
        # Validate name (must be alphanumeric + underscore)
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid agent name: {name}")
        
        self.logger.info(f"Initialized agent: {self.name} - {self.description}")
    
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input and return structured output.
        
        This method MUST be implemented by each subclass.
        
        Args:
            input_data: Input data (can be dict, string, object, etc.)
        
        Returns:
            Output as dictionary
        
        Implementation Notes:
        - Should handle None/empty inputs gracefully
        - Should not raise exceptions (use try-except internally)
        - Should return consistent structure
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate that input conforms to agent's requirements.
        
        Args:
            input_data: Input to validate
        
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate that output conforms to expected structure.
        
        Args:
            output: Output to validate
        
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Execute agent with error handling and result wrapping.
        
        This template method:
        1. Validates input
        2. Measures execution time
        3. Calls process() implementation
        4. Validates output
        5. Wraps result with metadata
        6. Handles exceptions gracefully
        
        Args:
            input_data: Input data to process
            **kwargs: Additional arguments (logged in metadata)
        
        Returns:
            AgentResult with execution details
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not self.validate_input(input_data):
                return AgentResult(
                    agent_name=self.name,
                    status=ExecutionStatus.FAILED,
                    result=None,
                    error_message="Input validation failed",
                    error_type="ValidationError",
                    metadata={"input_preview": str(input_data)[:100]}
                )
            
            # Check if agent is enabled
            if not self.enabled:
                return AgentResult(
                    agent_name=self.name,
                    status=ExecutionStatus.SKIPPED,
                    result=None,
                    metadata={"reason": "Agent disabled"}
                )
            
            # Process
            self.logger.debug(f"Executing {self.name}")
            output = self.process(input_data)
            
            # Validate output
            if not self.validate_output(output):
                return AgentResult(
                    agent_name=self.name,
                    status=ExecutionStatus.FAILED,
                    result=None,
                    error_message="Output validation failed",
                    error_type="ValidationError",
                    metadata={"output_keys": list(output.keys()) if isinstance(output, dict) else "N/A"}
                )
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Get confidence score
            confidence = output.pop("_confidence", 0.8)
            
            # Build result
            result = AgentResult(
                agent_name=self.name,
                status=ExecutionStatus.COMPLETED,
                result=output,
                confidence_score=confidence,
                execution_time_ms=execution_time_ms,
                timestamp=datetime.now().isoformat(),
                metadata={"kwargs": kwargs}
            )
            
            self.logger.debug(f"✅ {self.name} completed in {execution_time_ms:.1f}ms")
            return result
        
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            self.logger.error(f"❌ {self.name} failed: {str(e)}", exc_info=True)
            
            return AgentResult(
                agent_name=self.name,
                status=ExecutionStatus.FAILED,
                result=None,
                error_message=str(e),
                error_type=type(e).__name__,
                execution_time_ms=execution_time_ms,
                metadata={"input_preview": str(input_data)[:100] if input_data else "None"}
            )
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get agent configuration (can be overridden by subclasses).
        
        Returns:
            Configuration dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "type": self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        """String representation."""
        status = "✅ enabled" if self.enabled else "⊘ disabled"
        return f"<{self.__class__.__name__}: {self.name} ({status})>"
