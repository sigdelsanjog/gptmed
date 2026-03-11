"""
Agent Orchestrator - Coordinates agent execution and workflows.
Central controller for multi-agent pipelines.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import time
from datetime import datetime

from .base_agent import BaseAgent
from .registry import AgentRegistry
from .result import AgentResult, ExecutionStatus
from .logger import AgentLogger
from .exceptions import OrchestratorError


@dataclass
class WorkflowStep:
    """Definition of a single step in a workflow."""
    agent_name: str
    depends_on: List[str] = None  # Agent names this step depends on
    pass_input: bool = True  # Whether to pass previous result as input
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


class AgentOrchestrator:
    """
    Orchestrator for multi-agent workflows.
    
    Responsibilities:
    1. Execute sequences of agents (pipelines)
    2. Handle agent dependencies
    3. Pass results between agents
    4. Track execution flow and timing
    5. Handle errors and retries
    6. Provide execution visibility
    
    Architecture Decisions:
    - Sequential execution by default (simple, predictable)
    - Parallel execution support for independent agents
    - Result chaining: output of agent N → input of agent N+1
    - Error handling: Fail fast or continue (configurable)
    - Memory: Shared context across agents in workflow
    """
    
    def __init__(self, registry: AgentRegistry = None):
        """
        Initialize orchestrator.
        
        Args:
            registry: Agent registry (default: singleton)
        """
        self.registry = registry or AgentRegistry()
        self.logger = AgentLogger
        self.execution_history: List[AgentResult] = []
    
    def execute_agent(self, agent_name: str, input_data: Any = None, **kwargs) -> AgentResult:
        """
        Execute a single agent.
        
        Args:
            agent_name: Name of agent to execute
            input_data: Input data for agent
            **kwargs: Additional arguments
        
        Returns:
            AgentResult
        """
        try:
            agent = self.registry.get(agent_name)
            result = agent.execute(input_data, **kwargs)
            self.execution_history.append(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Failed to execute agent '{agent_name}': {str(e)}")
            result = AgentResult(
                agent_name=agent_name,
                status=ExecutionStatus.FAILED,
                result=None,
                error_message=str(e),
                error_type=type(e).__name__
            )
            self.execution_history.append(result)
            return result
    
    def execute_workflow(self, steps: List[WorkflowStep], 
                        initial_input: Any = None,
                        fail_fast: bool = True,
                        timeout_sec: float = 300) -> Dict[str, AgentResult]:
        """
        Execute a workflow of sequential agent steps.
        
        Args:
            steps: List of WorkflowStep definitions
            initial_input: Input for first agent
            fail_fast: If True, stop on first failure
            timeout_sec: Maximum total execution time
        
        Returns:
            Dictionary mapping agent_name → AgentResult
        """
        workflow_start = time.time()
        results = {}
        current_input = initial_input
        
        self.logger.info(f"Starting workflow with {len(steps)} steps")
        
        for i, step in enumerate(steps):
            # Check timeout
            elapsed = time.time() - workflow_start
            if elapsed > timeout_sec:
                self.logger.warning(f"Workflow timeout after {elapsed:.1f}s")
                break
            
            # Log step
            self.logger.info(f"Step {i+1}/{len(steps)}: Executing {step.agent_name}")
            
            # Get input for this step
            if step.pass_input and current_input is not None:
                step_input = current_input
            else:
                step_input = initial_input
            
            # Execute agent
            result = self.execute_agent(step.agent_name, step_input)
            results[step.agent_name] = result
            
            # Log result
            self.logger.debug(f"  {result.to_summary()}")
            
            # Handle failure
            if result.is_failed():
                self.logger.warning(f"  Agent failed: {result.error_message}")
                if fail_fast:
                    self.logger.error(f"Workflow stopped (fail_fast=True)")
                    break
            else:
                # Pass result to next agent
                current_input = result.result
        
        workflow_time = (time.time() - workflow_start) * 1000
        self.logger.info(f"Workflow completed in {workflow_time:.1f}ms")
        
        return results
    
    def execute_pipeline(self, agent_names: List[str], 
                        initial_input: Any = None,
                        **kwargs) -> Dict[str, AgentResult]:
        """
        Execute a simple linear pipeline of agents.
        
        Shorthand for execute_workflow with auto-generated steps.
        
        Args:
            agent_names: List of agent names in execution order
            initial_input: Input for first agent
            **kwargs: Additional arguments (fail_fast, timeout_sec)
        
        Returns:
            Dictionary mapping agent_name → AgentResult
        """
        steps = [WorkflowStep(agent_name=name) for name in agent_names]
        return self.execute_workflow(steps, initial_input, **kwargs)
    
    def get_execution_summary(self) -> str:
        """Get summary of all executions."""
        if not self.execution_history:
            return "No executions yet"
        
        total = len(self.execution_history)
        successful = len([r for r in self.execution_history if r.is_success()])
        failed = len([r for r in self.execution_history if r.is_failed()])
        total_time = sum(r.execution_time_ms for r in self.execution_history)
        
        summary = f"""
        ╔════════════════════════════════════════╗
        ║  ORCHESTRATOR EXECUTION SUMMARY        ║
        ╚════════════════════════════════════════╝
        
        Total Executions: {total}
        ✅ Successful: {successful}
        ❌ Failed: {failed}
        ⏱️  Total Time: {total_time:.1f}ms
        ⏲️  Average Time: {total_time/total:.1f}ms
        
        ─────────────────────────────────────────
        """
        
        for result in self.execution_history:
            summary += f"\n{result.to_summary()}"
        
        return summary
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
        self.logger.info("Cleared execution history")
    
    def get_workflow_results(self, agent_names: List[str] = None) -> Dict[str, Any]:
        """
        Get results from most recent workflow execution.
        
        Args:
            agent_names: If provided, only return results for these agents
        
        Returns:
            Dictionary of agent results
        """
        # Find most recent executions for each agent
        recent_results = {}
        
        for result in reversed(self.execution_history):
            if agent_names and result.agent_name not in agent_names:
                continue
            
            if result.agent_name not in recent_results:
                recent_results[result.agent_name] = result
        
        return recent_results
