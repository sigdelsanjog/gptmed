"""
Custom exceptions for agentic framework.
Enables fine-grained error handling and debugging.
"""


class AgenticFrameworkError(Exception):
    """Base exception for all agentic framework errors."""
    pass


class AgentExecutionError(AgenticFrameworkError):
    """Raised when an agent fails during execution."""
    pass


class AgentValidationError(AgenticFrameworkError):
    """Raised when agent output validation fails."""
    pass


class AgentNotFoundError(AgenticFrameworkError):
    """Raised when requested agent is not registered."""
    pass


class OrchestratorError(AgenticFrameworkError):
    """Raised when orchestrator encounters issues."""
    pass


class ToolExecutionError(AgenticFrameworkError):
    """Raised when a tool fails to execute."""
    pass


class ConfigurationError(AgenticFrameworkError):
    """Raised when configuration is invalid."""
    pass


class MemoryError(AgenticFrameworkError):
    """Raised when memory operations fail."""
    pass
