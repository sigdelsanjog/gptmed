"""
Agent Registry - Manages registered agents.
Enables dynamic agent discovery and coordination.
"""

from typing import Dict, List, Optional, Type
from .base_agent import BaseAgent
from .logger import AgentLogger
from .exceptions import AgentNotFoundError


class AgentRegistry:
    """
    Central registry for all agents in the framework.
    
    Responsibilities:
    - Track available agents
    - Register/deregister agents dynamically
    - Allow agent lookup by name or type
    - Support enabling/disabling agents
    
    Design Pattern: Singleton (shared across system)
    """
    
    _instance = None
    _agents: Dict[str, BaseAgent] = {}
    _logger = AgentLogger
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, agent: BaseAgent) -> None:
        """
        Register an agent.
        
        Args:
            agent: Agent instance to register
        
        Raises:
            ValueError: If agent name already registered
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' already registered")
        
        self._agents[agent.name] = agent
        self._logger.info(f"Registered agent: {agent.name}")
    
    def get(self, agent_name: str) -> BaseAgent:
        """
        Get agent by name.
        
        Args:
            agent_name: Name of agent to retrieve
        
        Returns:
            Agent instance
        
        Raises:
            AgentNotFoundError: If agent not found
        """
        if agent_name not in self._agents:
            raise AgentNotFoundError(f"Agent '{agent_name}' not found")
        
        return self._agents[agent_name]
    
    def list_agents(self, enabled_only: bool = False) -> List[str]:
        """
        List all registered agents.
        
        Args:
            enabled_only: If True, only return enabled agents
        
        Returns:
            List of agent names
        """
        agents = list(self._agents.keys())
        
        if enabled_only:
            agents = [name for name in agents if self._agents[name].enabled]
        
        return agents
    
    def get_agent_info(self, agent_name: str) -> Dict[str, str]:
        """
        Get agent information.
        
        Args:
            agent_name: Name of agent
        
        Returns:
            Agent info dictionary
        """
        agent = self.get(agent_name)
        return {
            "name": agent.name,
            "description": agent.description,
            "enabled": str(agent.enabled),
            "type": agent.__class__.__name__
        }
    
    def list_agent_info(self, enabled_only: bool = False) -> List[Dict[str, str]]:
        """
        Get information for all agents.
        
        Args:
            enabled_only: If True, only return enabled agents
        
        Returns:
            List of agent info dictionaries
        """
        agents = self.list_agents(enabled_only=enabled_only)
        return [self.get_agent_info(name) for name in agents]
    
    def enable_agent(self, agent_name: str) -> None:
        """Enable an agent."""
        agent = self.get(agent_name)
        agent.enabled = True
        self._logger.info(f"Enabled agent: {agent_name}")
    
    def disable_agent(self, agent_name: str) -> None:
        """Disable an agent."""
        agent = self.get(agent_name)
        agent.enabled = False
        self._logger.info(f"Disabled agent: {agent_name}")
    
    def unregister(self, agent_name: str) -> None:
        """Unregister an agent."""
        if agent_name in self._agents:
            del self._agents[agent_name]
            self._logger.info(f"Unregistered agent: {agent_name}")
    
    def clear(self) -> None:
        """Clear all registered agents (use with caution)."""
        self._agents.clear()
        self._logger.warning("Cleared all registered agents")
    
    def get_registry_status(self) -> Dict:
        """Get status of entire registry."""
        total = len(self._agents)
        enabled = len([a for a in self._agents.values() if a.enabled])
        
        return {
            "total_agents": total,
            "enabled_agents": enabled,
            "disabled_agents": total - enabled,
            "agents": self.list_agent_info()
        }
    
    def __repr__(self) -> str:
        """String representation."""
        agent_names = ", ".join(self.list_agents())
        return f"<AgentRegistry: {len(self._agents)} agents [{agent_names}]>"
