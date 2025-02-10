# agent_registry.py

# A simple registry to map agent IDs (e.g., thread IDs) to agent instances.
agent_registry = {}

def register_agent(agent_id: str, agent_instance):
    """Register an agent instance under a given agent ID."""
    agent_registry[agent_id] = agent_instance

def get_agent(agent_id: str):
    """Retrieve the agent instance for the given agent ID."""
    return agent_registry.get(agent_id)
