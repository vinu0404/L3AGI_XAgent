from ..conversational.xagent_integration import XAgentWrapper

def initialize_executor(agent_with_configs, tools, settings):
    """Initialize agent executor with XAgent"""
    
    # Create XAgent wrapper
    xagent = XAgentWrapper(settings)
    
    # Register tools
    for tool in tools:
        xagent.tool_registry.register_tool(tool)
    
    return xagent