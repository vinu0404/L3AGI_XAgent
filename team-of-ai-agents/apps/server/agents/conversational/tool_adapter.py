"""Tool Adapter for converting between L3AGI and XAgent tool formats"""

from typing import Any, Dict, List, Type
from pydantic import BaseModel
from langchain.tools import BaseTool

class ToolAdapter:
    """Adapter to convert L3AGI tools to XAgent format"""
    
    def __init__(self, l3agi_tool: BaseTool):
        self.original_tool = l3agi_tool
        self.tool_name = l3agi_tool.name
        self.tool_description = l3agi_tool.description
        self.args_schema = l3agi_tool.args_schema
        
    async def tool_runner(self, **params: Dict[str, Any]) -> Any:
        """Execute the tool with converted parameters"""
        # Convert XAgent params to L3AGI format
        converted_params = self._convert_params(params)
        # Execute original tool
        result = await self._execute_original_tool(converted_params)
        # Convert result back to XAgent format
        return self._convert_result(result)
    
    def _convert_params(self, xagent_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert XAgent parameters to L3AGI format"""
        if not self.args_schema:
            return xagent_params
            
        converted = {}
        for field_name, field_info in self.args_schema.__fields__.items():
            if field_name in xagent_params:
                # Convert value to expected type
                converted[field_name] = field_info.type_(xagent_params[field_name])
        return converted
    
    async def _execute_original_tool(self, params: Dict[str, Any]) -> Any:
        """Execute the original L3AGI tool"""
        if hasattr(self.original_tool.run, "__call__"):
            if hasattr(self.original_tool.run, "__await__"):
                return await self.original_tool.run(**params)
            return self.original_tool.run(**params)
        raise ValueError(f"Tool {self.tool_name} has no runnable implementation")
    
    def _convert_result(self, result: Any) -> Dict[str, Any]:
        """Convert L3AGI tool result to XAgent format"""
        if isinstance(result, (str, int, float, bool)):
            return {"output": str(result)}
        elif isinstance(result, dict):
            return result
        elif isinstance(result, BaseModel):
            return result.dict()
        return {"output": str(result)}

class ToolRegistry:
    """Registry for managing tool adapters"""
    
    def __init__(self):
        self.tools: Dict[str, ToolAdapter] = {}
        
    def register_tool(self, l3agi_tool: BaseTool) -> None:
        """Register a L3AGI tool and create its adapter"""
        adapted_tool = ToolAdapter(l3agi_tool)
        self.tools[adapted_tool.tool_name] = adapted_tool
        
    def get_tool(self, tool_name: str) -> ToolAdapter:
        """Get an adapted tool by name"""
        return self.tools.get(tool_name)
        
    def list_tools(self) -> List[Dict[str, str]]:
        """List all registered tools in XAgent format"""
        return [
            {
                "name": tool.tool_name,
                "description": tool.tool_description,
                "parameters": self._get_tool_parameters(tool)
            }
            for tool in self.tools.values()
        ]
    
    def _get_tool_parameters(self, tool: ToolAdapter) -> Dict[str, Dict[str, str]]:
        """Get tool parameters in XAgent format"""
        if not tool.args_schema:
            return {}
            
        return {
            name: {
                "type": str(field.type_),
                "description": field.field_info.description or ""
            }
            for name, field in tool.args_schema.__fields__.items()
        }