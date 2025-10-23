"""Response transformation utilities for XAgent integration"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from XAgent.utils import TaskStatus

@dataclass
class XAgentResponse:
    """XAgent response structure"""
    output: str
    thoughts: List[Dict[str, Any]]
    tools_used: List[Dict[str, Any]]
    status: TaskStatus
    execution_time: Optional[float] = None

class ResponseTransformer:
    """Transform XAgent responses to L3AGI format"""
    
    def transform_response(self, xagent_response: XAgentResponse) -> Dict[str, Any]:
        """Transform XAgent response to L3AGI format"""
        return {
            "message": self._extract_message(xagent_response),
            "status": self._map_status(xagent_response.status),
            "metadata": self._build_metadata(xagent_response)
        }
        
    def _extract_message(self, response: XAgentResponse) -> str:
        """Extract the final message from XAgent's output"""
        # Clean and format the output
        message = response.output.strip()
        
        # Add thought process if available
        if response.thoughts:
            thought_process = "\n\nThought Process:\n"
            for thought in response.thoughts:
                if "thought" in thought:
                    thought_process += f"- {thought['thought']}\n"
            message += thought_process
            
        return message
        
    def _map_status(self, status: TaskStatus) -> str:
        """Map XAgent's TaskStatus to L3AGI's status format"""
        status_map = {
            TaskStatus.COMPLETED: "success",
            TaskStatus.FAILED: "error",
            TaskStatus.IN_PROGRESS: "success",
            TaskStatus.WAITING: "success"
        }
        return status_map.get(status, "error")
        
    def _build_metadata(self, response: XAgentResponse) -> Dict[str, Any]:
        """Build metadata from XAgent response"""
        return {
            "tools_used": [
                tool["name"] for tool in response.tools_used
            ] if response.tools_used else [],
            "execution_time": response.execution_time,
            "thought_process": [
                thought.get("thought", "") 
                for thought in response.thoughts
            ] if response.thoughts else []
        }

class StreamingResponseHandler:
    """Handle streaming responses from XAgent"""
    
    def __init__(self):
        self.buffer: List[Dict[str, Any]] = []
        self.transformer = ResponseTransformer()
        
    async def handle_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an individual chunk of streaming response"""
        if isinstance(chunk, dict):
            # Transform and buffer the chunk
            transformed = self._transform_chunk(chunk)
            self.buffer.append(transformed)
            return transformed
            
        # Handle raw string output
        return {
            "message": str(chunk),
            "status": "success",
            "metadata": {}
        }
        
    def _transform_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a chunk into L3AGI format"""
        return {
            "message": chunk.get("output", ""),
            "status": "success",
            "metadata": {
                "thought": chunk.get("thought", ""),
                "tool": chunk.get("tool_name", ""),
                "is_final": chunk.get("is_final", False)
            }
        }
        
    def get_final_response(self) -> Dict[str, Any]:
        """Combine buffered chunks into final response"""
        if not self.buffer:
            return {
                "message": "",
                "status": "error",
                "metadata": {"error": "No response generated"}
            }
            
        # Combine all messages
        messages = []
        tools_used = set()
        thoughts = []
        
        for chunk in self.buffer:
            if chunk.get("message"):
                messages.append(chunk["message"])
            if chunk.get("metadata", {}).get("tool"):
                tools_used.add(chunk["metadata"]["tool"])
            if chunk.get("metadata", {}).get("thought"):
                thoughts.append(chunk["metadata"]["thought"])
                
        return {
            "message": "\n".join(messages),
            "status": "success",
            "metadata": {
                "tools_used": list(tools_used),
                "thought_process": thoughts
            }
        }