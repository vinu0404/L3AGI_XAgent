import pytest
import asyncio
from apps.server.agents.conversational.xagent_integration import XAgentWrapper
from apps.server.agents.conversational.tool_adapter import ToolAdapter, ToolRegistry

class TestXAgentIntegration:
    
    @pytest.fixture
    async def xagent_wrapper(self):
        """Create XAgent wrapper for testing"""
        config = {
            "toolserver": {
                "host": "localhost",
                "port": 8080
            },
            "model": {
                "name": "gpt-4",
                "temperature": 0.7
            }
        }
        return XAgentWrapper(config)
    
    @pytest.mark.asyncio
    async def test_basic_execution(self, xagent_wrapper):
        """Test basic XAgent execution"""
        response = await xagent_wrapper.run(
            prompt="What is 2+2?",
            tools=[],
            settings={}
        )
        
        assert response is not None
        assert response["status"] == "success"
        assert "4" in response["message"]
    
    @pytest.mark.asyncio
    async def test_tool_registration(self, xagent_wrapper):
        """Test tool registration and execution"""
        # Create mock tool
        class MockTool:
            name = "calculator"
            description = "Performs calculations"
            
            def run(self, expression: str):
                return eval(expression)
        
        # Register tool
        tool = MockTool()
        xagent_wrapper.tool_registry.register_tool(tool)
        
        # Execute with tool
        response = await xagent_wrapper.run(
            prompt="Calculate 10 * 5",
            tools=[tool],
            settings={}
        )
        
        assert response["status"] == "success"
        assert "50" in response["message"]
        assert "calculator" in response["metadata"]["tools_used"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, xagent_wrapper):
        """Test error handling"""
        # Test with invalid tool
        class BrokenTool:
            name = "broken"
            description = "Always fails"
            
            def run(self):
                raise Exception("Tool error")
        
        tool = BrokenTool()
        xagent_wrapper.tool_registry.register_tool(tool)
        
        response = await xagent_wrapper.run(
            prompt="Use the broken tool",
            tools=[tool],
            settings={}
        )
        
        assert response["status"] == "error"
        assert "error" in response["message"].lower()
    
    @pytest.mark.asyncio
    async def test_memory_sync(self, xagent_wrapper):
        """Test memory synchronization"""
        # First interaction
        await xagent_wrapper.run(
            prompt="My name is Alice",
            tools=[],
            settings={}
        )
        response = await xagent_wrapper.run(
            prompt="What is my name?",
            tools=[],
            settings={}
        )
        
        assert "Alice" in response["message"]


class TestToolAdapter:
    
    def test_tool_adaptation(self):
        """Test L3AGI to XAgent tool adaptation"""
        class L3AGITool:
            name = "test_tool"
            description = "Test tool"
            
            def run(self, arg1: str):
                return f"Result: {arg1}"
        
        tool = L3AGITool()
        adapter = ToolAdapter(tool)
        
        assert adapter.tool_name == "test_tool"
        assert adapter.tool_description == "Test tool"
    
    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test async tool execution"""
        class AsyncTool:
            name = "async_tool"
            description = "Async test tool"
            
            async def run(self, value: int):
                await asyncio.sleep(0.1)
                return value * 2
        
        tool = AsyncTool()
        adapter = ToolAdapter(tool)
        result = await adapter.tool_runner(value=5)
        
        assert result["output"] == 10


class TestResponseTransformer:
    
    def test_response_transformation(self):
        """Test XAgent to L3AGI response transformation"""
        from apps.server.agents.conversational.response_transformer import ResponseTransformer
        
        # Mock XAgent response
        class MockXAgentResponse:
            output = "Test response"
            status = "COMPLETED"
            tools_used = [{"name": "tool1"}, {"name": "tool2"}]
            execution_time = 1.5
            thoughts = [{"content": "thinking...", "type": "reasoning"}]
        
        transformer = ResponseTransformer()
        result = transformer.transform_response(MockXAgentResponse())
        
        assert result["message"] == "Test response"
        assert result["status"] == "success"
        assert len(result["metadata"]["tools_used"]) == 2
        assert result["metadata"]["execution_time"] == 1.5