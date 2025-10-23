import pytest
from apps.server.agents.conversational.conversational import ConversationalAgent

class TestConversationalAgentIntegration:
    
    @pytest.fixture
    def agent(self):
        """Create conversational agent"""
        return ConversationalAgent()
    
    @pytest.mark.asyncio
    async def test_end_to_end_conversation(self, agent):
        """Test complete conversation flow"""
        settings = {
            "model": "gpt-4",
            "temperature": 0.7
        }
        
        response = await agent.run(
            settings=settings,
            tools=[],
            prompt="Hello, how are you?",
            history=[]
        )
        
        assert response is not None
        assert response["status"] == "success"
        assert len(response["message"]) > 0
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, agent):
        """Test multi-turn conversation with memory"""
        settings = {"model": "gpt-4"}
        
        # Turn 1
        response1 = await agent.run(
            settings=settings,
            tools=[],
            prompt="My favorite color is blue",
            history=[]
        )
        
        # Turn 2
        response2 = await agent.run(
            settings=settings,
            tools=[],
            prompt="What is my favorite color?",
            history=[
                {"role": "user", "content": "My favorite color is blue"},
                {"role": "assistant", "content": response1["message"]}
            ]
        )
        
        assert "blue" in response2["message"].lower()
    
    @pytest.mark.asyncio
    async def test_tool_chain_execution(self, agent):
        """Test execution of multiple tools in sequence"""
        class SearchTool:
            name = "search"
            description = "Search for information"
            def run(self, query: str):
                return f"Search results for: {query}"
        
        class CalculatorTool:
            name = "calculator"
            description = "Perform calculations"
            def run(self, expression: str):
                return str(eval(expression))
        
        tools = [SearchTool(), CalculatorTool()]
        
        response = await agent.run(
            settings={"model": "gpt-4"},
            tools=tools,
            prompt="Search for Python and calculate 10 + 5",
            history=[]
        )
        
        assert response["status"] == "success"
        assert len(response["metadata"]["tools_used"]) > 0