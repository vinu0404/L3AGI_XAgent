import pytest
import time
import asyncio
from apps.server.agents.conversational.conversational import ConversationalAgent

class TestPerformance:
    
    @pytest.mark.asyncio
    async def test_response_time(self):
        """Test response time is within acceptable limits"""
        agent = ConversationalAgent()
        
        start = time.time()
        response = await agent.run(
            settings={"model": "gpt-4"},
            tools=[],
            prompt="What is 2+2?",
            history=[]
        )
        duration = time.time() - start
        
        assert duration < 5.0  
        assert response["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        agent = ConversationalAgent()
        
        async def make_request(prompt):
            return await agent.run(
                settings={"model": "gpt-4"},
                tools=[],
                prompt=prompt,
                history=[]
            )
        
        tasks = [
            make_request(f"Question {i}")
            for i in range(5)
        ]
        
        start = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start
        
        assert len(results) == 5
        assert all(r["status"] == "success" for r in results)
        assert duration < 15.0 