"""Integration of XAgent into L3AGI Framework"""

import asyncio
from langchain.schema import SystemMessage
from XAgent.core import XAgentCoreComponents, XAgentParam
from XAgent.agent.dispatcher import XAgentDispatcher
from XAgent.workflow.base_query import AutoGPTQuery
from XAgent.toolserver_interface import ToolServerInterface

from agents.base_agent import BaseAgent
from agents.handle_agent_errors import handle_agent_error
from config import Config
from memory.zep.zep_memory import ZepMemory
from postgres import PostgresChatMessageHistory
from services.pubsub import ChatPubSubService
from services.voice import speech_to_text, text_to_speech
from typings.agent import AgentWithConfigsOutput
from typings.config import AccountSettings, AccountVoiceSettings
from utils.model import get_llm
from utils.system_message import SystemMessageBuilder
from agents.conversational.tool_adapter import ToolRegistry
from agents.conversational.response_transformer import ResponseTransformer, StreamingResponseHandler

class XAgentWrapper(BaseAgent):
    """Wrapper class to integrate XAgent with L3AGI"""
    
    def __init__(self, config):
        self.core = XAgentCoreComponents()
        self.param = XAgentParam(config=config)
        self.tool_registry = ToolRegistry()
        self.response_transformer = ResponseTransformer()
        self.streaming_handler = StreamingResponseHandler()
        
    async def run(
        self,
        settings: AccountSettings,
        voice_settings: AccountVoiceSettings,
        chat_pubsub_service: ChatPubSubService, 
        agent_with_configs: AgentWithConfigsOutput,
        tools,
        prompt: str,
        voice_url: str,
        history: PostgresChatMessageHistory,
        human_message_id: str,
        run_logs_manager,
        pre_retrieved_context: str,
    ):
        """Run XAgent with given inputs"""
        
        try:
            # Initialize core components
            query = AutoGPTQuery(task=prompt)
            self.param.build_query({"task": prompt})
            
            # Build XAgent components
            self.core.build(self.param, interaction=None) # TODO: Add proper interaction object
            self.core.start()
            
            # Process voice input if present
            if voice_url:
                prompt = speech_to_text(voice_url, agent_with_configs.configs, voice_settings)
                
            # Get base LLM
            llm = get_llm(settings, agent_with_configs)

            # Initialize memory
            memory = ZepMemory(
                session_id=str(self.session_id),
                url=Config.ZEP_API_URL,
                api_key=Config.ZEP_API_KEY,
                memory_key="chat_history",
                return_messages=True,
            )
            memory.human_name = self.sender_name 
            memory.ai_name = agent_with_configs.agent.name

            # Register and adapt tools
            for tool in tools:
                self.tool_registry.register_tool(tool)
            
            # Build system message
            system_message = SystemMessageBuilder(agent_with_configs, pre_retrieved_context).build()
            
            # Run XAgent with adapted tools
            adapted_tools = self.tool_registry.list_tools()
            result = await self.core.agent_dispatcher.dispatch_task(
                query,
                tools=adapted_tools,
                system_message=system_message,
                memory=memory
            )
            
            # Transform response
            transformed_result = self.response_transformer.transform_response(result)
            
            # Get transformed response
            response = transformed_result["message"] if transformed_result else "No response generated"
            
            # Handle voice response if configured
            voice_url = None
            if "Voice" in agent_with_configs.configs.response_mode:
                voice_url = text_to_speech(response, agent_with_configs.configs, voice_settings)
            
            # Create AI message
            ai_message = history.create_ai_message(
                response,
                human_message_id,
                agent_with_configs.agent.id,
                voice_url,
            )
            
            # Send message via pubsub
            chat_pubsub_service.send_chat_message(chat_message=ai_message)
            
            # Cleanup
            self.core.close()
            
            return response
            
        except Exception as err:
            error_msg = handle_agent_error(err)
            
            # Save error to memory
            memory.save_context(
                {
                    "input": prompt,
                    "chat_history": memory.load_memory_variables({})["chat_history"], 
                },
                {
                    "output": error_msg,
                }
            )
            
            return error_msg