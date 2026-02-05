from abc import abstractmethod
from typing import AsyncIterator

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from protocol import AIAppProtocol, AppInfo, ChatContext, MessageChunk, Role, ToolCall


class BaseAIApp(AIAppProtocol):
    """Base implementation for AI Apps with default ainvoke and astream."""
    
    def __init__(self, app):
        self.app = app
    
    @abstractmethod
    def id(self) -> str:
        """Must be implemented by subclasses."""
        ...
    
    @abstractmethod
    def name(self) -> str:
        """Must be implemented by subclasses."""
        ...
    
    @abstractmethod
    def description(self) -> str:
        """Must be implemented by subclasses."""
        ...
    
    def version(self) -> str:
        """Default version - can be overridden."""
        return "1.0.0"
    
    def info(self) -> AppInfo:
        """Return information about the app."""
        return AppInfo(
            id=self.id(),
            name=self.name(),
            description=self.description(),
            version=self.version()
        )

    async def ainvoke(self, ctx: ChatContext, content: str) -> str:
        """Default implementation - can be overridden by subclasses."""
        config = {
            "configurable": { 
                "thread_id": ctx.session_id, 
                "user_id": ctx.user_id,
                "customer_id": ctx.customer_id,
                "workspace_id": ctx.workspace_id,
            }
        }
        messages = [HumanMessage(content=content)]
        try:
            response = await self.app.ainvoke({"messages": messages}, config)
            return response["messages"][-1].content
        except Exception as e:
            print(f"Error in ainvoke: {e}")
            return f"error: {e}"

    async def astream(self, ctx: ChatContext, content: str) -> AsyncIterator[MessageChunk]:
        """Default implementation - can be overridden by subclasses."""
        config = {
            "configurable": { 
                "thread_id": ctx.session_id, 
                "user_id": ctx.user_id,
                "customer_id": ctx.customer_id,
                "workspace_id": ctx.workspace_id,
            }
        }

        messages = [HumanMessage(content=content)]

        current_nodes = None
        
        try:
            async for event in self.app.astream({"messages": messages}, config, 
                stream_mode=["messages"], 
                subgraphs=True,
            ):
                nodes, _, message = event
                chunk, _ = message

                if isinstance(chunk, AIMessageChunk):
                    tool_calls = []
                    for tool_call_chunk in chunk.tool_call_chunks:
                        tool_call = ToolCall(
                            id=tool_call_chunk.get("id"),
                            name=tool_call_chunk.get("name"),
                            args=tool_call_chunk.get("args"),
                        )
                        tool_calls.append(tool_call)

                    ai_chunk = MessageChunk(
                        nodes=list(nodes),
                        role=Role.AI,
                        content=chunk.content,
                        tool_calls=tool_calls,
                        is_new=True if current_nodes != nodes else False,
                    )
                    yield ai_chunk

                elif isinstance(chunk, ToolMessage):
                    tool_chunk = MessageChunk(
                        nodes=list(nodes),
                        role=Role.TOOL,
                        content=chunk.content,
                        tool_call_id=chunk.tool_call_id,
                        is_new=True if current_nodes != nodes else False,
                    )
                    yield tool_chunk

                current_nodes = nodes

        except Exception as e:
            print(f"Error in astream: {e}")
            error_chunk = MessageChunk(
                nodes=list(nodes) if current_nodes else [],
                role=Role.AI,
                content=f"error: {e}",
                is_complete=True,
            )
            yield error_chunk
