from typing import Literal, TypedDict, Optional
from pydantic import BaseModel
import uuid

from langchain_core.messages import merge_message_runs
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.messages import HumanMessage, ToolMessage, SystemMessage, trim_messages
from langchain.tools import tool, BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from trustcall import create_extractor

from protocol import ChatContext
from .base import BaseAIApp


class Triple(BaseModel): 
    """Store all new facts, preferences, and relationships as triples."""
    subject: str
    predicate: str
    object: str
    context: str | None = None

def create_semantic_memory_manager(llm: BaseChatModel, memory: BaseCheckpointSaver, store: BaseStore) -> CompiledStateGraph:

    @dynamic_prompt
    async def prompt(request: ModelRequest) -> SystemMessage:
        ctx: ChatContext = request.runtime.context

        memories = None
        if ctx.user_id:
            memories = await request.runtime.store.asearch(
                ("users", ctx.user_id, "triples"),
                query=request.messages[-1].content,  # use the latest user message as the query for relevant memories
            )

        system_prompt = f"""
        You are a semantic memory manager. Extract and manage all important knowledge, rules, and events using the provided tools.

        Existing memories: (may be empty)
        <memories>
        {memories}
        </memories>
        
        Use the manage_memory tool to update and contextualize existing memories, create new ones, or delete old ones that are no longer valid.
        You can also expand your search of existing memories to augment using the search tool."""
        
        return SystemMessage(content=system_prompt)

    return create_agent(
        llm,
        tools=[
            create_manage_memory_tool(("users", "{user_id}", "triples"), schema=Triple),
            create_search_memory_tool(("users", "{user_id}", "triples")),
        ],
        middleware=[prompt],
        context_schema=ChatContext,
        checkpointer=memory,
        store=store,
        name="semantic_memory_manager"
    )


class UserProfile(BaseModel):
    """Represents the full representation of a user."""
    name: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None

def create_user_profile_manager(llm: BaseChatModel, memory: BaseCheckpointSaver, store: BaseStore) -> CompiledStateGraph:

    class UpdateMemory(TypedDict):
        """ Decision on what memory type to update """
        update_type: Literal["user"]


    async def call_model(state: MessagesState, runtime: Runtime[ChatContext]):
        ctx = runtime.context

        results = await runtime.store.asearch(
            ("users", ctx.user_id, "profile")
        )

        profile = None
        if results:
            profile = results[0]

        system_prompt = f"""
        You are a user profile manager.
        
        You help manage and maintain the user's profile.
        
        Here is the current user profile (may be empty if no information has been collected yet):
        <user_profile>
        {profile}
        </user_profile>
        
        Your instructions:
        
        1. Carefully read the user's recent messages.
        
        2. Decide whether the profile should be updated:
           - If new personal information is provided
           - If existing information is modified
           - If the user requests a change
           - If information is confirmed
        
           If an update is needed, call the `UpdateMemory` tool with type "user".
        
        3. After calling a tool or if no update is needed, respond naturally and concisely to the user.
        """

        model = llm.bind_tools([UpdateMemory], parallel_tool_calls=False)

        recent_messages = trim_messages(
            state["messages"], 
            max_tokens=10,
            token_counter=len,
            start_on="human",
        )

        messages = [SystemMessage(content=system_prompt)] + recent_messages
        response = model.invoke(messages)

        return {"messages": [response]}

    def route_message(state: MessagesState) -> Literal["__end__", "update_user"]:
        message = state["messages"][-1]
        if len(message.tool_calls) == 0:
            return "__end__"
        else:
            tool_call = message.tool_calls[0]
            if tool_call["args"]["update_type"] == "user":
                return "update_user"
            else:
                raise ValueError


    extractor = create_extractor(
        llm,
        tools=[UserProfile],
        tool_choice="UserProfile",
    )

    async def update_user(state: MessagesState, runtime: Runtime[ChatContext]) -> MessagesState:
        ctx = runtime.context

        results = await runtime.store.asearch(
            ("users", ctx.user_id, "profile")
        )

        id = str(uuid.uuid4())
        profile = UserProfile() 
        if results:
            id = results[0].key
            profile = UserProfile.model_validate(results[0].value)

        existing = {
            "UserProfile": profile.model_dump(),
        }

        prompt = f"""
        Reflect on the following interaction. 

        Use the provided tools to retain any necessary memories about the user. 
        
        Use parallel tool calling to handle updates and insertions simultaneously.
        """

        # add short-term memory to the model
        recent_messages = trim_messages(
            state["messages"], 
            max_tokens=10,
            token_counter=len,
            start_on="human",
        )
        updated_messages = list(merge_message_runs(messages=[SystemMessage(content=prompt)] + recent_messages[:-1]))

        result = await extractor.ainvoke({
            "messages": updated_messages,
            "existing": existing,
        })

        tool_msg = ToolMessage(
            content="no updates",
            tool_call_id=state["messages"][-1].tool_calls[0]["id"],
        )

        if len(result["responses"]) == 0:
            return {"messages": [tool_msg]}

        await runtime.store.aput(
            ("users", ctx.user_id, "profile"),
            key=id,
            value=result["responses"][0].model_dump(),
        )

        tool_msg.content = "updated user"
        return {"messages": [tool_msg]}


    workflow = StateGraph(MessagesState, ChatContext)
    workflow.add_node("manager", call_model)
    workflow.add_node("update_user", update_user)

    workflow.add_edge(START, "manager")
    workflow.add_conditional_edges("manager", route_message)
    workflow.add_edge("update_user", "manager")

    return workflow.compile(
        checkpointer=memory,
        store=store,
        name="user_profile_manager"
    )

class BasicAIApp(BaseAIApp):
    def __init__(self, memory: BaseCheckpointSaver, store: BaseStore, toolkit: dict[str, list[BaseTool]]):
        llm = init_chat_model("openai:gpt-5-mini")

        tools = toolkit["mcpblade"]

        # Create a semantic memory manager
        semantic_memory_manager = create_semantic_memory_manager(llm, memory, store)

        @tool("semantic_memory_manager", description="Agent for managing semantic memory")
        async def call_semantic_memory_manager(query: str):
            result = await semantic_memory_manager.ainvoke({"messages": [ HumanMessage(content=query) ]})
            return result["messages"][-1].content


        # Create a User Profile manager
        user_profile_manager = create_user_profile_manager(llm, memory, store)

        @tool("user_profile_manager", description="Agent for managing user profiles")
        async def call_user_profile_manager(query: str):
            result = await user_profile_manager.ainvoke({"messages": [ HumanMessage(content=query) ]})
            return result["messages"][-1].content


        system_prompt = """
        You are a helpful AI assistant coordinating two specialized agents:

        1. Semantic Memory Manager:
           - Manages and queries structured semantic knowledge in triple form.
           - Supports fact insertion, deletion, search, and reasoning over stored knowledge.

        2. User Profile Manager:
           - Handles simple user profile data including name, language, and timezone.
           - Supports only updating existing user profiles and searching for user profile information.

        Your task is to interpret user requests and delegate them to the appropriate agent.

        Do not execute actions directly; always delegate to the correct agent based on the request type.
        """

        self.app = create_agent(
            llm,
            [call_semantic_memory_manager, call_user_profile_manager] + tools,
            system_prompt=system_prompt,
            context_schema=ChatContext,
            checkpointer=memory,
            store=store,
        )

    def id(self) -> str:
        return "basic"

    def name(self) -> str:
        return "Basic"

    def description(self) -> str:
        return "An AI app that manages semantic memory and user profiles."

    def version(self) -> str:
        return "1.0.0"
