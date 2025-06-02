from typing import Literal, TypedDict, Optional
from pydantic import BaseModel
import uuid

from langchain.chat_models.base import BaseChatModel, init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, merge_message_runs, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import CONF, CONFIG_KEY_STORE
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from langgraph_supervisor import create_supervisor
from langmem import create_manage_memory_tool, create_search_memory_tool
from trustcall import create_extractor

from protocol import AIAppProtocol


class Triple(BaseModel): 
    """Store all new facts, preferences, and relationships as triples."""
    subject: str
    predicate: str
    object: str
    context: str | None = None

def create_semantic_memory_manager(llm: BaseChatModel, memory: BaseCheckpointSaver, store: BaseStore) -> CompiledGraph:
    def prompt(state: MessagesState, config: RunnableConfig):
        """Prepare messages with context from existing memories."""
        conf = config.get(CONF)
        store: BaseStore = conf.get(CONFIG_KEY_STORE)
        user_id: str = conf.get("user_id")

        memories = None
        if user_id:
            memories = store.search(
                ("users", user_id, "triples"),
                query=state["messages"][-1].content,
            )

        system_prompt = f"""
        You are a semantic memory manager. Extract and manage all important knowledge, rules, and events using the provided tools.

        Existing memories: (may be empty)
        <memories>
        {memories}
        </memories>
        
        Use the manage_memory tool to update and contextualize existing memories, create new ones, or delete old ones that are no longer valid.
        You can also expand your search of existing memories to augment using the search tool."""
        
        return [
            SystemMessage(content=system_prompt),
            *state["messages"]
        ]

    return create_react_agent(
        llm,
        prompt=prompt,
        tools=[
            create_manage_memory_tool(("users", "{user_id}", "triples"), schema=Triple),
            create_search_memory_tool(("users", "{user_id}", "triples")),
        ],
        checkpointer=memory,
        store=store,
        name="semantic_memory_manager"
    )


class UserProfile(BaseModel):
    """Represents the full representation of a user."""
    name: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None

def create_user_profile_manager(llm: BaseChatModel, memory: BaseCheckpointSaver, store: BaseStore) -> CompiledGraph:

    class UpdateMemory(TypedDict):
        """ Decision on what memory type to update """
        update_type: Literal["user"]


    def call_model(state: MessagesState, config: RunnableConfig):
        conf = config.get(CONF)
        store: BaseStore = conf.get(CONFIG_KEY_STORE)
        user_id: str = conf.get("user_id")

        results = store.search(
            ("users", user_id, "profile")
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

    def update_user(state: MessagesState, config: RunnableConfig) -> MessagesState:
        conf = config.get(CONF)
        store: BaseStore = conf.get(CONFIG_KEY_STORE)
        user_id: str = conf.get("user_id")

        results = store.search(
            ("users", user_id, "profile")
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

        result = extractor.invoke({
            "messages": updated_messages,
            "existing": existing,
        })

        tool_msg = ToolMessage(
            content="no updates",
            tool_call_id=state["messages"][-1].tool_calls[0]["id"],
        )

        if len(result["responses"]) == 0:
            return {"messages": [tool_msg]}

        store.put(
            ("users", user_id, "profile"),
            key=id,
            value=result["responses"][0].model_dump(),
        )

        tool_msg.content = "updated user"
        return {"messages": [tool_msg]}

    workflow = StateGraph(MessagesState)
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

class BasicApp(AIAppProtocol):
    def __init__(self, memory: BaseCheckpointSaver, store: BaseStore):
        llm = init_chat_model("openai:gpt-4o-mini")

        # Create a semantic memory manager
        semantic_memory_manager = create_semantic_memory_manager(llm, memory, store)

        # Create a User Profile manager
        user_profile_manager = create_user_profile_manager(llm, memory, store)

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

        self.app = create_supervisor(
            agents=[
                semantic_memory_manager,
                user_profile_manager,
            ],
            model=llm,
            prompt=system_prompt,
        ).compile(checkpointer=memory, store=store)

    def invoke(self, content: str, session_id: str) -> str:
        config = {
            "configurable": { 
                "thread_id": session_id,
                "user_id": "mirror520",
            } 
        }

        messages = [HumanMessage(content=content)]
        response = self.app.invoke({"messages": messages}, config)
        return response["messages"][-1].content

    async def ainvoke(self, content: str, session_id: str) -> str:
        config = {
            "configurable": { 
                "thread_id": session_id,
                "user_id": "mirror520",
            } 
        }

        messages = [HumanMessage(content=content)]
        response = await self.app.ainvoke({"messages": messages}, config)
        return response["messages"][-1].content

