from langchain.chat_models.base import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import CONF
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.store.base import BaseStore

from .base import BaseAIApp


SYSTEM_INSTRUCTION = """You are an AI coding assistant that helps users write, debug, and test code using Docker containers as isolated sandbox environments.

## Workspace Environment:
- Workspace ID: {workspace_id}
- Host Workspace Path: {workspace_path}
- Container Mount Strategy: You decide where to mount the workspace in containers
- All code should be placed in the host workspace for persistence

## Available Tools:
**File Operations**: Create, read, write files in workspace
**Images**: ImageDescription, ListImages, PullImage  
**Containers**: ListContainers, RunContainerOnce, RunContainer, RemoveContainer, RemoveAllContainers
**Execution**: SendToContainer, LogsContainer, ExecCommand, SendAndRead, Wait

## Mount Strategy Guidelines:
- **Check image defaults first**: Use ImageDescription to understand the Docker image's default working directory and structure
- **Choose appropriate mount point**: Consider mounting to image's default workdir or a suitable subdirectory
- **Common mount points**: /app, /code, /workspace, /usr/src/app, /home/user, or image's WORKDIR
- **You can change WorkDir**: Use `cd` commands or specify working directory when needed

## General Guidelines:
- Create files directly in host workspace: {workspace_path}
- Use ImageDescription to understand image structure and default directories before creating containers
- Use PullImage to download required Docker images if not available
- Use RunContainerOnce for simple tasks, RunContainer for complex operations
- Choose appropriate images (python:3.11, node:18, ubuntu:22.04, etc.)
- Keep Wait times short (1-10 seconds)
- Use LogsContainer with 'since' timestamp from SendToContainer responses
- Clean up with RemoveContainer when finished
- **Stop after 2-3 failures and ask user for guidance**

## Workflow:
1. Create/prepare files in host workspace: {workspace_path}
2. Pull required Docker image if not available (use PullImage)
3. **Use ImageDescription to understand image structure and default working directory**
4. Decide appropriate mount point based on image information
5. Create container with chosen mount strategy
6. Execute code (files accessible at your chosen mount point)
7. Get results with LogsContainer using proper timestamps
8. Save results to host workspace if needed
9. Clean up resources

## Example Decision Process:
1. Check `ImageDescription python:3.11` → might show WORKDIR /usr/src/app
2. Mount {workspace_path} to /usr/src/app or create subfolder
3. Or mount to /workspace and `cd /workspace` before execution

## Error Handling:
- After 2-3 failed attempts, summarize errors and ask for user guidance
- Provide clear error info and suggest solutions

Always explain your mounting strategy and verify the image's default directories before proceeding.
"""

class CodeAIApp(BaseAIApp):
    def __init__(self, memory: BaseCheckpointSaver, store: BaseStore, toolkit: dict[str, list[BaseTool]]):
        model = init_chat_model("openai:gpt-4.1-mini")

        tools = toolkit["mcpblade"] + toolkit["filesystem"]

        def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
            conf = config.get(CONF, {})
            # workspace_id = conf.get("workspace_id", "flarex")
            workspace_id = "flarex"
            workspace_path = f"/home/ar0660/.flarex/forge/workspaces/{workspace_id}"

            system_prompt = SYSTEM_INSTRUCTION.format(
                workspace_id=workspace_id,
                workspace_path=workspace_path,
            )

            return [
                SystemMessage(content=system_prompt),
                *state["messages"]
            ]

        self.app = create_react_agent(model, tools,
            prompt=prompt,
            checkpointer=memory,
            store=store,
        )

    def id(self) -> str:
        return "code"

    def name(self) -> str:
        return "Code AI"

    def description(self) -> str:
        return "An AI assistant that helps with coding tasks."

    def version(self) -> str:
        return "1.0.0"
