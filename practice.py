from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    client = MultiServerMCPClient(
        {
            "iiot": {
                "command": "iiot_mcp",
                "args": [
                    "--creds", "/home/ar0660/.flarex/edge/user.creds",
                ],
                "transport": "stdio",
            },
            # "mcpblade": {
            #     "command": "mcpblade_mcp_server",
            #     "args": [
            #         "--edge-id", "01JXCHPCT4S10YKVPG4XGRDGCX",
            #     ],
            #     "env": {
            #         "NATS_CREDS": "/home/ar0660/.flarex/iiot/user.creds",
            #     },
            #     "transport": "stdio",
            # },
            # "filesystem": {
            #     "command": "mcpblade_mcp_server",
            #     "args": [
            #         "--edge-id", "01JXCHPCT4S10YKVPG4XGRDGCX",
            #         "--server-id", "filesystem",
            #         "--cmd", "npmx -y @modelcontextprotocol/server-filesystem /home/ar0660/joke/"
            #     ],
            #     "env": {
            #         "NATS_CREDS": "/home/ar0660/.flarex/iiot/user.creds",
            #     },
            #     "transport": "stdio",
            # }
        },
    )

    tools = await client.get_tools()

    class ContextInfo(BaseModel):
        edge_id: str

    @dynamic_prompt
    def prompt(request: ModelRequest) -> SystemMessage:
        ctx: ContextInfo = request.runtime.context

        system_prompt = f"""
        You are an assistant for Industrial IoT (IIoT) systems.
        You can use the iiot tools to check whether edge devices are reachable.
        Use device IP, port, and protocol to test connectivity and report the result.

        You are given the following information:
        - Edge ID: {ctx.edge_id}
        """

        return SystemMessage(content=system_prompt)

    iiot_agent = create_agent(
        "openai:gpt-5-mini", 
        tools,
        middleware=[prompt],
        context_schema=ContextInfo,
        name="iiot_assistant",
    )

    @tool("iiot_agent", description="IIoT agent for checking device connectivity")
    async def call_iiot_agent(query: str):
        try:
            result = await iiot_agent.ainvoke({"messages": [ HumanMessage(content=query) ]})
            return result["messages"][-1].content
        except Exception as e:
            return f"Error: {str(e)}"


    chat_agent = create_agent(
        "openai:gpt-5-mini",
        system_prompt="You are a chat assistant.",
        name="chat_assistant",
    )

    @tool("chat_agent", description="Chat agent for general conversation")
    async def call_chat_agent(query: str):
        result = await chat_agent.ainvoke({"messages": [ HumanMessage(content=query) ]})
        return result["messages"][-1].content

    supervisor = create_agent(
        model="openai:gpt-5-mini",
        tools=[ call_iiot_agent, call_chat_agent ],
        system_prompt="You are a supervisor that manages multiple agents.",
    )

    ctx = ContextInfo(edge_id="01JXCHPCT4S10YKVPG4XGRDGCX")

    response = await supervisor.ainvoke(
        { "messages": [ HumanMessage(content="幫我查查 127.0.0.1:502 能不能連線") ] }, 
        context=ctx
    )

    print(response["messages"][-1].content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
