from langchain_core.messages import AIMessageChunk, AnyMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import CONF
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph_supervisor import create_supervisor

async def main():
    async with MultiServerMCPClient(
        {
            "iiot": {
                "command": "iiot_mcp",
                "args": [
                    "--creds", "/home/ar0660/.flarex/edge/user.creds",
                ],
                "transport": "stdio",
            },
            "mcpblade": {
                "command": "mcpblade_mcp_server",
                "args": [
                    "--edge-id", "01JXCHPCT4S10YKVPG4XGRDGCX",
                ],
                "env": {
                    "NATS_CREDS": "/home/ar0660/.flarex/iiot/user.creds",
                },
                "transport": "stdio",
            },
            "filesystem": {
                "command": "mcpblade_mcp_server",
                "args": [
                    "--edge-id", "01JXCHPCT4S10YKVPG4XGRDGCX",
                    "--server-id", "filesystem",
                    "--cmd", "npmx -y @modelcontextprotocol/server-filesystem /home/ar0660/joke/"
                ],
                "env": {
                    "NATS_CREDS": "/home/ar0660/.flarex/iiot/user.creds",
                },
                "transport": "stdio",
            }
        },
    ) as client:

        tools = client.get_tools()

        def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
            conf = config.get(CONF)
            edge_id = conf.get("edge_id")

            system_prompt = f"""
            You are an assistant for Industrial IoT (IIoT) systems.
            You can use the iiot tools to check whether edge devices are reachable.
            Use device IP, port, and protocol to test connectivity and report the result.

            You are given the following information:
            - Edge ID: {edge_id}
            """

            return [
                SystemMessage(content=system_prompt),
                *state["messages"],
            ]

        iiot_agent = create_react_agent(
            "openai:gpt-4o-mini", 
            tools,
            prompt=prompt,
            name="iiot_assistant",
        )

        chat_agent = create_react_agent(
            "openai:gpt-4o-mini",
            [],
            prompt="You are a chat assistant.",
            name="chat_assistant",
        )

        supervisor = create_supervisor(
            agents=[ iiot_agent, chat_agent ],
            model=ChatOpenAI(model="gpt-4o-mini"),
            prompt="You are a supervisor that manages multiple agents.",
        ).compile()

        config = { "configurable" : { 
            "edge_id": "01J6TRZ0RWW334GPRMH5NSKJQA" 
        } }

        # response = await agent.ainvoke({
        #     "messages": [ HumanMessage(content="幫我查查 127.0.0.1:10502 能不能連線") ]
        # }, config)
        # print(response)

        previous_nodes = None
        async for event in supervisor.astream({
            "messages": [ HumanMessage(content="幫我查查 Modbus 127.0.0.1:10502 能不能連線") ]
        }, config, stream_mode=["messages"], subgraphs=True):
            nodes, mode, message = event
            chunk, metadata = message
            if previous_nodes != nodes:
                print(list(nodes))
                print("\n")
                previous_nodes = nodes
            if isinstance(chunk, AIMessageChunk):
                if chunk.content:
                    print(chunk.content, end="|", flush=True)
                if chunk.tool_call_chunks:
                    for tool_call_chunk in chunk.tool_call_chunks:
                        if tool_call_chunk["name"]:
                            print(f"Tool name: {tool_call_chunk["name"]}")
                            print("Tool args: ")
                        if tool_call_chunk["args"]:
                            print(tool_call_chunk["args"], end="|", flush=True)
            elif isinstance(chunk, ToolMessage):
                print(chunk.content)

            # for name, data in nodes.items():
            #     print(f"Node: {name}")
            #     for message in data["messages"]:
            #         print(f"ID: {message.id}")
            #         print(f"{message.name}")
            #         message.pretty_print()
            #         print("\n")
            #     print("\n")
            # print("\n" + "-" * 20 + "\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
