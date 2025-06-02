from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import CONF
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

async def main():
    async with MultiServerMCPClient(
        {
            "iiot": {
                "command": "iiot_mcp",
                "args": [
                    "--creds", "/home/ar0660/.flarex/edge/user.creds",
                ],
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

        agent = create_react_agent(
            "openai:gpt-4o-mini", 
            tools,
            prompt=prompt,
        )

        config = { "configurable" : { 
            "edge_id": "01J6TRZ0RWW334GPRMH5NSKJQA" 
        } }

        response = await agent.ainvoke({
            "messages": [ HumanMessage(content="幫我查查 127.0.0.1:10502 能不能連線") ]
        }, config)

        print(response)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
