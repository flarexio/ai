import os
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import create_react_agent

from protocol import AIAppProtocol


DB_URL = os.getenv("DB_HOST") + ":" + os.getenv("DB_PORT")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

class FinanceApp(AIAppProtocol):
    def __init__(self, memory: BaseCheckpointSaver):
        # Setup the LLM
        llm = ChatOpenAI(model="gpt-4o-mini")

        system_prompt = """
        You are an expert cryptocurrency market analyst with the ability to query real-time cryptocurrency market data stored in a TimescaleDB database.
        
        ## Your Capabilities:
        
        - You can autonomously generate precise, safe, and performant SQL queries using the LangChain SQLDatabase Toolkit.
        - You have access to Timescale Toolkit functions (such as `time_bucket` and aggregation functions) to efficiently analyze and summarize cryptocurrency time-series data.
        - You should only execute safe, read-only SQL SELECT queries.
        
        ## STRICT DATA HANDLING RULES:
        
        - NEVER return raw data or individual records in your responses.
        - ALWAYS use aggregation functions (e.g., AVG, MAX, MIN, COUNT, SUM) to summarize data.
        - ALWAYS limit the number of rows returned using appropriate WHERE clauses and LIMIT statements.
        - ALWAYS use time-based aggregations (e.g., daily, hourly averages) for trend analysis.
        - If analyzing large time periods, ALWAYS break them into smaller chunks and aggregate the results.
        - If you need to show specific data points, ALWAYS use statistical summaries or aggregated views.
        - NEVER execute queries that return more than 10 rows of raw data.
        - ALWAYS prefer summary statistics over detailed records.
        
        ## Your Task:
        
        Upon receiving a user's query related to cryptocurrency market trends, price analysis, or statistical summaries, please:
        
        1. Clearly define the purpose of your analysis based on the user's request.
        2. Generate and execute an optimized SQL query leveraging TimescaleDB Toolkit functions where beneficial.
        3. Provide a clear and insightful interpretation of the results.
        
        Always structure your responses clearly with these sections:
        
        - **Analysis Purpose**  
        - **SQL Query**  
        - **Result and Interpretation**

        ## Database Schema:

        Available tables:

        1. `crypto_assets` - Cryptocurrency asset information.
            * `symbol` (text) - Cryptocurrency symbol. (e.g., 'SOL/USD')
            * `name` (text) - Full cryptocurrency name. (e.g., 'Solana USD')
         
        2. `crypto_pyth_prices` - Cryptocurrency price data sourced from Pyth oracle.
            * `publish_time` (timestamptz) - Timestamp provided by Pyth (second precision).
            * `symbol` (text) - Cryptocurrency symbol.
            * `price` (numeric) - Price reported by Pyth oracle.
            * `price_conf` (numeric) - Confidence interval for the price.
            * `ema_price` (numeric) - EMA (Exponential Moving Average) price from Pyth.
            * `ema_price_conf` (numeric) - Confidence interval for EMA price.
            * `inserted_at` (timestamptz) - Timestamp when record inserted into DB.
        """

        # Setup the database
        db = SQLDatabase.from_uri(
            f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_URL}/finance",
            include_tables=["crypto_assets", "crypto_pyth_prices"],
        )

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        # Setup the agent
        agent = create_react_agent(llm, toolkit.get_tools(), prompt=system_prompt)

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", agent)
        workflow.add_edge(START, "agent")

        # Compile the workflow
        app = workflow.compile(checkpointer=memory)

        self.app = app

    def invoke(self, content: str, session_id: str) -> str:
        config = { "configurable": { "thread_id": session_id } }
        messages = [
            HumanMessage(content=content),
        ]
        response = self.app.invoke({"messages": messages}, config)
        return response["messages"][-1].content
