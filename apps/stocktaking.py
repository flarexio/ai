from langchain.chat_models.base import init_chat_model
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore

from .base import BaseAIApp


SYSTEM_PROMPT = """
你是一個庫存進貨助理。

你的任務是：
從使用者提供的進貨單文字或圖片中提取食材資料，並使用 Excel 工具寫入指定的 Excel 檔案中。

你可以使用 Excel 相關工具來建立工作表或寫入資料。
請根據進貨單內容，正確使用這些工具完成動作，不需要解釋或轉換格式。
"""

class StocktakingApp(BaseAIApp):
    def __init__(self, memory: BaseCheckpointSaver, store: BaseStore, toolkit: dict[str, list[BaseTool]]):
        llm = init_chat_model("openai:gpt-4o-mini")

        tools = toolkit["excel"]

        self.app = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SYSTEM_PROMPT,
        ).compile(checkpointer=memory, store=store)

    def id(self) -> str:
        return "stocktaking"

    def name(self) -> str:
        return "Stocktaking"

    def description(self) -> str:
        return "An AI assistant for managing semantic knowledge and user profiles in stocktaking applications."

    def version(self) -> str:
        return "1.0.0"
