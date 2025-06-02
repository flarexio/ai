from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from protocol import ChatRepositoryProtocol, Message, Session


class ChatDatabaseRepository(ChatRepositoryProtocol):
    __slots__ = ["conn", "memory"]

    def __init__(self, memory: AsyncPostgresSaver):
        self.conn = memory.conn
        self.memory = memory

    async def migrate(self):
        cmd = """
            CREATE TABLE IF NOT EXISTS sessions (
                id         VARCHAR(26)  NOT NULL PRIMARY KEY,
                app_name   VARCHAR(255) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        await self.conn.execute(cmd)

    async def store_session(self, session: Session):
        cmd = """
            INSERT INTO sessions (id, app_name)
            VALUES (%s, %s)
        """
        await self.conn.execute(cmd, (session.id, session.app_name))

    async def list_sessions(self) -> list[Session]:
        query = """
            SELECT id, app_name FROM sessions
        """
        result = await self.conn.execute(query)
        return [Session(id=row["id"], app_name=row["app_name"]) for row in result]

    async def find_session(self, session_id: str) -> Session:
        query = """
            SELECT id, app_name FROM sessions WHERE id = %s
        """
        result = await self.conn.execute(query, (session_id,))
        row = await result.fetchone()
        if not row:
            raise ValueError("session not found")
        return Session(id=row["id"], app_name=row["app_name"])

    async def list_messages(self, session_id: str) -> list[Message]:
        if not await self.find_session(session_id):
            raise ValueError("session not found")

        config = {
            "configurable": {
                "thread_id": session_id,
                "limit": 1,
            }
        }

        messages: list[Message] = []
        try:
            tuple = self.memory.get_tuple(config)
            raws = tuple.checkpoint["channel_values"]["messages"]

            for raw in raws:
                if raw.content == "":
                    continue

                if isinstance(raw, HumanMessage):
                    messages.append(Message(role="human", content=raw.content))
                elif isinstance(raw, AIMessage):
                    messages.append(Message(role="ai", content=raw.content))
                elif isinstance(raw, SystemMessage):
                    messages.append(Message(role="system", content=raw.content))
                elif isinstance(raw, ToolMessage):
                    messages.append(Message(role="tool", content=raw.content))
                else:
                    raise ValueError("invalid message")
        except Exception as e:
            print(e)
        finally:
            return messages
