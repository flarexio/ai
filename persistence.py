from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver

from protocol import Message, RepositoryProtocol, Session


class DBRepository(RepositoryProtocol):
    __slots__ = ["conn", "memory"]

    def __init__(self, memory: PostgresSaver):
        self.conn = memory.conn
        self.memory = memory

        cmd = """
            CREATE TABLE IF NOT EXISTS sessions (
                id         VARCHAR(26)  NOT NULL PRIMARY KEY,
                app_name   VARCHAR(255) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        self.conn.execute(cmd)

    def store_session(self, session_id: str, app_name: str) -> None:
        cmd = """
            INSERT INTO sessions (id, app_name)
            VALUES (%s, %s)
        """
        self.conn.execute(cmd, (session_id, app_name))

    def list_sessions(self) -> list[Session]:
        query = """
            SELECT id, app_name FROM sessions
        """
        result = self.conn.execute(query)
        return [Session(id=row["id"], app_name=row["app_name"]) for row in result]

    def find_session(self, session_id: str) -> Session:
        query = """
            SELECT id, app_name FROM sessions WHERE id = %s
        """
        result = self.conn.execute(query, (session_id,))
        row = result.fetchone()
        if not row:
            raise ValueError("session not found")
        return Session(id=row["id"], app_name=row["app_name"])

    def list_messages(self, session_id: str) -> list[Message]:
        if not self.find_session(session_id):
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
