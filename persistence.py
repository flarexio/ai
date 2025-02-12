from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from protocol import Message, RepositoryProtocol


class DBRepository(RepositoryProtocol):
    __slots__ = ["conn", "memory"]

    def __init__(self, memory: PostgresSaver):
        self.conn = memory.conn
        self.memory = memory

        cmd = """
            CREATE TABLE IF NOT EXISTS sessions (
                id         VARCHAR(26)  NOT NULL PRIMARY KEY,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        self.conn.execute(cmd)

    def store_session(self, session_id: str) -> None:
        cmd = """
            INSERT INTO sessions (id)
            VALUES (%s)
        """
        self.conn.execute(cmd, (session_id,))

    def list_sessions(self) -> list[str]:
        query = """
            SELECT id FROM sessions
        """
        result = self.conn.execute(query)
        return [row["id"] for row in result]

    def find_session(self, session_id: str) -> str:
        query = """
            SELECT id FROM sessions WHERE id = %s
        """
        result = self.conn.execute(query, (session_id,))
        return result.fetchone()

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
                if isinstance(raw, HumanMessage):
                    messages.append(Message(role="human", content=raw.content))
                elif isinstance(raw, AIMessage):
                    messages.append(Message(role="ai", content=raw.content))
                else:
                    raise ValueError("invalid message")
        except Exception as e:
            print(e)
        finally:
            return messages
