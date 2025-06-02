from ulid import ULID

from decorator import logging
from persistences.db.chat import ChatRepositoryProtocol
from protocol import AIAppProtocol, ChatContext, ChatServiceProtocol, Message, Session


@logging
class ChatService(ChatServiceProtocol):
    __slots__ = ["apps", "chats"]

    def __init__(self, chats: ChatRepositoryProtocol):
        self.apps: dict[str, AIAppProtocol] = {}
        self.chats = chats

    def add_app(self, name: str, app: AIAppProtocol):
        self.apps[name] = app

    def find_app(self, name: str) -> AIAppProtocol:
        if name not in self.apps:
            raise ValueError("app not found")
        return self.apps[name]

    async def create_session(self, app_name: str) -> str:
        session_id = str(ULID())
        session = Session(
            id=session_id,
            app_name=app_name,
        )
        await self.chats.store_session(session)
        return session_id

    async def list_sessions(self) -> list[Session]:
        return await self.chats.list_sessions()

    async def send_message(self, ctx: ChatContext, content: str) -> str:
        session = await self.chats.find_session(ctx.session_id)
        if not session:
            raise ValueError("session not found")
        app = self.find_app(session.app_name)
        response = await app.ainvoke(content, session.id)
        return response

    async def list_messages(self, session_id: str) -> list[Message]:
        return await self.chats.list_messages(session_id)
