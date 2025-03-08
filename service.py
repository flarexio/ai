from ulid import ULID

from decorator import logging
from persistence import RepositoryProtocol
from protocol import AIAppProtocol, ChatContext, ChatServiceProtocol, Message, Session


@logging
class ChatService(ChatServiceProtocol):
    __slots__ = ["apps", "repo"]

    def __init__(self, repo: RepositoryProtocol):
        self.apps: dict[str, AIAppProtocol] = {}
        self.repo = repo

    def add_app(self, name: str, app: AIAppProtocol) -> None:
        self.apps[name] = app

    def find_app(self, name: str) -> AIAppProtocol:
        if name not in self.apps:
            raise ValueError("app not found")
        return self.apps[name]

    def create_session(self, app_name: str) -> str:
        app = self.find_app(app_name)
        ulid = str(ULID())
        app.init_session(ulid)
        self.repo.store_session(ulid, app_name)
        return ulid

    def list_sessions(self) -> list[Session]:
        return self.repo.list_sessions()

    def send_message(self, ctx: ChatContext, content: str) -> str:
        session = self.repo.find_session(ctx.session_id)
        if not session:
            raise ValueError("session not found")
        app = self.find_app(session.app_name)
        response = app.invoke(content, session.id)
        return response

    def list_messages(self, session_id: str) -> list[Message]:
        return self.repo.list_messages(session_id)
