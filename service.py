from ulid import ULID

from decorator import logging
from persistence import RepositoryProtocol
from protocol import AIAppProtocol, ChatContext, ChatServiceProtocol, Message


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

    def create_session(self) -> str:
        ulid = str(ULID())
        self.repo.store_session(ulid)
        return ulid

    def list_sessions(self) -> list[str]:
        return self.repo.list_sessions()

    def send_message(self, ctx: ChatContext, content: str) -> str:
        if not self.repo.find_session(ctx.session_id):
            raise ValueError("session not found")

        app = self.find_app(ctx.app_name)
        response = app.invoke(content, ctx.session_id)
        return response

    def list_messages(self, session_id: str) -> list[Message]:
        return self.repo.list_messages(session_id)
