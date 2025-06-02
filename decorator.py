from typing import Type

from protocol import AIAppProtocol, ChatContext, ChatServiceProtocol, Message, Session


def logging(cls: Type[ChatServiceProtocol]) -> Type[ChatServiceProtocol]:
    """A decorator for logging."""

    class LoggingDecorator(ChatServiceProtocol):
        def __init__(self, *args, **kwargs):
            self.next = cls(*args, **kwargs)

        def add_app(self, name: str, app: AIAppProtocol):
            self.next.add_app(name, app)
            print(f"app added | app={name}")

        def find_app(self, name: str) -> AIAppProtocol:
            app = self.next.find_app(name)
            print(f"app found | app={name}")
            return app

        async def create_session(self, app_name: str) -> str:
            session_id = await self.next.create_session(app_name)
            print(f"session created | app_name={app_name} | session_id={session_id}")
            return session_id

        async def list_sessions(self) -> list[Session]:
            sessions = await self.next.list_sessions()
            print(f"sessions listed | count={len(sessions)}")
            return sessions

        async def send_message(self, ctx: ChatContext, content: str) -> str:
            response = await self.next.send_message(ctx, content)
            print(f"message sent | session_id={ctx.session_id}")
            print(f"Human: {content}")
            print(f"AI: {response}")
            print("-" * 100)
            return response

        async def list_messages(self, session_id: str) -> list[Message]:
            messages = await self.next.list_messages(session_id)
            print(f"messages listed | session_id={session_id} | count={len(messages)}")
            for message in messages:
                print(f"{message.role}: {message.content}")
            print("-" * 100)
            return messages

    return LoggingDecorator
