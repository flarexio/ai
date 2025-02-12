from typing import Type

from protocol import AIAppProtocol, ChatServiceProtocol, Message


def logging(cls: Type[ChatServiceProtocol]) -> Type[ChatServiceProtocol]:
    """A decorator for logging."""

    class LoggingDecorator(ChatServiceProtocol):
        def __init__(self, *args, **kwargs):
            self.next = cls(*args, **kwargs)

        def add_app(self, name: str, app: AIAppProtocol) -> None:
            self.next.add_app(name, app)
            print(f"app added | app={name}")

        def find_app(self, name: str) -> AIAppProtocol:
            app = self.next.find_app(name)
            print(f"app found | app={name}")
            return app

        def create_session(self) -> str:
            session_id = self.next.create_session()
            print(f"session created | session_id={session_id}")
            return session_id

        def list_sessions(self) -> list[str]:
            sessions = self.next.list_sessions()
            print(f"sessions listed | count={len(sessions)}")
            return sessions

        def send_message(self, content: str, session_id: str) -> str:
            response = self.next.send_message(content, session_id)
            print(f"message sent | session_id={session_id}")
            print(f"Human: {content}")
            print(f"AI: {response}")
            print("-" * 100)
            return response

        def list_messages(self, session_id: str) -> list[Message]:
            messages = self.next.list_messages(session_id)
            print(f"messages listed | session_id={session_id} | count={len(messages)}")
            for message in messages:
                print(f"{message.role}: {message.content}")
            print("-" * 100)
            return messages

    return LoggingDecorator
