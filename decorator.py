from typing import AsyncIterator, Type

from protocol import AIAppProtocol, ChatContext, ChatServiceProtocol, Message, MessageChunk, Session


def logging(cls: Type[ChatServiceProtocol]) -> Type[ChatServiceProtocol]:
    """A decorator for logging."""

    class LoggingDecorator(ChatServiceProtocol):
        def __init__(self, *args, **kwargs):
            self.next = cls(*args, **kwargs)

        def add_app(self, name: str, app: AIAppProtocol):
            self.next.add_app(name, app)
            print(f"app added | app={name}")

        async def list_apps(self) -> list[str]:
            apps = await self.next.list_apps()
            print(f"apps listed | count={len(apps)}")
            return apps

        async def find_app(self, name: str) -> AIAppProtocol:
            app = await self.next.find_app(name)
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

        async def stream_message(self, ctx: ChatContext, content: str) -> AsyncIterator[MessageChunk]:
            print(f"Human: {content}")
            print(f"streaming message | session_id={ctx.session_id}")
            stream: AsyncIterator[MessageChunk] = self.next.stream_message(ctx, content)
            async for chunk in stream:
                if chunk.is_new is True:
                    print("\n\n")
                    print("---")
                    print(f"Nodes: {chunk.nodes}")
                    print(f"{chunk.role.value}: ", end="", flush=True)

                if chunk.content != '':
                    print(chunk.content, end="", flush=True)

                if chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        if tool_call.id:
                            print("\n")
                            print(f"[Tool Call: {tool_call.id}]")
                            print(f" Name: {tool_call.name}")
                            print(" Args: ", end="", flush=True)
                        if tool_call.args != '':
                            print(tool_call.args, end="", flush=True)

                yield chunk
            print("\nStreaming complete.")
            print("-" * 100)

        async def list_messages(self, session_id: str) -> list[Message]:
            messages = await self.next.list_messages(session_id)
            print(f"messages listed | session_id={session_id} | count={len(messages)}")
            for message in messages:
                print(f"{message.role}: {message.content}")
            print("-" * 100)
            return messages

    return LoggingDecorator
