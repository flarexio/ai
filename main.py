import asyncio
import os
import signal

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from apps.basic import BasicAIApp
from apps.iiot import IIoTAIApp
from endpoint import (
    ListAppsEndpoint,
    CreateSessionEndpoint,
    ListSessionsEndpoint,
    SendMessageEndpoint,
    StreamMessageEndpoint,
    ListMessagesEndpoint,
)
from kit import EndpointProtocol
from persistences.db import ChatDatabaseRepository, IIoTMongoDBRepository
from service import ChatService
from transports.http import HTTPTransport
from transports.nats import NATSTransport

DB_ADDRESS = os.getenv("DB_HOST") + ":" + os.getenv("DB_PORT")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_URL= f"postgres://{DB_USERNAME}:{DB_PASSWORD}@{DB_ADDRESS}/ai"

NATS_URL = os.getenv("NATS_URL")
NATS_CREDS = os.getenv("NATS_CREDS")

MONGO_URL = os.getenv("MONGO_HOST") + ":" + os.getenv("MONGO_PORT")
MONGO_AUTH = os.getenv("MONGO_USERNAME") + ":" + os.getenv("MONGO_PASSWORD")
MONGO_URI = f"mongodb://{MONGO_AUTH}@{MONGO_URL}"


async def main():
    try:
        async with (
            AsyncPostgresSaver.from_conn_string(DB_URL) as memory, 
            AsyncPostgresStore.from_conn_string(DB_URL, index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
            }) as store,
            MultiServerMCPClient(
                {
                    "iiot": {
                        "command": "iiot_mcp",
                        "args": [
                            "--creds", "/home/ar0660/.flarex/iiot/user.creds",
                        ],
                        "transport": "stdio",
                    },
                    "time": {
                        "command": "uvx",
                        "args": [
                            "mcp-server-time",
                            "--local-timezone=Asia/Taipei",
                        ],
                        "transport": "stdio",
                    }
                },
            ) as mcp_client,
        ):
            # Create the chat service
            await memory.setup()
            await store.setup()

            # Add repositories
            chats = ChatDatabaseRepository(memory)
            await chats.migrate()

            # Create the chat service
            svc = ChatService(chats)

            toolkit = mcp_client.server_name_to_tools

            # Add AI apps
            svc.add_app("basic", BasicAIApp(memory, store, toolkit))

            # IIoT app
            iiot_repo = IIoTMongoDBRepository(MONGO_URI)
            svc.add_app("iiot", IIoTAIApp(memory, iiot_repo, toolkit))

            # Setup endpoints
            endpoints: dict[str, EndpointProtocol] = {
                "list_apps": ListAppsEndpoint(svc),
                "create_session": CreateSessionEndpoint(svc),
                "list_sessions": ListSessionsEndpoint(svc),
                "send_message": SendMessageEndpoint(svc),
                "stream_message": StreamMessageEndpoint(svc),
                "list_messages": ListMessagesEndpoint(svc),
            }

            # Setup transports
            http = HTTPTransport(host="0.0.0.0", port=8000)
            nats = NATSTransport(
                url=NATS_URL,
                creds=NATS_CREDS,
            )

            # Set endpoints
            http.set_endpoints(endpoints)
            nats.set_endpoints(endpoints)

            # Setup signal handler
            stop = asyncio.Event()

            def signal_handler():
                print("Received signal to stop")
                stop.set()

            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_running_loop().add_signal_handler(sig, signal_handler)

            try:
                # Start transports
                await asyncio.gather(
                    http.serve(),
                    nats.serve(),
                    stop.wait(),
                )
            except asyncio.CancelledError:
                print("Async tasks cancelled, shutting down gracefully...")
            finally:
                # Shutdown transports
                print("Server shutdown")
                await nats.shutdown()
    except* ProcessLookupError:
        print("Suppressed ProcessLookupError during MCP client shutdown.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user")
