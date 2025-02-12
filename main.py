import asyncio
import os
import signal

from langgraph.checkpoint.postgres import PostgresSaver

from apps.basic import BasicApp
from endpoint import (
    CreateSessionEndpoint,
    ListSessionsEndpoint,
    SendMessageEndpoint,
    ListMessagesEndpoint,
)
from kit import EndpointProtocol
from persistence import DBRepository
from service import ChatService
from transports.http import HTTPTransport
from transports.nats import NATSTransport


DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

NATS_URL = os.getenv("NATS_URL")
NATS_CREDS = os.getenv("NATS_CREDS")

async def main():
    DB_URI = f"postgres://{DB_USERNAME}:{DB_PASSWORD}@{DB_URL}/ai"
    with PostgresSaver.from_conn_string(DB_URI) as memory:
        # Setup repository
        memory.setup()
        repo = DBRepository(memory)

        # Setup service
        svc = ChatService(repo)

        # Setup apps
        svc.add_app("basic", BasicApp(memory))

        # Setup endpoints
        endpoints: dict[str, EndpointProtocol] = {
            "create_session": CreateSessionEndpoint(svc),
            "list_sessions": ListSessionsEndpoint(svc),
            "send_message": SendMessageEndpoint(svc),
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
        finally:
            # Shutdown transports
            print("Server shutdown")
            await nats.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
