import asyncio
import os
import signal

from endpoint import CreateSessionEndpoint, SendMessageEndpoint
from kit import EndpointProtocol
from service import ChatService
from transports.http import HTTPTransport
from transports.nats import NATSTransport

# Setup service
svc = ChatService()

# Setup endpoints
endpoints: dict[str, EndpointProtocol] = {
    "create_session": CreateSessionEndpoint(svc),
    "send_message": SendMessageEndpoint(svc),
}

async def main():
    # Setup transports
    http = HTTPTransport(host="0.0.0.0", port=8000)
    nats = NATSTransport(
        url="wss://nats.flarex.io",
        creds=os.getenv("NATS_CREDS"),
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
