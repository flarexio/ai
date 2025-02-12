from abc import abstractmethod
from typing import Generic, TypeVar, Protocol


Service = TypeVar("Service")
Request = TypeVar("Request")
Response = TypeVar("Response")

class EndpointProtocol(Protocol, Generic[Request, Response]):
    async def handle(self, request: Request | None = None) -> Response:
        pass


class Endpoint(Generic[Service, Request, Response]):
    def __init__(self, service: Service):
        self.service = service

    @abstractmethod
    async def handle(self, request: Request | None = None) -> Response:
        pass
