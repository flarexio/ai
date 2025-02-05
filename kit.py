from abc import abstractmethod
from typing import Generic, TypeVar, Protocol

Request = TypeVar("Request")
Response = TypeVar("Response")

class EndpointProtocol(Protocol, Generic[Request, Response]):
    async def handle(self, request: Request | None = None) -> Response:
        pass


class Endpoint(Generic[Request, Response]):
    def __init__(self, service: any):
        self.service = service

    @abstractmethod
    async def handle(self, request: Request | None = None) -> Response:
        pass
