from typing import Protocol

from .customer import Customer
from .factory import Factory
from .survey import SurveyFactory


class IIoTRepositoryProtocol(Protocol):
    def list_customers(self) -> list[Customer]:
        """List all customers."""
        ...

    def find_customer(self, customer_id: str) -> Customer:
        """Find a customer by id."""
        ...

    def store_customer(self, customer: Customer):
        """Store a customer."""
        ... 

    def list_factories(self, customer_id: str) -> list[SurveyFactory]:
        """List all factories."""
        ...

    def find_factory(self, factory_id: str) -> Factory:
        """Find a factory by id."""
        ...

    def store_factory(self, factory: Factory):
        """Store a factory."""
        ...

    def list_surveys(self, customer_id: str) -> list[SurveyFactory]:
        """List all survey factories."""
        ...

    def find_survey(self, survey_id: str) -> SurveyFactory:
        """Find a survey factory by id."""
        ...

    def store_survey(self, survey: SurveyFactory):
        """Store a survey factory."""
        ...
