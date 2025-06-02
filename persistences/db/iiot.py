import pymongo

from apps.iiot.model import Customer, Factory, IIoTRepositoryProtocol, SurveyFactory


class IIoTMongoDBRepository(IIoTRepositoryProtocol):
    __slot__ = ("customers", "factories", "survey_factories")

    def __init__(self, uri: str):
        client = pymongo.MongoClient(uri)
        db = client["iiot"]
        self.customers = db["customers"]
        self.factories = db["factories"]
        self.survey_factories = db["survey_factories"]

    def list_customers(self) -> list[Customer]:
        customers = []
        for doc in self.customers.find():
            doc["customer_id"] = doc.pop("_id")
            customer = Customer.model_validate(doc)
            customers.append(customer)
        return customers

    def find_customer(self, customer_id: str) -> Customer:
        doc = self.customers.find_one({"_id": customer_id})
        if doc is None:
            return Customer()
        doc["customer_id"] = doc.pop("_id")
        customer = Customer.model_validate(doc)
        return customer

    def store_customer(self, customer: Customer):
        doc = customer.model_dump()
        doc["_id"] = doc.pop("customer_id")
        self.customers.replace_one(
            {"_id": customer.customer_id},
            doc,
            upsert=True,
        )

    def list_factories(self, customer_id: str) -> list[Factory]:
        factories = []
        for doc in self.factories.find({"customer_id": customer_id}):
            doc["factory_id"] = doc.pop("_id")
            factory = Factory.model_validate(doc)
            factories.append(factory)
        return factories

    def find_factory(self, factory_id: str) -> Factory:
        doc = self.factories.find_one({"_id": factory_id})
        if doc is None:
            return Factory()
        doc["factory_id"] = doc.pop("_id")
        factory = Factory.model_validate(doc)
        return factory

    def store_factory(self, factory: Factory):
        doc = factory.model_dump()
        doc["_id"] = doc.pop("factory_id")
        self.factories.replace_one(
            {"_id": factory.factory_id},
            doc,
            upsert=True,
        )

    def list_surveys(self, customer_id: str) -> list[SurveyFactory]:
        surveys = []
        for doc in self.survey_factories.find({"customer_id": customer_id}):
            doc["survey_id"] = doc.pop("_id")
            survey = SurveyFactory.model_validate(doc)
            surveys.append(survey)
        return surveys

    def find_survey(self, survey_id: str) -> SurveyFactory:
        doc = self.survey_factories.find_one({"_id": survey_id})
        if doc is None:
            return SurveyFactory()
        doc["survey_id"] = doc.pop("_id")
        survey = SurveyFactory.model_validate(doc)
        return survey

    def store_survey(self, survey: SurveyFactory):
        doc = survey.model_dump()
        doc["_id"] = doc.pop("survey_id")
        self.survey_factories.replace_one(
            {"_id": survey.survey_id},
            doc,
            upsert=True,
        )
