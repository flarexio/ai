from pydantic import BaseModel, Field
from typing import Literal, Optional


class EdgeContext(BaseModel):
    """Edge context information."""
    edge_id: str = Field(description="Unique identifier for the edge services")

IndustryExamples = [
    'automotive', 'semiconductor', 'machine_tool', 'plastic_injection', 'pharmaceutical',
    'food_beverage', 'textile', 'chemical', 'electronics_assembly', 'logistics', 'energy', 'steel', 
]

CustomerStatus = Literal[
    "survey_pending",
    "survey_in_progress",
    "survey_completed",
    "quotation_ready",
    "integration_in_progress",
    "integration_completed",
    "finalized"
]

class Customer(BaseModel):
    """Represents a client organization that owns one or more factories and surveys."""
    customer_id: str = Field(description="Unique identifier for the customer")
    name: str = Field(description="Full name of the customer or organization")
    industry: str = Field(description="Industry sector the customer belongs to", examples=IndustryExamples)
    description: Optional[str] = Field(None, description="General notes or description of this customer")
    status: CustomerStatus = Field("survey_pending", description="Project status of the customer")
    edge_context: Optional[EdgeContext] = Field(None,
        description="Contextual information about the edge services associated with this customer"
    )
