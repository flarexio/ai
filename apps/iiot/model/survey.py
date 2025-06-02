from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal


class SurveyPoint(BaseModel):
    """Represents a raw signal or control point discovered during a field survey."""
    name: str = Field(description="Field label or tag name of the point")
    description: Optional[str] = Field(None, description="Additional explanation or observed function")
    address: Optional[str] = Field(None, description="Observed or assumed address, e.g., I0.0, 40001")
    signal_type: Optional[Literal['Digital', 'Analog']] = Field(None, description="Type of signal")
    extra_data: Dict[str, Any] = Field({}, description="Additional metadata collected during the survey")
    is_ignored: bool = Field(False, description="Whether this point should be ignored in the model")

class SurveyController(BaseModel):
    """Represents a controller or device observed during a field survey."""
    name: str = Field(description="Field label or tag name of the controller")
    type: str = Field(
        description="Type of controller observed", 
        examples=["PLC", "CNC", "Sensor Node", "Gateway"],
    )
    vendor: Optional[str] = Field(None, description="Manufacturer of the controller")
    model: Optional[str] = Field(None, description="Model number or name of the controller")
    protocol: Optional[str] = Field(None, description="Communication protocol used by the controller")
    points: List[SurveyPoint] = Field([], description="List of points discovered in this controller")
    extra_data: Dict[str, Any] = Field({}, description="Additional metadata collected during the survey")
    is_ignored: bool = Field(False, description="Whether this controller should be ignored in the model")

class SurveyMachine(BaseModel):
    """
    Represents a machine template observed during a field survey.
    
    ⚠️ IMPORTANT: This represents a TEMPLATE, not individual machines.
    
    For multiple identical machines:
    - ✅ CORRECT: 2 identical boilers → ONE SurveyMachine with quantity=2
    - ❌ WRONG: 2 identical boilers → TWO separate SurveyMachine entries
    
    Use `planned_machine_ids` to specify individual IDs for each machine instance.
    """
    name: str = Field(description="Machine type/model name (e.g., 'Boiler', 'CNC Machine')")
    description: Optional[str] = Field(None, description="Additional explanation or observed function")
    quantity: int = Field(1, description="Number of identical machines of this type")
    planned_machine_ids: List[str] = Field([],
        description="Individual IDs for each machine instance. Length should match quantity.",
        examples=["For 2 boilers: ['Boiler-A1', 'Boiler-A2']"]
    )
    controllers: List[SurveyController] = Field([], description="List of controllers discovered in this machine type")
    extra_data: Dict[str, Any] = Field({}, description="Additional metadata collected during the survey")
    is_ignored: bool = Field(False, description="Whether this machine template should be ignored in the model")

class SurveyArea(BaseModel):
    """
    A logical or physical area within the site surveyed.

    Note:
    - The `machines` field contains machine templates, not individual instances.
    - If multiple identical machines are present (e.g., "2 boilers"), create **one** SurveyMachine entry with `quantity=2`.
    - Do **not** create multiple SurveyMachine entries for identical machines.
    """
    code: Optional[str] = Field(None, description="Logical area code (e.g. A, B, C)")
    name: str = Field(description="Area name such as Line 1, Warehouse Zone")
    machines: List[SurveyMachine] = Field([], 
        description="List of machine templates in the area. Each template may represent multiple identical machines via the `quantity` field.")
    is_production_line: Optional[bool] = Field(None, 
        description="Whether this area is expected to become a production line in the factory model"
    )
    extra_data: Dict[str, Any] = Field({}, description="Additional metadata collected during the survey")
    is_ignored: bool = Field(False, description="Whether this area should be ignored in the model")

class SurveyFactory(BaseModel):
    """Top-level container for a full survey on a factory site."""
    survey_id: str = Field(description="Unique identifier for the survey")
    factory_name: str = Field(description="Client factory name")
    survey_date: str = Field(
        description="Date of the site visit in RFC3339 format with timezone", 
        examples=["2025-05-20T10:00:00+08:00"])
    areas: List[SurveyArea] = Field([], description="List of surveyed areas")
    extra_data: Dict[str, Any] = Field({}, description="Additional metadata collected during the survey")
    customer_id: str = Field(description="ID of the customer this survey belongs to")
