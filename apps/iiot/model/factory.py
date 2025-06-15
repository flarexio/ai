from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, List


AccessMode = Literal["read_only", "write_only", "read_write"]

class Point(BaseModel):
    """Represents any interaction point of a controller: signal, status, or parameter.
    
    The 'options' field must be configured based on the controller's driver schema.
    Use Schema tool to query point configuration requirements for the specific driver.
    """
    name: str = Field(
        description="Name of the point", 
        examples=["CurrentTemperature", "TargetTemperature"],
    )
    display: str = Field(
        description="Human-readable name of the point", 
        examples=["Target Temperature", "Current Temperature"],
    )
    type: Literal["bool", "int", "float", "string"] = Field(description="Data type of the point")
    access: AccessMode = Field("read_only", description="Access mode for the point")
    unit: str = Field(
        description="Measurement unit, if applicable",
        examples=["°C", "°F", "m/s", "rpm", "kW", "V", "A", "Pa", "mm"],
    )
    options: Dict[str, Any] = Field({}, 
        description="""Driver-specific point configuration.
        MUST be configured based on the controller's driver schema.
        Query Schema tool with: Schema.get_point_schema(driver_name) to get required parameters.""",
        examples=[
            { "register": "coil", "address": 0, "offset": 0, "quantity": 1, "unit": "bit", "data_type": "bool" },
            { 
                "register": "holding", "address": 0, "offset": 0, "quantity": 2, 
                "unit": "dword", "data_type": "float", "byte_order": "big_endian", "word_order": "little_endian", 
            },
            { "node_id": "ns=1;i=1001", "data_type": "double" },
            { "device": "D", "address": 400, "offset": 0, "quantity": 2, "unit": "dword", "data_type": "float" }, 
        ]
    )


ControllerType = Literal[
    "plc",
    "cnc",
    "sensor_node",
    "gateway",
    "edge_device",
    "remote_io",
]

VendorExamples = [
    "advantech",
    "delta",
    "fanuc",
    "icp_das",
    "mitsubishi",
    "moxa",
    "omron",
    "siemens",
]

ProtocolExamples = [
    "modbus_tcp",
    "modbus_rtu",
    "modbus_ascii",
    "mqtt",
    "opcua",
    "mc",
]

DriverExamples = [
    "modbus",
    "opcua"
    "mqtt"
    "mc",
]

class Controller(BaseModel):
    """Represents a controller (PLC, CNC, Sensor Node) managing system points.
    
    Before configuring this controller, use the Schema tool to query the specific 
    driver schema based on the 'driver' field, then populate the 'options' field 
    with the required driver-specific configuration parameters.
    """
    controller_id: str = Field(description="Unique identifier for the controller")
    type: ControllerType = Field(description="Type of controller")
    vendor: str = Field(
        description="Manufacturer of the controller",
        examples=VendorExamples,
    )
    model: str = Field(description="Model number or name of the controller")
    protocol: str = Field(
        description="Communication protocol used by the controller",
        examples=ProtocolExamples,
    )
    driver: str = Field(
        description="""Driver or library used for communication. 
        IMPORTANT: Before configuring options, query the Schema tool with this driver name 
        to get the required configuration parameters.""", 
        examples=DriverExamples,
    )
    address: str = Field(description="Network address or identifier of the controller")
    points: List[Point] = Field([], description="All logical points managed by the controller")
    options: Dict[str, Any] = Field({},
        description="""Driver-specific configuration options.
        MUST be populated based on the schema retrieved from Schema tool using the 'driver' field.
        Query format: Schema.get_driver_schema(driver_name)""",
        examples=[
            { "slave_id": 1 },
            { "endpoint_url": "opc.tcp://192.168.1.100:4840", "security_mode": "None" },
        ]
    )


class MaintenanceLog(BaseModel):
    """Represents a maintenance record for a machine."""
    log_id: str = Field(description="Unique identifier for the maintenance log")
    timestamp: str = Field(description="Date and time of the maintenance activity")
    description: str = Field(description="Description of the maintenance work")
    technician: str = Field(description="Technician who performed the maintenance")


MachineStatus = Literal[
    "idle",           # The machine is not currently in operation
    "starting",       # The machine is in the process of starting up
    "running",        # The machine is actively producing
    "paused",         # The machine is temporarily halted
    "stopped",        # The machine is not operational
    "fault",          # The machine has encountered an error
    "maintenance",    # The machine is undergoing maintenance
    "emergency_stop", # The machine has been stopped due to an emergency
]

class Machine(BaseModel):
    """Represents a machine within a production line."""
    machine_id: str = Field(description="Unique identifier for the machine")
    name: str = Field(description="Human-readable name of the machine")
    status: MachineStatus = Field("idle", description="Current operational state of the machine")
    controllers: List[Controller] = Field([], description="Controllers attached to the machine")
    maintenance_logs: List[MaintenanceLog] = Field([], description="Maintenance logs related to this machine")


ProductionLineStatus = Literal[
    "idle",        # The line is not currently in operation
    "running",     # The line is actively producing
    "paused",      # The line is temporarily halted
    "stopped",     # The line is not operational
    "fault",       # The line has encountered an error
    "maintenance", # The line is undergoing maintenance
    "startup",     # The line is in the process of starting up
    "shutdown",    # The line is in the process of shutting down
]

class ProductionLine(BaseModel):
    """Represents a production line composed of multiple machines."""
    line_id: str = Field(description="Unique identifier for the production line")
    name: str = Field(description="Name of the production line")
    status: ProductionLineStatus = Field("idle", description="Current operational status")
    machines: List[Machine] = Field([], description="List of machines on the production line")

class Factory(BaseModel):
    """Represents a factory site composed of production lines."""
    factory_id: str = Field(description="Unique identifier for the factory")
    name: str = Field(description="Name of the factory")
    location: str = Field(description="Geographical location of the factory")
    production_lines: List[ProductionLine] = Field([], description="List of production lines within the factory")
    customer_id: str = Field(description="ID of the customer this survey belongs to")
