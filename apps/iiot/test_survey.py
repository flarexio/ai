import pytest

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from trustcall import create_extractor

from .model import Customer, SurveyFactory


@pytest.mark.skip(reason="costly")
def test_customer_extractor_with_industry():
    model = init_chat_model("openai:gpt-5-mini")
    extractor = create_extractor(model, tools=[Customer], tool_choice="Customer")

    system_prompt = """
    Reflect on the following interaction.
    
    Use the provided tools to retain any necessary memories about the customer.
    
    Use parallel tool calling to handle updates and insertions simultaneously.

    IMPORTANT: Always set "json_doc_id": "Customer"
    """

    input = Customer(
        customer_id="C123", 
        name="Test Corp", 
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="這個客戶是在做塑膠射出的"),
    ]

    result = extractor.invoke({
        "messages": messages,
        "existing": {
            "Customer": input.model_dump(),
        }
    })

    output = Customer.model_validate(result["responses"][0])

    assert output.industry == "plastic_injection"


@pytest.mark.skip(reason="costly")
def test_customer_extractor_with_edge_id():
    model = init_chat_model("openai:gpt-5-mini")
    extractor = create_extractor(model, tools=[Customer], tool_choice="Customer")

    system_prompt = """
    Reflect on the following interaction.
    
    Use the provided tools to retain any necessary memories about the customer.
    
    Use parallel tool calling to handle updates and insertions simultaneously.

    IMPORTANT: Always set "json_doc_id": "Customer"
    """

    input = Customer(
        customer_id="C123", 
        name="Test Corp", 
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="客戶的邊緣服務，可以使用 EdgeID E1234 連線到它"),
    ]

    result = extractor.invoke({
        "messages": messages,
        "existing": {
            "Customer": input.model_dump(),
        }
    })

    output = Customer.model_validate(result["responses"][0])

    assert output.edge_context.edge_id == "E1234"

@pytest.mark.skip(reason="costly")
def test_survey_extractor():
    model = init_chat_model("openai:gpt-5-mini")
    extractor = create_extractor(model, tools=[SurveyFactory], tool_choice="SurveyFactory")

    system_prompt = """
    Reflect on the following interaction.
    
    Use the provided tools to retain any necessary memories about the survey factory.
    
    Use parallel tool calling to handle updates and insertions simultaneously.

    IMPORTANT:
    1. Always set "json_doc_id": "SurveyFactory"
    2. If the user input does not specify an area or production line, assign the assets to a common area.
    """

    input = SurveyFactory(
        survey_id="S123", 
        factory_name="Test Factory", 
        survey_date="2025-05-20", 
        engineer="Mirror Lin", 
        customer_id="C123",
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="客戶有金瑛發的殺菌釜2台，是採用三菱PLC FX-3G，預計要採集狀態、溫度、壓力及時間"),
    ]

    result = extractor.invoke({
        "messages": messages,
        "existing": {
            "SurveyFactory": input.model_dump(),
        },
    })

    output = SurveyFactory.model_validate(result["responses"][0])

    assert len(output.areas) == 1
    assert len(output.areas[0].assets) == 2

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="客戶有金瑛發的殺菌釜2台，是採用三菱PLC FX-3G，預計要採集狀態、溫度、壓力及時間"),
        AIMessage(content="已完成"),
        HumanMessage(content="再來有2台金瑛發的萃取釜，其他資訊都一樣"),
    ]

    result = extractor.invoke({
        "messages": messages,
        "existing": {
            "SurveyFactory": output.model_dump(),
        },
    })

    output = SurveyFactory.model_validate(result["responses"][0])

    assert len(output.areas) == 1
    assert len(output.areas[0].assets) == 4
