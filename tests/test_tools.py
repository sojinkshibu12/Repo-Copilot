from core.tools import TOOL_DEFINITIONS


def test_all_tools_have_name():
    for tool in TOOL_DEFINITIONS:
        assert "name" in tool
        assert tool["name"]


def test_all_tools_have_description():
    for tool in TOOL_DEFINITIONS:
        assert "description" in tool
        assert tool["description"]


def test_all_tools_have_input_schema():
    for tool in TOOL_DEFINITIONS:
        assert "input_schema" in tool
        assert "properties" in tool["input_schema"]


def test_classify_tool_has_enum():
    classify = next(t for t in TOOL_DEFINITIONS if t["name"] == "classify_issue")
    props = classify["input_schema"]["properties"]
    assert "classification" in props
    assert props["classification"]["type"] == "string"
    assert "enum" in props["classification"]


def test_no_duplicate_tool_names():
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert len(names) == len(set(names))
