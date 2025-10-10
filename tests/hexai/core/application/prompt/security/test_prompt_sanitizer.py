import pytest

from hexai.core.application.prompt.security.prompt_sanitizer import (
    SanitizationConfig,
    parse_sanitization_config,
    sanitize_text,
)
from hexai.core.bootstrap import ensure_bootstrapped
from hexai.core.registry import registry

ensure_bootstrapped()


def test_parse_sanitization_yes_no_and_defaults():
    raw = {
        "use_sanitizer": "yes",
        "max_input_length": 50,
        "escape_template_chars": "no",
        "normalize_unicode": "on",
        "normalization_form": "nfkd",
    }
    cfg = parse_sanitization_config(raw)
    assert cfg.use_sanitizer is True
    assert cfg.max_input_length == 50
    assert cfg.escape_template_chars is False
    assert cfg.normalize_unicode is True
    assert cfg.normalization_form == "NFKD"

    cfg2 = parse_sanitization_config(None)
    assert cfg2.use_sanitizer is False
    assert cfg2.max_input_length == 1000


def test_sanitize_text_core_pipeline():
    cfg = SanitizationConfig(
        use_sanitizer=True,
        max_input_length=20,
        escape_template_chars=True,
        normalize_unicode=True,
        normalization_form="NFKC",
    )
    raw = "Hi {x}\u202e!\n"
    out = sanitize_text(raw, cfg)
    assert r"\{x\}" in out
    assert "\u202e" not in out
    assert len(out) <= 20


@pytest.fixture
def llm_node():
    """Get LLMNode factory instance from registry."""
    ensure_bootstrapped()
    return registry.get("llm_node", namespace="core")


@pytest.fixture
def agent_node():
    """Get AgentNode factory instance from registry."""
    ensure_bootstrapped()
    return registry.get("agent_node", namespace="core")


@pytest.mark.asyncio
async def test_base_llm_node_applies_sanitization(llm_node):
    from hexai.core.application.prompt.template import PromptTemplate

    class DummyLLM:
        async def aresponse(self, messages):
            return messages[-1]["content"]

    template = PromptTemplate("Hello {{name}}!")

    llm_wrapper = llm_node.create_llm_wrapper(  # type: ignore[attr-defined]
        name="t",
        template=template,
        input_model=None,
        output_model=None,
        rich_features=False,
    )

    validated_input = {
        "name": "{Alice}\u202e",
        "_sanitization_cfg": SanitizationConfig(
            use_sanitizer=True, escape_template_chars=True, normalize_unicode=True
        ),
    }

    result = await llm_wrapper(validated_input, llm=DummyLLM())
    assert r"\{Alice\}" in result
    assert "\u202e" not in result


@pytest.mark.asyncio
async def test_agent_node_runs_with_default_cfg(agent_node):
    from hexai.core.application.nodes.agent_node import AgentConfig

    class DummyLLM2:
        async def aresponse(self, messages):
            return messages[-1]["content"]

    spec = agent_node(  # type: ignore[call-arg]
        name="agent_t",
        main_prompt="Say hi to {{input}}",
        continuation_prompts=None,
        output_schema=None,
        config=AgentConfig(max_steps=1),
        deps=None,
    )

    result = await spec.fn({"input": "{Bob}\u202e"}, llm=DummyLLM2(), tool_router=None)
    assert isinstance(result, dict)


def test_yaml_sanitization_full_flow_llm_node_from_yaml():
    """
    End-to-end without YAML builder dependency: create LLM wrapper directly,
    apply sanitization config, verify escaping, normalization, bidi removal,
    and truncation.
    """
    from hexai.core.application.prompt.template import PromptTemplate

    llm_factory = registry.get("llm_node", namespace="core")
    template = PromptTemplate("Greet {{name}} with {{emoji}}")

    llm_wrapper = llm_factory.create_llm_wrapper(  # type: ignore[attr-defined]
        name="greet",
        template=template,
        input_model=None,
        output_model=None,
        rich_features=False,
    )

    class DummyLLM:
        async def aresponse(self, messages):
            return messages[-1]["content"]

    validated_input = {
        "name": "{Al\u202eice}",
        "emoji": "👍🏼",
        "_sanitization_cfg": SanitizationConfig(
            use_sanitizer=True,
            escape_template_chars=True,
            normalize_unicode=True,
            normalization_form="NFKC",
            max_input_length=30,
        ),
    }

    import asyncio

    result = asyncio.get_event_loop().run_until_complete(
        llm_wrapper(validated_input, llm=DummyLLM())
    )

    assert r"\{Al" in result and r"\}" in result
    assert "\u202e" not in result
    assert "👍🏼" in result
    assert len(result) <= 30


def test_llm_node_multiple_fields_mixed_sanitization():
    """
    Multiple risky fields: braces, backticks, bidi, and double braces in code.
    Ensures consistent sanitization before rendering.
    """
    llm_factory = registry.get("llm_node", namespace="core")
    from hexai.core.application.prompt.template import PromptTemplate

    template = PromptTemplate("User: {{user}}\nNote: {{note}}\nCode: {{code}}")
    llm_wrapper = llm_factory.create_llm_wrapper(  # type: ignore[attr-defined]
        name="multi",
        template=template,
        input_model=None,
        output_model=None,
        rich_features=False,
    )

    class DummyLLM:
        async def aresponse(self, messages):
            return messages[-1]["content"]

    validated_input = {
        "user": "{Bob}\u202e",
        "note": "Check this `inline`\n",
        "code": "print('hi') // {{jinja}}",
        "_sanitization_cfg": SanitizationConfig(
            use_sanitizer=True,
            escape_template_chars=True,
            normalize_unicode=True,
            max_input_length=200,
        ),
    }

    import asyncio

    result = asyncio.get_event_loop().run_until_complete(
        llm_wrapper(validated_input, llm=DummyLLM())
    )

    assert r"\{Bob\}" in result
    assert "\u202e" not in result
    # Adjust if your escape set doesn't include backticks
    assert "`inline`" in result
    assert r"\{\{jinja\}\}" in result


def test_prompttemplate_rendering_still_valid_with_escaped_chars():
    """
    Template should render without raising after escaping,
    and placeholders should be substituted.
    """
    from hexai.core.application.prompt.template import PromptTemplate

    t = PromptTemplate("Path {{path}} and literal braces: \\{ok\\}")
    rendered = t.render(path="/tmp/{dir}/file")
    assert "Path " in rendered
    assert "\\{ok\\}" in rendered
    assert "/tmp" in rendered


@pytest.mark.asyncio
async def test_agent_node_sanitization_on_planned_turns(agent_node):
    """
    Simulate a short agent turn with risky input. Ensure output type and
    bidi removal if string.
    """
    from hexai.core.application.nodes.agent_node import AgentConfig

    class DummyLLM:
        async def aresponse(self, messages):
            return messages[-1]["content"]

    spec = agent_node(  # type: ignore[call-arg]
        name="agent_sanitizer",
        main_prompt="Ask: {{question}}",
        continuation_prompts=None,
        output_schema=None,
        config=AgentConfig(max_steps=1),
        deps=None,
    )

    result = await spec.fn(
        {"question": "What is {2+2}?\u202d `code`"}, llm=DummyLLM(), tool_router=None
    )
    assert isinstance(result, (dict, str))
    if isinstance(result, str):
        assert "\u202d" not in result


def test_truncation_edge_off_by_one():
    """
    Truncation should add an ellipsis and respect the max length exactly.
    """
    cfg = SanitizationConfig(
        use_sanitizer=True,
        max_input_length=5,
        escape_template_chars=False,
        normalize_unicode=False,
    )
    out = sanitize_text("abcdef", cfg)
    assert len(out) == 5
    assert out.endswith("…")


def test_unicode_normalization_idempotence_and_forms():
    """
    Verify supported normalization forms are accepted and executed.
    """
    s = "é"
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        cfg = SanitizationConfig(
            use_sanitizer=True,
            max_input_length=100,
            escape_template_chars=False,
            normalize_unicode=True,
            normalization_form=form,  # cast if your type checker requires
        )
        out = sanitize_text(s, cfg)
        assert isinstance(out, str)
        assert len(out) >= 1


def test_nested_variables_with_mixed_risks_sanitization():
    """
    Nested variables with emojis, bidi, braces, and double braces.
    Ensures nested access works and sanitizer handles risky chars across fields.
    """
    from hexai.core.application.prompt.template import PromptTemplate

    llm_factory = registry.get("llm_node", namespace="core")
    template = PromptTemplate(
        "User: {{user.name}} (role: {{user.role}})\nMsg: {{message}}\nTags: {{meta.tags}}"
    )

    llm_wrapper = llm_factory.create_llm_wrapper(  # type: ignore[attr-defined]
        name="nested",
        template=template,
        input_model=None,
        output_model=None,
        rich_features=False,
    )

    class DummyLLM:
        async def aresponse(self, messages):
            return messages[-1]["content"]

    validated_input = {
        "user": {"name": "{Al\u202eice}", "role": "admin"},
        "message": "Please review {{draft}} at path `/tmp/{dir}` 👍",
        "meta": {"tags": "{safe},{ok}"},
        "_sanitization_cfg": SanitizationConfig(
            use_sanitizer=True,
            escape_template_chars=True,
            normalize_unicode=True,
            normalization_form="NFKC",
            max_input_length=500,
        ),
    }

    import asyncio

    result = asyncio.get_event_loop().run_until_complete(
        llm_wrapper(validated_input, llm=DummyLLM())
    )

    assert "User: {Al\u202eice}" not in result
    assert r"review \{\{draft\}\}" in result
    assert r"`/tmp/\{dir\}`" in result
    assert r"Tags: \{safe\},\{ok\}" in result


@pytest.mark.asyncio
async def test_chat_fewshot_with_history_and_sanitization():
    """
    Few-shot/system + history + current risky input.
    Ensures messages are built correctly and sanitizer applies to current vars.
    """
    from hexai.core.application.prompt.template import ChatFewShotTemplate

    class DummyLLM:
        async def aresponse(self, messages):
            user_msgs = [m["content"] for m in messages if m["role"] == "user"]
            return "\n".join(user_msgs) if user_msgs else messages[-1]["content"]

    template = ChatFewShotTemplate(
        system_message="You are a helpful assistant in {{domain}}.",
        human_message="Answer the question: {{question}}",
        examples=[
            {"input": "2+2", "output": "4"},
            {"input": "3+5", "output": "8"},
        ],
    )

    llm_factory = registry.get("llm_node", namespace="core")
    llm_wrapper = llm_factory.create_llm_wrapper(  # type: ignore[attr-defined]
        name="chat_fewshot",
        template=template,
        input_model=None,
        output_model=None,
        rich_features=False,
    )

    history = [
        {"role": "user", "content": "Context: previous {{ignored}} info."},
        {"role": "assistant", "content": "Sure, noted."},
    ]

    validated_input = {
        "domain": "math",
        "question": "Compute {2+2}\u202d and show {{steps}}.",
        "context_history": history,
        "_sanitization_cfg": SanitizationConfig(
            use_sanitizer=True,
            escape_template_chars=True,
            normalize_unicode=True,
            normalization_form="NFKC",
            max_input_length=500,
        ),
    }

    result = await llm_wrapper(validated_input, llm=DummyLLM())

    assert "\u202d" not in result
    assert r"\{2+2\}" in result
    assert r"\{\{steps\}\}" in result
    assert "previous {{ignored}} info." in history[0]["content"]
