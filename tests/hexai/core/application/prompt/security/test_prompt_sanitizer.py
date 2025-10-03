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
