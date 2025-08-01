"""
Example 10: Agent Nodes - Function-First Tool Architecture.

This example demonstrates integration with the existing tool architecture:
- Define tools as actual functions (real by default)
- Auto-generate ToolDefinitions for existing ToolDescriptionManager
- Integrate with EnhancedToolParser and ToolRouter port
- Seamless integration with agent workflow
"""

import asyncio
from typing import Any

from hexai.adapters.function_tool_router import FunctionBasedToolRouter
from hexai.adapters.mock.mock_llm import MockLLM
from hexai.app.application.nodes.agent_node import AgentConfig, ReActAgentNode
from hexai.app.application.orchestrator import Orchestrator
from hexai.app.domain.dag import DirectedGraph
from hexai.validation import coerce_validator


# Define real tool functions with proper type hints
async def search_medical_literature(query: str, database: str = "pubmed") -> dict[str, Any]:
    """Search medical literature for research papers."""
    await asyncio.sleep(0.1)  # Simulate API call
    return {
        "query": query,
        "database": database,
        "papers": [
            {"title": f"Medical study on {query}", "relevance": 0.95, "year": 2023},
            {"title": f"Clinical trial for {query}", "relevance": 0.87, "year": 2022},
        ],
        "total_found": 2,
    }


async def calculate_risk_score(factors: str, patient_age: int = 65) -> dict[str, Any]:
    """Calculate medical risk score based on factors."""
    await asyncio.sleep(0.05)

    # Simple risk calculation for demo
    base_risk = 0.1
    age_factor = patient_age / 100
    factor_count = len(factors.split(","))

    risk_score = min(base_risk + age_factor + (factor_count * 0.1), 1.0)

    return {
        "factors": factors,
        "patient_age": patient_age,
        "risk_score": round(risk_score, 3),
        "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
    }


async def generate_treatment_plan(condition: str, risk_level: str = "medium") -> dict[str, Any]:
    """Generate treatment recommendations."""
    await asyncio.sleep(0.08)

    treatments = {
        "low": ["monitoring", "lifestyle changes"],
        "medium": ["medication", "regular checkups", "lifestyle changes"],
        "high": ["immediate intervention", "specialist consultation", "medication"],
    }

    return {
        "condition": condition,
        "risk_level": risk_level,
        "treatments": treatments.get(risk_level, treatments["medium"]),
        "follow_up_weeks": 2 if risk_level == "high" else 4 if risk_level == "medium" else 8,
    }


async def main() -> None:
    """Demonstrate function-first tool architecture."""
    print("🤖 Example 10: Function-First Tool Architecture")
    print("=" * 60)
    print()
    print("🎯 Key Benefits:")
    print("   • Tools are real functions by default")
    print("   • Auto-extract schemas from type hints")
    print("   • No ToolDefinition duplication")
    print("   • Easy testing with real vs mock")
    print()

    # Test 1: Function-Based Router (Real Tools)
    print("⚙️  Test 1: Function-Based Router (Real Tools)")
    print("-" * 50)

    # Create router and register real functions
    real_router = FunctionBasedToolRouter()
    real_router.register_function(search_medical_literature, "search")
    real_router.register_function(calculate_risk_score, "calculate_risk")
    real_router.register_function(generate_treatment_plan, "generate_plan")

    # Show auto-extracted schemas
    print("🔍 Auto-extracted tool schemas:")
    for tool_name, schema in real_router.get_all_tool_schemas().items():
        print(f"   📋 {tool_name}: {schema['description']}")
        for param in schema["parameters"]:
            required = "required" if param["required"] else "optional"
            print(f"      - {param['name']}: {param['type']} ({required})")
    print()

    # Create agent with real tools
    agent_factory = ReActAgentNode()
    mock_llm = MockLLM(
        responses=[
            "I'll search for information about diabetes treatment.\n\nINVOKE_TOOL: search(query='diabetes treatment', database='pubmed')\n\nBased on the search results, I'll calculate the risk score.\n\nINVOKE_TOOL: calculate_risk(factors='diabetes,hypertension', patient_age=70)\n\nGiven the high risk level, I'll generate a treatment plan.\n\nINVOKE_TOOL: generate_plan(condition='diabetes', risk_level='high')\n\nBased on the medical literature and risk assessment, I recommend immediate intervention with specialist consultation and medication, with follow-up in 2 weeks."
        ]
    )

    # Create agent (no ToolDefinition needed!)
    medical_agent = agent_factory(
        name="medical_agent",
        main_prompt="You are a medical AI assistant. Analyze the patient case: {{input}}. Use available tools to research, assess risk, and generate treatment plans.",
        config=AgentConfig(max_steps=3),
    )

    # Create and run pipeline
    graph = DirectedGraph()
    graph.add(medical_agent)

    orchestrator = Orchestrator(
        validator=coerce_validator(), ports={"llm": mock_llm, "tool_router": real_router}
    )

    print("🚀 Running medical agent with real tools...")
    result = await orchestrator.run(
        graph, {"input": "70-year-old patient with diabetes and hypertension"}
    )

    agent_result = result["medical_agent"]
    print(f"   📊 Tools used: {len(agent_result.tools_used)}")
    print(f"   🛠️  Tool calls: {agent_result.tools_used}")
    print(f"   📝 Response: {agent_result.result[:100]}...")
    print()

    # Show real tool execution results
    print("🔬 Real tool execution results:")
    for call in real_router.get_call_history():
        print(f"   🧪 {call['tool_name']}: {str(call['result'])[:80]}...")
    print()

    # Test 2: Function Router with Mock Functions (replaces MockToolRouter)
    print("🎭 Test 2: Function Router with Mock Functions")
    print("-" * 50)

    # Create router with simple mock functions (replaces MockToolRouter)
    mock_router = FunctionBasedToolRouter()

    async def mock_search(query: str) -> str:
        return "Mock search found 5 relevant papers on diabetes"

    async def mock_calculate_risk(factors: str) -> str:
        return "Mock calculation: high risk (0.85)"

    async def mock_generate_plan(condition: str) -> str:
        return "Mock plan: immediate intervention recommended"

    mock_router.register_function(mock_search, "search")
    mock_router.register_function(mock_calculate_risk, "calculate_risk")
    mock_router.register_function(mock_generate_plan, "generate_plan")

    # Create new MockLLM with different response for mock test
    mock_llm_2 = MockLLM(
        responses=[
            "I'll analyze this case using available tools.\n\nINVOKE_TOOL: search(query='patient analysis')\n\nINVOKE_TOOL: calculate_risk(factors='multiple')\n\nINVOKE_TOOL: generate_plan(condition='complex')\n\nBased on mock analysis, treatment recommended."
        ]
    )

    print("🚀 Running same agent with mock tools...")
    result = await orchestrator.run(
        graph,
        {"input": "Complex patient case"},
        additional_ports={"llm": mock_llm_2, "tool_router": mock_router},
    )

    agent_result = result["medical_agent"]
    print(f"   📊 Tools used: {len(agent_result.tools_used)}")
    print(f"   🛠️  Tool calls: {agent_result.tools_used}")
    print(f"   📝 Response: {agent_result.result[:100]}...")
    print()

    # Test 3: Decorator Pattern
    print("🎨 Test 3: Decorator Pattern")
    print("-" * 35)

    decorator_router = FunctionBasedToolRouter()

    @decorator_router.tool
    async def quick_diagnosis(symptoms: str) -> dict[str, Any]:
        """Provide quick diagnostic suggestions."""
        await asyncio.sleep(0.03)
        return {
            "symptoms": symptoms,
            "suggestions": ["Further testing needed", "Consider common conditions"],
            "confidence": 0.6,
        }

    @decorator_router.tool
    async def check_drug_interactions(medications: str) -> dict[str, Any]:
        """Check for drug interactions."""
        await asyncio.sleep(0.04)
        med_list = medications.split(",")
        return {
            "medications": med_list,
            "interactions": (
                "No major interactions found"
                if len(med_list) < 3
                else "Potential interaction detected"
            ),
            "severity": "low",
        }

    print(f"🔧 Registered tools via decorator: {decorator_router.get_available_tools()}")

    # Test tool directly
    diagnosis = await decorator_router.call_tool("quick_diagnosis", {"symptoms": "fever, headache"})
    interactions = await decorator_router.call_tool(
        "check_drug_interactions", {"medications": "aspirin,ibuprofen"}
    )

    print(f"   🩺 Quick diagnosis: {diagnosis}")
    print(f"   💊 Drug interactions: {interactions}")
    print()

    # Test 4: Demo Router
    print("🚀 Test 4: Pre-built Demo Router")
    print("-" * 40)

    demo_router = FunctionBasedToolRouter()
    print(f"📦 Demo router tools: {demo_router.get_available_tools()}")

    # Test demo tools
    search_result = await demo_router.call_tool("search", {"query": "AI healthcare"})
    calc_result = await demo_router.call_tool("calculate", {"expression": "100 + 50"})

    print(f"   🔍 Search result: {search_result}")
    print(f"   🧮 Calculation: {calc_result}")
    print()

    # Test 5: Integration with Existing Architecture
    print("🔗 Test 5: Integration with ToolDescriptionManager")
    print("-" * 55)

    # Create function-based router
    integration_router = FunctionBasedToolRouter()

    @integration_router.tool
    async def medical_search(query: str, database: str = "pubmed") -> dict:
        """Search medical literature for research papers."""
        await asyncio.sleep(0.1)
        return {"papers": [f"Paper about {query}"], "database": database}

    @integration_router.tool
    async def risk_calculator(factors: str, age: int) -> dict:
        """Calculate medical risk based on patient factors."""
        await asyncio.sleep(0.05)
        return {"risk_score": 0.3, "factors": factors, "age": age}

    # Get auto-generated ToolDefinitions for existing architecture
    tool_definitions = integration_router.get_tool_definitions()

    print("🔧 Auto-generated ToolDefinitions:")
    for tool_def in tool_definitions:
        print(f"   📋 {tool_def.name}: {tool_def.simplified_description}")
        print(f"      Parameters: {len(tool_def.parameters)} params")
        print(f"      Example: {tool_def.examples[0] if tool_def.examples else 'None'}")
    print()

    # Create agent using runtime tool discovery (no available_tools needed)
    integration_agent = agent_factory(
        name="integration_agent",
        main_prompt="You are a medical AI. Use tools to research: {{input}}",
        config=AgentConfig(max_steps=2),
    )

    # Test with real tools
    integration_graph = DirectedGraph()
    integration_graph.add(integration_agent)

    integration_llm = MockLLM(
        responses=[
            "I'll search for medical information.\n\nINVOKE_TOOL: medical_search(query='diabetes treatment', database='pubmed')\n\nNow I'll calculate risk.\n\nINVOKE_TOOL: risk_calculator(factors='diabetes', age=65)\n\nBased on the search and risk calculation, here's my analysis."
        ]
    )

    print("🚀 Running agent with auto-generated ToolDefinitions...")
    result = await orchestrator.run(
        integration_graph,
        {"input": "diabetes treatment for elderly"},
        additional_ports={"llm": integration_llm, "tool_router": integration_router},
    )

    agent_result = result["integration_agent"]
    print(f"   📊 Tools used: {len(agent_result.tools_used)}")
    print(f"   🛠️  Tool calls: {agent_result.tools_used}")
    print(f"   📝 Response: {agent_result.result[:100]}...")
    print()

    # Show tool execution history
    print("🔬 Integration tool execution:")
    for call in integration_router.get_call_history():
        print(f"   🧪 {call['tool_name']}: {str(call['result'])[:60]}...")
    print()

    # Performance Summary
    print("📈 Architecture Comparison")
    print("-" * 35)
    print("   ❌ Old ToolDefinition approach:")
    print("      • Define metadata separately")
    print("      • Manually implement in router")
    print("      • Easy to get out of sync")
    print("      • Mock by default")
    print()
    print("   ✅ New Function-First approach:")
    print("      • Define once as real function")
    print("      • Auto-extract metadata")
    print("      • Real by default")
    print("      • Easy testing with mocks")
    print("      • Type-safe with hints")
    print()

    print("🎯 Key Benefits:")
    print("   ✅ No duplication - define tools once")
    print("   ✅ Real tools by default")
    print("   ✅ Auto-extract schemas from type hints")
    print("   ✅ Easy testing with mock overrides")
    print("   ✅ Better developer experience")
    print("   ✅ Type safety and validation")
    print()

    print("🔗 Next: This architecture eliminates ToolDefinition complexity!")


if __name__ == "__main__":
    asyncio.run(main())
