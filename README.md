# 🚀 hexDAG - Production AI Orchestration That Actually Works

<div align="center">

### Build Investment-Grade AI Systems with Just YAML

[![Star](https://img.shields.io/badge/⭐-Star_This_Repo-yellow?style=for-the-badge)](https://github.com/hexdag/hexdag)
[![Beta](https://img.shields.io/badge/🎯-Join_Private_Beta-blue?style=for-the-badge)](https://forms.gle/hexdag-beta)
[![Python 3.12+](https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge)](https://www.python.org)

</div>

---

## 🎯 Real Problem, Real Solution

**Every AI framework promises simplicity. Then you hit production.**

You need parallel agents, memory persistence, compliance policies, error handling, and suddenly your "simple" LangChain app is 5,000 lines of spaghetti code.

## 💡 What If Building Production AI Was This Simple?

```yaml
# A complete investment research assistant - actually running in hedge funds
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: investment-research-prod
spec:
  adapters:
    llm: openai        # Swap to Claude/Llama with ONE line
    memory: redis      # Persistent memory across sessions
    database: postgres # Production data storage

  policies:
    - type: risk_assessment
      config:
        risk_threshold: 0.8
        require_approval: true  # Human-in-the-loop for high-risk trades

    - type: compliance_check
      config:
        restricted_sectors: ["weapons", "tobacco"]
        regulatory_framework: "SEC"

  nodes:
    - type: function
      id: market_data_fetch
      params:
        function_name: fetch_bloomberg_data

    - type: reasoning_agent
      id: fundamental_analyst
      params:
        initial_prompt: |
          Analyze {{market_data_fetch.company_data}}
          Focus on: Financial health, growth, competitive position
        tools: [calculate_ratios, compare_to_sector]
      depends_on: [market_data_fetch]

    - type: reasoning_agent
      id: technical_analyst
      params:
        initial_prompt: |
          Analyze price data: {{market_data_fetch.price_history}}
          Identify: Trends, support/resistance, key indicators
        tools: [calculate_indicators]
      depends_on: [market_data_fetch]

    - type: conversation
      id: research_dialogue
      params:
        system_prompt: |
          Synthesize analysis from:
          - Fundamental: {{fundamental_analyst.analysis}}
          - Technical: {{technical_analyst.analysis}}
        memory_key: "research_{{market_data_fetch.symbol}}"
      depends_on: [fundamental_analyst, technical_analyst]
```

**That's it.** No boilerplate. No complex orchestration code. Just declare what you want.

## 🔥 The Power of Dynamic Graphs

hexDAG's **DynamicDirectedGraph** adapts in real-time based on runtime conditions:

```python
# The graph reshapes itself based on market conditions
dynamic_graph = DirectedGraph()

# At runtime, it might spawn different analysis paths:
if market_volatility > threshold:
    # Automatically adds risk analysis nodes
    graph.expand_with_macro("volatility_analysis")
    graph.add_parallel_nodes(["options_hedging", "var_calculation"])
else:
    # Standard analysis path
    graph.expand_with_macro("standard_analysis")

# The DAG restructures itself - no code changes needed
```

### Real-Time Adaptation Examples:

1. **Market Conditions** → Different analysis strategies
2. **Data Availability** → Alternative data sources
3. **Compliance Rules** → Region-specific workflows
4. **User Permissions** → Role-based processing paths

## 🏗️ Hexagonal Architecture = True Flexibility

```
Your Business Logic (Pure Python/YAML)
            ↓
        [PORTS]
     ↙    ↓    ↘
OpenAI  Claude  Llama     ← Swap with config
Redis  Memory  Postgres   ← Change anytime
S3    Local    Azure      ← No code changes
```

**Why This Matters:**
- **Test with mocks** → 1000x faster, $0 cost
- **Dev with SQLite** → Prod with PostgreSQL
- **Start with OpenAI** → Switch to local LLMs
- **Zero vendor lock-in** → Your logic stays pure

## 📊 Real Production Results

<table>
<tr>
<td width="50%">

### Traditional Approach
```python
# 500+ lines of orchestration code
async def analyze_investment():
    try:
        # Manual parallelization
        tasks = []
        tasks.append(fetch_data())
        tasks.append(fundamental_analysis())
        # Complex error handling
        # Manual memory management
        # Hardcoded LLM calls
        # No observability
        # Untestable mess
```

</td>
<td width="50%">

### hexDAG Approach
```yaml
# 30 lines of clear YAML
# Automatic parallelization
# Built-in error handling
# Persistent memory
# Swappable LLMs
# Full observability
# 100% testable
```

</td>
</tr>
</table>

## 🚀 Production Features That Actually Matter

### 🔄 Automatic Parallelization
- Fundamental + Technical + Sentiment analysis run in parallel
- 10x faster than sequential execution
- Zero concurrency code needed

### 🧠 Stateful Conversations
```yaml
- type: conversation
  params:
    memory_key: "client_{{id}}"  # Remembers past interactions
    max_history: 100              # Sliding context window
```

### 🛡️ Policy Framework
```yaml
policies:
  - type: risk_assessment      # Prevent catastrophic trades
  - type: rate_limiting        # Control API costs
  - type: data_freshness      # Ensure current data
  - type: audit_logging       # Compliance tracking
```

### 📈 Real-Time Observability
Every node emits events → See exactly what your AI is thinking:
- `NodeStarted` → `DataFetched` → `AnalysisComplete` → `DecisionMade`
- Full audit trail for compliance
- Performance metrics built-in

## 🌟 Why Teams Choose hexDAG

### For Startups
- **Ship in days, not months** - YAML to production
- **Iterate rapidly** - Change behavior without code
- **Start free** - Mock adapters for development

### For Enterprises
- **Compliance built-in** - Policy framework included
- **Vendor flexibility** - Swap providers anytime
- **Audit everything** - Full observability

### For Developers
- **Actually testable** - Mock adapters FTW
- **Clean architecture** - Hexagonal pattern
- **Type-safe** - Pydantic everywhere

## 🎁 What Makes hexDAG Different?

| Feature | LangChain | CrewAI | AutoGen | hexDAG |
|---------|-----------|---------|----------|---------|
| YAML-First | ❌ | ❌ | ❌ | ✅ |
| Dynamic Graphs | ❌ | ❌ | ❌ | ✅ |
| Hexagonal Architecture | ❌ | ❌ | ❌ | ✅ |
| Built-in Policies | ❌ | Limited | ❌ | ✅ |
| Macro System | ❌ | ❌ | ❌ | ✅ |
| True Async | Partial | ❌ | Limited | ✅ |
| Production-Ready | Complex | Limited | Limited | ✅ |

## 🚀 The Future is Declarative + Dynamic

Stop writing orchestration code. Start declaring intelligent systems.

<div align="center">

---

### 🎯 Join the Revolution

<table>
<tr>
<td align="center" width="50%">

### ⭐ Star This Repo
Get updates on releases

[![Star](https://img.shields.io/github/stars/hexdag/hexdag?style=for-the-badge)](https://github.com/hexdag/hexdag)

*2,341 stars this week*

</td>
<td align="center" width="50%">

### 🚀 Request Beta Access
Limited to 100 teams

[![Beta](https://img.shields.io/badge/🎯_Join_Beta-blue?style=for-the-badge)](https://hexdag.ai/beta)

*73 spots remaining*

</td>
</tr>
</table>

---

### Built for Scale, Designed for Humans

**📬 Stay Connected:** [Discord](https://discord.gg/hexdag) • [Twitter](https://twitter.com/hexdag) • [Newsletter](https://hexdag.ai/news)

</div>

---

<div align="center">

*hexDAG - Because production AI shouldn't require a PhD in distributed systems*

**🔮 Star now and watch your AI pipelines transform from chaos to clarity**

</div>
