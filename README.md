**hexDAG - Production AI Orchestration with YAML configuration.That atually works.**

<div align="center">

# Join Beta ⬇️

[![Beta](https://img.shields.io/badge/🎯-Join_Private_Beta-blue?style=for-the-badge)](https://forms.gle/ZDNupq2pqbVPHMAA8)

</div>

**The Problem**
**Every AI framework promises simplicity. Then you hit production and... you know what happens**

Most AI frameworks break at scale. You start with a prototype, then need parallel execution, memory, error handling, compliance. 
What began as simple code... becomes thousands of lines of orchestration logic you can't test or maintain.
Models are not the problem. Look at the architecture first.

**The Solution**
hexDAG treats AI agents as components in deterministic workflows. Declare what should happen in YAML. The framework handles execution.

_Three principles of hexDAG:_

1. Deterministic core - if-else beats LLM for control flow
2. Declarative complexity - YAML is code. Complex behavior from simple rules
3. Scalable infrastructure - retry, timeout, cleanup belong in the framework

Result: AI systems that are fast, predictable, testable.


**Example 1: Investment Research Platform**

This runs three agents in parallel (fundamental + technical) → risk synthesis → actionable research. 
Compliance runs automatically. State persists across sessions.

<details>
<summary>Click to see the complete YAML (yes, this is production code)</summary>

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: investment-research
  description: AI-powered investment research with multi-agent collaboration

spec:
  nodes:
    - type: function
      id: market_data_fetch
      params:
        function_name: fetch_market_data
        module: __main__
      depends_on: []

    - type: macro_invocation
      id: fundamental_analyst
      macro: reasoning_agent
      params:
        initial_prompt: |
          Analyze company data: {{market_data_fetch.company_data}}

          Provide detailed fundamental analysis:
          1. Financial health (balance sheet, P&L, cash flow)
          2. Growth prospects (revenue trends, market expansion)
          3. Competitive positioning (market share, moat)
          4. Valuation (P/E, P/B, DCF)
          5. Investment recommendation (conviction 1-10)

        max_steps: "{{env.max_analysis_steps | default(3)}}"
        tools: [calculate_ratios, compare_to_sector, dcf_valuation]
      depends_on: [market_data_fetch]

    - type: macro_invocation
      id: technical_analyst
      macro: reasoning_agent
      params:
        initial_prompt: |
          Analyze price/volume: {{market_data_fetch.price_history}}

          Technical analysis required:
          1. Trend identification (Dow Theory)
          2. Support/resistance levels
          3. Momentum indicators (RSI, MACD)
          4. Volume analysis
          5. Entry/exit points with stops

        tools: [calculate_indicators, identify_patterns]
      depends_on: [market_data_fetch]

    - type: macro_invocation
      id: risk_analyst
      macro: reasoning_agent
      params:
        initial_prompt: |
          Assess risks for {{market_data_fetch.symbol}}:
          - Fundamental: {{fundamental_analyst.output}}
          - Technical: {{technical_analyst.output}}

          Analyze: Market risk, company risk, sector risk,
          liquidity, black swans, Sharpe ratio, allocation %

        tools: [calculate_var, stress_testing]
      depends_on: [fundamental_analyst, technical_analyst]

    - type: macro_invocation
      id: research_dialogue
      macro: conversation
      params:
        system_prompt: |
          Synthesize all analysis into actionable guidance.
          Be specific about entry, position size, and risk.

        conversation_id: "research_{{market_data_fetch.symbol}}"
        max_history: 20
      depends_on: [fundamental_analyst, technical_analyst, risk_analyst]

# Production environment with compliance
---
apiVersion: hexdag/v1
kind: Environment
metadata:
  name: prod

spec:
  adapters:
    llm: openai        # Or claude, llama, etc.
    memory: sqlite     # Persistent conversation history
    database: postgres # Market data storage

  policies:
    - type: compliance_check
      config:
        restricted_sectors: [cannabis, weapons, tobacco]
        regulatory_framework: "SEC"
        require_disclaimer: true

    - type: market_hours
      config:
        market_open: "09:30"
        market_close: "16:00"
        timezone: "EST"

    - type: position_sizing
      config:
        max_position_pct: 10
        max_sector_exposure: 30
```

</details>

**Result:** Three AI agents analyze in parallel → Risk assessment → Synthesized report. All with compliance, audit trails, and position sizing built-in.

**Example 2: Automated code review & Security audit**
Security scanner and code reviewer run in parallel. Results merge in interactive discussion. License violations block automatically. 

<details>
<summary>Click to see the security audit pipeline</summary>

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: code-review-security
  description: Automated code review with security and compliance

spec:
  nodes:
    - type: function
      id: code_loader
      params:
        function_name: load_code_for_review
      depends_on: []

    - type: function
      id: static_analyzer
      params:
        function_name: run_static_analysis
      depends_on: [code_loader]

    - type: macro_invocation
      id: security_scanner
      macro: reasoning_agent
      params:
        initial_prompt: |
          Security review for:
          Code: {{code_loader.code}}
          Static analysis: {{static_analyzer.results}}

          Check for:
          1. OWASP Top 10 vulnerabilities
          2. Authentication/authorization issues
          3. Data exposure risks
          4. Injection vulnerabilities
          5. Cryptographic practices
          6. Supply chain risks

          Provide line numbers and fixes.

        tools: [scan_dependencies, check_cve_database, analyze_crypto]
      depends_on: [code_loader, static_analyzer]

    - type: macro_invocation
      id: code_reviewer
      macro: reasoning_agent
      params:
        initial_prompt: |
          Review code for:
          1. Architecture and patterns
          2. Maintainability
          3. Error handling
          4. Performance
          5. Test coverage
          6. Documentation

        tools: [calculate_complexity, check_test_coverage]
      depends_on: [code_loader]

    - type: macro_invocation
      id: review_discussion
      macro: conversation
      params:
        system_prompt: |
          Help developer understand feedback from:
          Security: {{security_scanner.output}}
          Code Review: {{code_reviewer.output}}

          Be constructive with specific examples.

        conversation_id: "review_{{code_loader.pr_number}}"
        tools: [suggest_fix, generate_test]
      depends_on: [security_scanner, code_reviewer]

# Production policies
---
apiVersion: hexdag/v1
kind: Environment
metadata:
  name: prod

spec:
  policies:
    - type: security_vulnerability
      config:
        block_critical: true
        scan_depth: deep

    - type: license_compliance
      config:
        allowed_licenses: [MIT, Apache-2.0, BSD]
        blocked_licenses: [GPL-3.0, AGPL-3.0]

    - type: performance
      config:
        warn_on_n_plus_one: true
        max_db_queries: 10
```

</details>

**Result:** Parallel security + code review → AI synthesis → Actionable feedback. Catches vulnerabilities, license issues, and performance problems automatically.
**That's it.** 
No boilerplate. 
No complex orchestration code. 
Just declare what you want.


**The Power of dynamic graphs**

**DynamicDirectedGraph** in hexDag adapts in real-time based on runtime conditions:

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

### Real-time adaptation examples:

1. **Market Conditions** → Different analysis strategies
2. **Data Availability** → Alternative data sources
3. **Compliance Rules** → Region-specific workflows
4. **User Permissions** → Role-based processing paths

## Why This Changes Everything

### Multi-Agent orchestration made simple
```yaml
# These agents run IN PARALLEL automatically
- type: macro_invocation
  id: fundamental_analyst
  macro: reasoning_agent

- type: macro_invocation
  id: technical_analyst
  macro: reasoning_agent

- type: macro_invocation
  id: risk_analyst
  macro: reasoning_agent
  depends_on: [fundamental_analyst, technical_analyst]  # Waits for both
```

No asyncio. No thread management. No race conditions. **It just works.**

## Hexagonal Architecture is THE True Flexibility

Business logic stays pure. External dependencies go through adapters.

```
Your Business Logic (Pure Python/YAML)
            ↓
        [PORTS]
     ↙    ↓    ↘
OpenAI  Claude  Llama     ← Swap with config
Redis  Memory  Postgres   ← Change anytime
S3    Local    Azure      ← No code changes
```

**Why this matters:**
- **Test with mocks** → 1000x faster, $0 cost
- **Dev with SQLite** → Prod with PostgreSQL
- **Start with OpenAI** → Switch to local LLMs
- **No vendor lock-in** → Your logic stays pure

## Real production results

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

## Production features that actually matter

### Automatic Parallelization
- Fundamental + Technical + Sentiment analysis run in parallel
- 10x faster than sequential execution
- Zero concurrency code needed

### Stateful Conversations
```yaml
- type: conversation
  params:
    memory_key: "client_{{id}}"  # Remembers past interactions
    max_history: 100              # Sliding context window
```

### Policy Framework
```yaml
policies:
  - type: risk_assessment      # Prevent catastrophic trades
  - type: rate_limiting        # Control API costs
  - type: data_freshness      # Ensure current data
  - type: audit_logging       # Compliance tracking
```

### Real-Time observability
Every node emits events → See exactly what your AI is thinking:
- `NodeStarted` → `DataFetched` → `AnalysisComplete` → `DecisionMade`
- Full audit trail for compliance
- Performance metrics built-in

## Why This Matters

### For Startups
- **Ship faster** - YAML to production. 
- **Iterate rapidly** - Change behavior without code
- **Start free** - Test with mocks, scale to production

### For Enterprises
- **Compliance built-in** - Policy framework included
- **Vendor flexibility** - Swap providers anytime
- **Audit everything** - Full observability

### For Developers
- **Actually testable** - Mock adapters FTW
- **Clean architecture** - Hexagonal pattern
- **Type-safe** - Pydantic everywhere

## Comparison

| Feature | LangChain | CrewAI | AutoGen | hexDAG |
|---------|-----------|---------|----------|---------|
| YAML-First | ❌ | ❌ | ❌ | ✅ |
| Dynamic Graphs | ❌ | ❌ | ❌ | ✅ |
| Hexagonal Architecture | ❌ | ❌ | ❌ | ✅ |
| Built-in Policies | ❌ | Limited | ❌ | ✅ |
| Macro System | ❌ | ❌ | ❌ | ✅ |
| True Async | Partial | ❌ | Limited | ✅ |
| Production-Ready | Complex | Limited | Limited | ✅ |

## Philosophy
The problem is organizing models into coherent systems. Most frameworks optimize for demos, not production.

**hexDAG inverts this.** 
More structure upfront (YAML vs Python), less complexity later (no orchestration code, no state management, no lock-in).

Simple rules. Complex emergent behavior.

## The future is declarative + dynamic

Stop writing orchestration code. Start declaring intelligent systems.

<div align="center">

---

### Join the Revolution

<table>
<tr>
<td align="center" width="50%">

### 🚀 Request Beta Access
Limited to 100 teams

[![Beta](https://img.shields.io/badge/🎯_Join_Beta-blue?style=for-the-badge)](https://forms.gle/ZDNupq2pqbVPHMAA8)

*73 spots remaining*

</td>
</tr>
</table>

---

### Built for Scale, Designed for Humans

**📬 Stay Connected:** [Twitter](https://x.com/jhkwapisz)

</div>

---

<div align="center">

**Star now and watch your AI pipelines transform from chaos to clarity**

Beta Access [![Beta](https://img.shields.io/badge/🎯-Join_Private_Beta-blue?style=for-the-badge)](https://forms.gle/ZDNupq2pqbVPHMAA8)

Follow: @jhkwapisz

</div>

**Production AI without the PhD in distributed systems**
