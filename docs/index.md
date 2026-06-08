# hexDAG

**Developer-first workflow engine for AI agents.** Compose n8n-like automations in YAML or Python, run LangGraph-style agent flows as typed DAGs, and ship them with observability, replay, and human approval.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Install hexDAG and build your first pipeline in minutes.

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-map:{ .lg .middle } __Framework Guide__

    ---

    Comprehensive guide — layers, compilation, execution, data flow, and how everything connects.

    [:octicons-arrow-right-24: Framework Guide](GUIDE.md)

-   :material-layers-triple:{ .lg .middle } __Architecture__

    ---

    Design philosophy, hexagonal architecture, and decision rules.

    [:octicons-arrow-right-24: Architecture](ARCHITECTURE.md)

-   :material-book-open-variant:{ .lg .middle } __Reference__

    ---

    Auto-generated API reference for nodes, ports, adapters, and tools.

    [:octicons-arrow-right-24: Reference](reference/nodes.md)

</div>

---

## How It All Connects

An order lifecycle in hexDAG: **Agent** reasons about request → transitions **Entity** to "processing" → emits **Event** → triggers **Process** (pipeline) → uses **Memory** for context → **Expressions** enforce rules → **Capabilities** scope permissions → **Observers** track everything.

All concepts are tightly integrated via Events, Services, and the kernel execution engine.

## Architecture Overview

<style>
  .arch-svg { display: block; margin: 0 auto; max-width: 920px; }
  .arch-svg a text { cursor: pointer; }
  .arch-svg a:hover rect { filter: brightness(1.1); }
  .arch-svg a:hover text { text-decoration: underline; }
  .arch-svg .layer { rx: 10; ry: 10; }
  .arch-svg .box { rx: 6; ry: 6; cursor: pointer; }
  .arch-svg .box:hover { filter: brightness(1.15); }
  .arch-svg .label { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; pointer-events: none; }
  .arch-svg .arrow { fill: none; stroke-width: 1.8; }
  .arch-svg .arrow-head { stroke: none; }
  .arch-svg .note { font-family: -apple-system, sans-serif; font-size: 10px; fill: #656d76; }
  .arch-svg .step-num { font-family: -apple-system, sans-serif; font-weight: 700; font-size: 14px; }
</style>

<svg class="arch-svg" viewBox="0 0 920 660" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="ah" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#8b949e" class="arrow-head"/></marker>
  <marker id="ahb" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#0969da" class="arrow-head"/></marker>
  <marker id="ahg" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#1a7f37" class="arrow-head"/></marker>
  <marker id="ahp" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#8250df" class="arrow-head"/></marker>
</defs>

<!-- ======== STEP 1: YAML ======== -->
<text class="label step-num" x="26" y="38" fill="#0969da">1</text>
<text class="label" x="44" y="38" font-size="13" font-weight="600" fill="#0969da">You write YAML</text>

<a href="GUIDE/#5-yaml-syntax-the-4-special-syntaxes">
  <rect class="box" x="24" y="48" width="560" height="62" fill="#ddf4ff" stroke="#0969da" stroke-width="1.5"/>
  <text class="label" x="44" y="68" font-size="12.5" font-weight="600" fill="#0969da">YAML Pipeline</text>
  <text class="label" x="44" y="84" font-size="11" fill="#0969da">kind: Pipeline &middot; nodes &middot; ports &middot; state_machines</text>
  <text class="label" x="44" y="100" font-size="10" fill="#57606a">Special syntax: !include &middot; ${VAR} &middot; {{ template }} &middot; node.field references</text>
</a>

<!-- Arrow 1→2 -->
<path class="arrow" d="M304,110 L304,138" stroke="#8b949e" marker-end="url(#ah)"/>

<!-- ======== STEP 2: COMPILER ======== -->
<text class="label step-num" x="26" y="158" fill="#cf222e">2</text>
<text class="label" x="44" y="158" font-size="13" font-weight="600" fill="#cf222e">Compiler processes it</text>

<rect class="layer" x="24" y="166" width="560" height="86" fill="#ffebe9" fill-opacity="0.35" stroke="#cf222e" stroke-opacity="0.3" stroke-width="1.5"/>

<a href="GUIDE/#13-the-compiler-in-detail">
  <rect class="box" x="38" y="178" width="160" height="36" fill="#fff" stroke="#cf222e" stroke-width="1.5"/>
  <text class="label" x="118" y="198" font-size="11.5" font-weight="600" fill="#cf222e" text-anchor="middle">YamlPipelineBuilder</text>
  <text class="label" x="118" y="210" font-size="9.5" fill="#cf222e" text-anchor="middle">5-phase compilation</text>
</a>

<a href="GUIDE/#how-resolution-works">
  <rect class="box" x="212" y="178" width="100" height="36" fill="#fff" stroke="#cf222e" stroke-width="1.5"/>
  <text class="label" x="262" y="198" font-size="11.5" font-weight="600" fill="#cf222e" text-anchor="middle">Resolver</text>
  <text class="label" x="262" y="210" font-size="9.5" fill="#cf222e" text-anchor="middle">alias &rarr; class</text>
</a>

<a href="GUIDE/#auto-dependency-detection">
  <rect class="box" x="326" y="178" width="120" height="36" fill="#fff" stroke="#cf222e" stroke-width="1.5"/>
  <text class="label" x="386" y="198" font-size="11.5" font-weight="600" fill="#cf222e" text-anchor="middle">Ref Resolver</text>
  <text class="label" x="386" y="210" font-size="9.5" fill="#cf222e" text-anchor="middle">auto-detect deps</text>
</a>

<a href="GUIDE/#phase-3-validate">
  <rect class="box" x="460" y="178" width="110" height="36" fill="#fff" stroke="#cf222e" stroke-width="1.5"/>
  <text class="label" x="515" y="198" font-size="11.5" font-weight="600" fill="#cf222e" text-anchor="middle">Validator</text>
  <text class="label" x="515" y="210" font-size="9.5" fill="#cf222e" text-anchor="middle">schema &middot; cycles</text>
</a>

<text class="note" x="38" y="240" fill="#8b5c33">parse &rarr; env select &rarr; validate &rarr; preprocess (!include, ${}, {{}}) &rarr; build graph (nodes, macros, deps)</text>

<!-- Arrow 2→3 -->
<path class="arrow" d="M304,252 L304,280" stroke="#cf222e" marker-end="url(#ah)"/>
<text class="note" x="310" y="270">outputs: DirectedGraph + PipelineConfig</text>

<!-- ======== STEP 3: KERNEL ======== -->
<text class="label step-num" x="26" y="298" fill="#1a7f37">3</text>
<text class="label" x="44" y="298" font-size="13" font-weight="600" fill="#1a7f37">Kernel executes it</text>

<rect class="layer" x="24" y="306" width="560" height="170" fill="#dafbe1" fill-opacity="0.35" stroke="#1a7f37" stroke-opacity="0.3" stroke-width="1.5"/>

<a href="GUIDE/#14-the-orchestrator-in-detail">
  <rect class="box" x="38" y="318" width="200" height="54" fill="#fff" stroke="#1a7f37" stroke-width="2"/>
  <text class="label" x="138" y="340" font-size="13" font-weight="700" fill="#1a7f37" text-anchor="middle">Orchestrator</text>
  <text class="label" x="138" y="355" font-size="10" fill="#1a7f37" text-anchor="middle">walks DAG wave by wave</text>
  <text class="label" x="138" y="367" font-size="10" fill="#1a7f37" text-anchor="middle">asyncio.gather per wave</text>
</a>

<a href="GUIDE/#wave-based-execution">
  <rect class="box" x="254" y="318" width="140" height="40" fill="#fff" stroke="#1a7f37" stroke-width="1.5"/>
  <text class="label" x="324" y="336" font-size="11" font-weight="600" fill="#1a7f37" text-anchor="middle">DirectedGraph</text>
  <text class="label" x="324" y="350" font-size="9.5" fill="#1a7f37" text-anchor="middle">nodes + edges &rarr; waves</text>
</a>

<a href="GUIDE/#8-node-types">
  <rect class="box" x="408" y="318" width="160" height="40" fill="#fff" stroke="#1a7f37" stroke-width="1.5"/>
  <text class="label" x="488" y="336" font-size="11" font-weight="600" fill="#1a7f37" text-anchor="middle">NodeSpec[]</text>
  <text class="label" x="488" y="350" font-size="9.5" fill="#1a7f37" text-anchor="middle">fn &middot; schemas &middot; deps &middot; retry</text>
</a>

<a href="GUIDE/#event-system">
  <rect class="box" x="38" y="384" width="130" height="36" fill="#fff" stroke="#1a7f37" stroke-width="1.5"/>
  <text class="label" x="103" y="402" font-size="11" font-weight="600" fill="#1a7f37" text-anchor="middle">Event System</text>
  <text class="label" x="103" y="414" font-size="9.5" fill="#1a7f37" text-anchor="middle">Started &middot; Completed &middot; Failed</text>
</a>

<a href="GUIDE/#4-how-a-pipeline-runs-end-to-end">
  <rect class="box" x="182" y="384" width="140" height="36" fill="#fff" stroke="#1a7f37" stroke-width="1.5"/>
  <text class="label" x="252" y="402" font-size="11" font-weight="600" fill="#1a7f37" text-anchor="middle">PipelineRunner</text>
  <text class="label" x="252" y="414" font-size="9.5" fill="#1a7f37" text-anchor="middle">YAML &rarr; results</text>
</a>

<a href="GUIDE/#9-data-flow-between-nodes">
  <rect class="box" x="336" y="384" width="140" height="36" fill="#fff" stroke="#1a7f37" stroke-width="1.5"/>
  <text class="label" x="406" y="402" font-size="11" font-weight="600" fill="#1a7f37" text-anchor="middle">Data Flow</text>
  <text class="label" x="406" y="414" font-size="9.5" fill="#1a7f37" text-anchor="middle">mapping &middot; expr &middot; template</text>
</a>

<a href="GUIDE/#execution-components">
  <rect class="box" x="490" y="384" width="80" height="36" fill="#fff" stroke="#1a7f37" stroke-width="1.5"/>
  <text class="label" x="530" y="402" font-size="11" font-weight="600" fill="#1a7f37" text-anchor="middle">Context</text>
  <text class="label" x="530" y="414" font-size="9.5" fill="#1a7f37" text-anchor="middle">ports &middot; state</text>
</a>

<text class="note" x="38" y="464" fill="#2da44e">Execution: for each wave &rarr; check when &rarr; prepare inputs &rarr; validate &rarr; execute &rarr; retry on fail &rarr; emit events</text>

<!-- Arrow 3→4 -->
<path class="arrow" d="M304,476 L304,500" stroke="#1a7f37" marker-end="url(#ah)"/>

<!-- ======== STEP 4: STDLIB ======== -->
<text class="label step-num" x="26" y="518" fill="#8250df">4</text>
<text class="label" x="44" y="518" font-size="13" font-weight="600" fill="#8250df">Built with stdlib components</text>

<rect class="layer" x="24" y="526" width="560" height="50" fill="#fbefff" fill-opacity="0.35" stroke="#8250df" stroke-opacity="0.3" stroke-width="1.5"/>

<a href="GUIDE/#8-node-types">
  <rect class="box" x="38" y="536" width="80" height="30" fill="#fff" stroke="#8250df" stroke-width="1.5"/>
  <text class="label" x="78" y="555" font-size="11" font-weight="600" fill="#8250df" text-anchor="middle">Nodes</text>
</a>

<a href="GUIDE/#built-in-adapters">
  <rect class="box" x="130" y="536" width="90" height="30" fill="#fff" stroke="#8250df" stroke-width="1.5"/>
  <text class="label" x="175" y="555" font-size="11" font-weight="600" fill="#8250df" text-anchor="middle">Adapters</text>
</a>

<a href="GUIDE/#11-macros-vs-nodes">
  <rect class="box" x="232" y="536" width="80" height="30" fill="#fff" stroke="#8250df" stroke-width="1.5"/>
  <text class="label" x="272" y="555" font-size="11" font-weight="600" fill="#8250df" text-anchor="middle">Macros</text>
</a>

<a href="GUIDE/#middleware">
  <rect class="box" x="324" y="536" width="100" height="30" fill="#fff" stroke="#8250df" stroke-width="1.5"/>
  <text class="label" x="374" y="555" font-size="11" font-weight="600" fill="#8250df" text-anchor="middle">Middleware</text>
</a>

<a href="GUIDE/#10-services">
  <rect class="box" x="436" y="536" width="90" height="30" fill="#fff" stroke="#8250df" stroke-width="1.5"/>
  <text class="label" x="481" y="555" font-size="11" font-weight="600" fill="#8250df" text-anchor="middle">Services</text>
</a>

<!-- ======== STEP 5: RESULTS ======== -->
<path class="arrow" d="M304,576 L304,600" stroke="#8250df" marker-end="url(#ah)"/>

<text class="label step-num" x="26" y="620" fill="#1a7f37">5</text>
<rect class="box" x="140" y="604" width="330" height="40" fill="#dafbe1" stroke="#1a7f37" stroke-width="2"/>
<text class="label" x="305" y="624" font-size="13" font-weight="700" fill="#1a7f37" text-anchor="middle">Results</text>
<text class="label" x="305" y="638" font-size="10" fill="#1a7f37" text-anchor="middle">{node_name: output} for every node + events log</text>

<!-- ======== RIGHT SIDE: PORTS & ADAPTERS ======== -->
<rect class="layer" x="610" y="48" width="290" height="596" fill="#f6f8fa" stroke="#d0d7de" stroke-width="1.5"/>
<text class="label" x="630" y="72" font-size="13" font-weight="700" fill="#656d76">Ports &amp; Adapters</text>
<text class="note" x="630" y="86">Contracts + swappable implementations</text>

<!-- Ports -->
<text class="label" x="630" y="112" font-size="11" font-weight="600" fill="#656d76">Port contracts:</text>

<a href="GUIDE/#ports-and-supports-protocols">
  <rect class="box" x="630" y="120" width="80" height="28" fill="#fff" stroke="#656d76" stroke-width="1.5"/>
  <text class="label" x="670" y="138" font-size="11" font-weight="600" fill="#656d76" text-anchor="middle">LLM</text>
</a>
<a href="GUIDE/#ports-and-supports-protocols">
  <rect class="box" x="718" y="120" width="90" height="28" fill="#fff" stroke="#656d76" stroke-width="1.5"/>
  <text class="label" x="763" y="138" font-size="11" font-weight="600" fill="#656d76" text-anchor="middle">DataStore</text>
</a>
<a href="GUIDE/#ports-and-supports-protocols">
  <rect class="box" x="816" y="120" width="72" height="28" fill="#fff" stroke="#656d76" stroke-width="1.5"/>
  <text class="label" x="852" y="138" font-size="11" font-weight="600" fill="#656d76" text-anchor="middle">DB</text>
</a>
<a href="GUIDE/#ports-and-supports-protocols">
  <rect class="box" x="630" y="154" width="258" height="24" fill="none" stroke="#656d76" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="note" x="759" y="170" text-anchor="middle">+ Memory &middot; APICall &middot; FileStorage &middot; Secret &middot; VFS &middot; Spawner</text>
</a>

<!-- Supports* -->
<a href="GUIDE/#ports-and-supports-protocols">
  <rect class="box" x="630" y="190" width="258" height="32" fill="#fbefff" stroke="#8250df" stroke-width="1" stroke-dasharray="3 2"/>
  <text class="label" x="759" y="206" font-size="10" fill="#8250df" text-anchor="middle">Supports* sub-protocols (fine-grained capabilities)</text>
  <text class="note" x="759" y="218" text-anchor="middle" fill="#8250df">SupportsGeneration &middot; SupportsFunctionCalling &middot; SupportsKeyValue &middot; ...</text>
</a>

<!-- Adapters -->
<text class="label" x="630" y="248" font-size="11" font-weight="600" fill="#656d76">Adapters (implement ports):</text>

<a href="GUIDE/#built-in-adapters">
  <rect class="box" x="630" y="256" width="70" height="26" fill="#ddf4ff" stroke="#0969da" stroke-width="1"/>
  <text class="label" x="665" y="273" font-size="10.5" fill="#0969da" text-anchor="middle">OpenAI</text>
</a>
<a href="GUIDE/#built-in-adapters">
  <rect class="box" x="706" y="256" width="78" height="26" fill="#ddf4ff" stroke="#0969da" stroke-width="1"/>
  <text class="label" x="745" y="273" font-size="10.5" fill="#0969da" text-anchor="middle">Anthropic</text>
</a>
<a href="GUIDE/#built-in-adapters">
  <rect class="box" x="790" y="256" width="56" height="26" fill="#ddf4ff" stroke="#0969da" stroke-width="1"/>
  <text class="label" x="818" y="273" font-size="10.5" fill="#0969da" text-anchor="middle">Mock</text>
</a>
<a href="GUIDE/#built-in-adapters">
  <rect class="box" x="630" y="288" width="74" height="26" fill="#ddf4ff" stroke="#0969da" stroke-width="1"/>
  <text class="label" x="667" y="305" font-size="10.5" fill="#0969da" text-anchor="middle">Postgres</text>
</a>
<a href="GUIDE/#built-in-adapters">
  <rect class="box" x="710" y="288" width="60" height="26" fill="#ddf4ff" stroke="#0969da" stroke-width="1"/>
  <text class="label" x="740" y="305" font-size="10.5" fill="#0969da" text-anchor="middle">SQLite</text>
</a>
<a href="GUIDE/#built-in-adapters">
  <rect class="box" x="776" y="288" width="70" height="26" fill="#ddf4ff" stroke="#0969da" stroke-width="1"/>
  <text class="label" x="811" y="305" font-size="10.5" fill="#0969da" text-anchor="middle">InMemory</text>
</a>

<!-- Middleware -->
<a href="GUIDE/#middleware">
  <rect class="box" x="630" y="330" width="258" height="40" fill="#fff" stroke="#656d76" stroke-width="1.5"/>
  <text class="label" x="759" y="348" font-size="11" font-weight="600" fill="#656d76" text-anchor="middle">Middleware</text>
  <text class="note" x="759" y="362" text-anchor="middle">Retry &rarr; Timeout &rarr; RateLimit &rarr; Cache &rarr; compose()</text>
</a>

<!-- Port overrides -->
<a href="GUIDE/#port-override-levels-3-tiers">
  <rect class="box" x="630" y="384" width="258" height="42" fill="#f6f8fa" stroke="#d0d7de" stroke-width="1"/>
  <text class="label" x="759" y="400" font-size="10.5" font-weight="600" fill="#656d76" text-anchor="middle">3 override levels:</text>
  <text class="note" x="759" y="416" text-anchor="middle">Global &rarr; Per-type &rarr; Per-node</text>
</a>

<!-- Services -->
<a href="GUIDE/#10-services">
  <rect class="box" x="630" y="440" width="258" height="52" fill="#fff" stroke="#656d76" stroke-width="1.5"/>
  <text class="label" x="759" y="458" font-size="11" font-weight="600" fill="#656d76" text-anchor="middle">Services (wrap ports)</text>
  <text class="note" x="759" y="472" text-anchor="middle">@tool = agent-callable &middot; @step = DAG node</text>
  <text class="note" x="759" y="486" text-anchor="middle">ProcessRegistry &middot; EntityState &middot; Scheduler &middot; Memory</text>
</a>

<!-- Entity lifecycle -->
<a href="GUIDE/#12-entity-lifecycle-state-machines">
  <rect class="box" x="630" y="506" width="258" height="42" fill="#fff8c5" stroke="#9a6700" stroke-width="1.5"/>
  <text class="label" x="759" y="524" font-size="11" font-weight="600" fill="#9a6700" text-anchor="middle">Entity Lifecycle</text>
  <text class="note" x="759" y="540" text-anchor="middle" fill="#9a6700">State machines &middot; TransitionNode &middot; Handlers</text>
</a>

<!-- Drivers -->
<a href="GUIDE/#3-the-layer-model">
  <rect class="box" x="630" y="562" width="258" height="36" fill="#fff1e5" stroke="#bc4c00" stroke-width="1"/>
  <text class="label" x="759" y="580" font-size="11" font-weight="600" fill="#bc4c00" text-anchor="middle">Drivers</text>
  <text class="note" x="759" y="592" text-anchor="middle" fill="#bc4c00">Executor &middot; ObserverManager &middot; VFS &middot; Spawner</text>
</a>

<!-- "Why this matters" -->
<text class="label" x="630" y="620" font-size="10" font-weight="600" fill="#656d76">Why this matters:</text>
<text class="note" x="630" y="634">Change OpenAI &rarr; Anthropic: 1 YAML line.</text>

<!-- ======== CONNECTING ARROWS ======== -->
<!-- Orchestrator → Ports -->
<path class="arrow" d="M238,345 C440,345 540,200 628,160" stroke="#8250df" stroke-dasharray="5 3" stroke-width="1.5" marker-end="url(#ahp)"/>
<text class="note" x="440" y="288" fill="#8250df" transform="rotate(-22,440,288)">nodes call ports at runtime</text>

</svg>

---

## Guides

| Guide | Description |
|-------|-------------|
| [Framework Guide](GUIDE.md) | How all the pieces fit together — the comprehensive reference |
| [Node Guide](generated/mcp/node_guide.md) | Creating custom nodes |
| [Adapter Guide](generated/mcp/adapter_guide.md) | Creating custom adapters |
| [Tool Guide](generated/mcp/tool_guide.md) | Creating custom tools |
| [YAML Reference](generated/mcp/yaml_reference.md) | Pipeline, Macro, and Config manifests |
| [Pipeline Schema](generated/mcp/pipeline_schema_guide.md) | YAML pipeline schema reference |
| [Syntax Reference](generated/mcp/syntax_reference.md) | YAML syntax and custom tags |
| [CLI Reference](generated/mcp/cli_reference.md) | All CLI commands and options |
