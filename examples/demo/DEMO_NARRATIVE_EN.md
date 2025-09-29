# hexDAG Manifesto: 15 Principles

## Part I: Deterministic Philosophy

1. **The best AI is if-else**: Rule-based systems are deterministic, testable, and debuggable. If you can solve a problem with rules, you don't need an LLM.

2. **Programming with LLMs is stochastic programming**: Every model invocation is a probabilistic experiment.

3. **Programming is not quantum mechanics**: Your code shouldn't be in a superposition of states. It either works or it doesn't - there's no "maybe it will work".

4. **Validation is determinism in practice**: Every input, output, and state transition must have a contract. "Usually returns JSON" is not a production plan.

5. **You can do everything, but it's your problem**: It's easy to add your own node, adapter, or plugin. If it crashes, that's your problem too.

## Part II: Complexity Hierarchy

6. **80% of AI doesn't need agents**: Most problems are parsing, classification, and routing. Deterministic workflows are sufficient.

7. **Of the 20% that need agents, 80% don't need multi-agents**: One agent with good tools will solve almost everything.

8. **Multi-agents are usually bad architecture**: In 80% of cases, it's the programmer trying to hide their lack of problem understanding behind complexity.

9. **YAML is code too**: If you can't express the problem declaratively, you don't understand it well enough.

10. **Simple scales, clever breaks**: The best infrastructure is invisible. The worst requires a PhD to debug.

## Part III: Speed and Scalability

11. **Async-first or lose the race**: Blocking I/O in 2024 is consciously choosing slowness. Waiting is wasting money.

12. **Parallelism through analysis, not prayer**: Deterministic dependency analysis automatically parallelizes. Humans are terrible at this.

13. **Errors fast and loud**: Fail fast, fail loud. Silent errors kill production and budget.

14. **Framework does the heavy lifting**: Retry, timeout, error handling, caching - that's all infrastructure, not your problem.

15. **Clients care about only one thing**: That it works. Fast. Every time. Period.
