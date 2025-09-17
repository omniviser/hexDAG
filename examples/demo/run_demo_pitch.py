#!/usr/bin/env python
"""
hexDAG Demo Pitch Runner
"Because your startup demo should actually work"
"""

import asyncio
import random
import time
from pathlib import Path
from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# Mock implementations for demo purposes
console = Console()


class DemoPitchRunner:
    def __init__(self):
        self.console = console
        self.manifest_path = Path("demo_startup_pitch.yaml")
        self.langchain_path = Path("demo_langchain_comparison.py")

    async def simulate_hexdag_workflow(
        self, ticket_text: str, customer_tier: str
    ) -> dict[str, Any]:
        """Simulate hexDAG workflow execution"""
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            # Step 1: Parse ticket
            task = progress.add_task("[cyan]Parsing ticket with schema validation...", total=1)
            await asyncio.sleep(0.5)
            parsed = {
                "issue_type": "bug",
                "technical_details": "Customer reports application crashes during report generation",
                "customer_emotion": "frustrated",
            }
            progress.update(task, completed=1)

            # Step 2: Conditional routing
            task = progress.add_task(f"[yellow]Routing based on tier: {customer_tier}...", total=1)
            await asyncio.sleep(0.3)
            is_enterprise = customer_tier == "enterprise"
            progress.update(task, completed=1)

            # Step 3: Analysis
            if is_enterprise:
                task = progress.add_task("[red]Enterprise analysis (GPT-4o)...", total=1)
                await asyncio.sleep(0.8)
                analysis = {
                    "priority": "critical",
                    "action_items": [
                        "Immediate hotfix",
                        "Root cause analysis",
                        "Executive briefing",
                    ],
                    "executive_summary": "Critical production issue affecting enterprise client",
                }
            else:
                task = progress.add_task("[green]Standard analysis (o3-mini)...", total=1)
                await asyncio.sleep(0.4)
                analysis = {
                    "priority": "high",
                    "suggested_response": "We're investigating the issue and will update you shortly.",
                }
            progress.update(task, completed=1)

            # Step 4: Parallel notifications
            task = progress.add_task("[magenta]Sending notifications in parallel...", total=3)
            await asyncio.sleep(0.2)
            progress.update(task, advance=1)
            await asyncio.sleep(0.2)
            progress.update(task, advance=1)
            await asyncio.sleep(0.2)
            progress.update(task, advance=1)

        processing_time = (time.time() - start_time) * 1000

        # Generate demo ticket ID (not for security purposes)
        # nosec B311 - This is for demo ticket IDs only, not security-sensitive
        ticket_suffix = random.randint(100000, 999999)  # nosec B311
        return {
            "ticket_id": f"TKT-{ticket_suffix}",
            "response_sent": True,
            "processing_time_ms": processing_time,
            "parsed": parsed,
            "analysis": analysis,
            "tier": customer_tier,
        }

    async def simulate_langchain_workflow(
        self, ticket_text: str, customer_tier: str
    ) -> dict[str, Any]:
        """Simulate LangChain workflow with realistic failures"""
        start_time = time.time()
        errors = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            # Step 1: Parse ticket (with retry)
            task = progress.add_task("[cyan]Parsing ticket (attempt 1/3)...", total=3)
            await asyncio.sleep(0.5)
            progress.update(task, advance=1)

            # Simulate parsing failure
            await asyncio.sleep(0.5)
            errors.append("OutputParsingException: Invalid JSON response")
            progress.update(task, description="[cyan]Parsing ticket (attempt 2/3)...")
            progress.update(task, advance=1)

            await asyncio.sleep(0.5)
            parsed = {
                "issue_type": "other",  # Fallback
                "technical_details": "Error parsing ticket",
                "customer_emotion": "frustrated",
            }
            progress.update(task, advance=1)

            # Step 2: Manual routing
            task = progress.add_task("[yellow]Checking tier with if-else...", total=1)
            await asyncio.sleep(0.3)
            progress.update(task, completed=1)

            # Step 3: Analysis
            if customer_tier == "enterprise":
                task = progress.add_task("[red]Enterprise analysis...", total=1)
                await asyncio.sleep(1.2)  # Slower
                analysis = {"priority": "high", "action_items": ["Investigate issue"]}
            else:
                task = progress.add_task("[green]Standard analysis...", total=1)
                await asyncio.sleep(0.6)
                analysis = {"priority": "medium"}
            progress.update(task, completed=1)

            # Step 4: Sequential notifications (not parallel)
            task = progress.add_task("[magenta]Sending notifications sequentially...", total=3)
            await asyncio.sleep(0.4)
            progress.update(task, advance=1)
            await asyncio.sleep(0.4)
            progress.update(task, advance=1)
            await asyncio.sleep(0.4)
            progress.update(task, advance=1)

        processing_time = (time.time() - start_time) * 1000

        # Generate demo ticket ID (not for security purposes)
        # nosec B311 - This is for demo ticket IDs only, not security-sensitive
        ticket_suffix = random.randint(100000, 999999)  # nosec B311
        return {
            "ticket_id": f"TKT-{ticket_suffix}",
            "response_sent": True,
            "processing_time_ms": processing_time,
            "errors": errors,
            "parsed": parsed,
            "analysis": analysis,
        }

    def show_manifest_preview(self):
        """Display the YAML manifest with syntax highlighting"""
        self.console.print("\n[bold cyan]📄 hexDAG Manifest (110 lines of YAML)[/bold cyan]\n")

        yaml_preview = """name: support_ticket_analyzer
nodes:
  - type: llm
    id: ticket_parser
    params:
      model: gpt-4o
      fallback_model: o3-mini # One line fallback
      max_retries: 3
      timeout: 10
    output_schema:
      type: object
      properties:
        issue_type: {type: string}

  - type: conditional
    id: tier_router
    params:
      condition: "customer_tier == 'enterprise'"

  # ... 3 more nodes running in parallel"""

        syntax = Syntax(yaml_preview, "yaml", theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title="demo_startup_pitch.yaml", border_style="green"))

    def show_langchain_preview(self):
        """Display the Python/LangChain code"""
        self.console.print(
            "\n[bold red]🐍 LangChain Implementation (400+ lines of Python)[/bold red]\n"
        )

        python_preview = """# 15 imports
from langchain.chains import LLMChain
from langchain.callbacks import AsyncCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio, json, logging, traceback
# ... 7 more imports

class CustomerSupportAnalyzer:
    def __init__(self):
        self.parser_llm = ChatOpenAI(model="gpt-4o", ...)
        self.enterprise_llm = ChatOpenAI(model="gpt-4o", ...)
        # Manual setup...

    @retry(stop=stop_after_attempt(3))
    async def parse_ticket(self, ...):
        try:
            result = await chain.arun(...)
            parsed = self.ticket_parser.parse(result)
        except Exception as e:
            # Manual error handling
            return fallback_response

# ... 400+ more lines of manual orchestration, error handling, and prayers ..."""

        syntax = Syntax(python_preview, "python", theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title="demo_langchain_comparison.py", border_style="red"))

    def show_comparison_table(self, hexdag_result: dict, langchain_result: dict):
        """Display side-by-side comparison"""
        table = Table(title="\n⚔️  hexDAG vs LangChain - Same Workflow, Different Worlds")

        table.add_column("Metric", style="cyan", width=30)
        table.add_column("hexDAG", style="green", width=25)
        table.add_column("LangChain", style="red", width=25)

        table.add_row("Lines of Code", "110 (YAML)", "400+ (Python)")
        table.add_row(
            "Processing Time",
            f"{hexdag_result['processing_time_ms']:.0f}ms",
            f"{langchain_result['processing_time_ms']:.0f}ms",
        )
        table.add_row("Parallel Execution", "✅ Automatic (DAG)", "❌ Manual (asyncio)")
        table.add_row("Schema Validation", "✅ Built-in", "❌ OutputFixingParser")
        table.add_row("Error Handling", "✅ Framework", "❌ Try-except blocks")
        table.add_row("Retries", "✅ max_retries: 3", "❌ @retry decorator")
        table.add_row("Observability", "✅ Event system", "❌ Print statements")
        table.add_row("Parse Errors", "0", f"{len(langchain_result.get('errors', []))}")

        self.console.print(table)

    def show_manifest_principles(self):
        """Display the hexDAG manifesto"""
        manifest = """
## 🎯 Manifest hexDAG: 15 Zasad

### Część I: Filozofia deterministyczna
1. **Najlepsze AI to if-else** - Systemy regułowe są deterministyczne, testowalne i debugowalne
2. **Programowanie z LLM-ami to programowanie stochastyczne** - Każde wywołanie to eksperyment probabilistyczny
3. **Programowanie to nie mechanika kwantowa** - Albo działa, albo nie - nie ma "może zadziała"
4. **Walidacja to determinizm w praktyce** - "Zwykle zwraca JSON" to nie plan na produkcję
5. **Możesz wszystko, ale to twój problem** - Łatwo dodasz własny node. Jak się wywali, to też twój problem

### Część II: Hierarchia złożoności
6. **80% AI nie potrzebuje agentów** - Większość problemów to parsowanie, klasyfikacja i routing
7. **Z tych 20% co potrzebują agentów, 80% nie potrzebuje multi-agentów** - Jeden agent wystarczy
8. **Multi-agenty to zwykle słaba architektura** - W 80% przypadków to brak zrozumienia problemu
9. **YAML to też kod** - Jeśli nie umiesz zapisać problemu deklaratywnie, nie rozumiesz go
10. **Proste się skaluje, cwane się sypie** - Najlepsza infrastruktura jest niewidoczna

### Część III: Prędkość i skalowalność
11. **Async-first albo przegrasz wyścig** - Czekanie to strata pieniędzy
12. **Równoległość przez analizę, nie przez modlitwę** - DAG wie lepiej co można robić jednocześnie
13. **Błędy szybkie i głośne** - Ciche błędy zabijają produkcję i budżet
14. **Framework robi ciężką robotę** - Retry, timeout, error handling - to infrastruktura, nie twój problem
15. **Klientów obchodzi tylko jedno** - Że działa. Szybko. Za każdym razem. Kropka.
"""
        md = Markdown(manifest)
        self.console.print(Panel(md, title="The hexDAG Way", border_style="bold blue"))

    async def simulate_error_scenario(self):
        """Simulate what happens when LLM returns garbage"""
        self.console.print(
            "\n[bold yellow]⚠️  ERROR SCENARIO: When GPT-4o fails, who handles the fallback?[/bold yellow]\n"
        )

        bad_response = "Sure! I'd be happy to help you with that. Here's the JSON you requested:\n\n{issue_type: 'bug', technical_details: 'The application crashes', customer_emotion: frustrated}"

        self.console.print("[red]GPT-4o Response (invalid JSON):[/red]")
        self.console.print(Panel(bad_response, border_style="red"))

        # Show code comparison
        self.console.print("\n[bold cyan]📝 How to add fallback to cheaper model:[/bold cyan]\n")

        # hexDAG code
        hexdag_code = """# hexDAG: 1 line change in YAML
params:
  model: gpt-4o
  fallback_model: o3-mini # ← Add this line"""

        # LangChain code
        langchain_code = """# LangChain: 60+ lines of Python
class CustomerSupportAnalyzer:
    def __init__(self):
        self.primary_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            max_retries=3
        )
        self.fallback_llm = ChatOpenAI(
            model="o3-mini",  # Fallback model
            temperature=0.1,
            max_retries=2
        )

    async def parse_with_fallback(self, ticket_text, customer_tier):
        models = [
            (self.primary_llm, "gpt-4o"),
            (self.fallback_llm, "o3-mini")
        ]

        for i, (llm, model_name) in enumerate(models):
            try:
                logging.info(f"Attempting with {model_name}")
                chain = LLMChain(llm=llm, prompt=self.prompt_template)

                result = await chain.arun(
                    ticket_text=ticket_text,
                    customer_tier=customer_tier
                )

                if self.validate_json_structure(result):
                    return result
                else:
                    logging.warning(f"{model_name} returned invalid JSON")
                    continue

            except OutputParserException as e:
                logging.error(f"{model_name} parsing failed: {e}")
                if i < len(models) - 1:
                    continue
                else:
                    return self.get_fallback_response()

            except Exception as e:
                logging.error(f"{model_name} failed: {e}")
                if i < len(models) - 1:
                    await asyncio.sleep(0.5 * (i + 1))  # Backoff
                    continue
                else:
                    raise

        return self.get_fallback_response()

    def validate_json_structure(self, result):
        # 20 more lines of validation...
        pass

    def get_fallback_response(self):
        # 10 more lines of fallback logic...
        pass"""

        # Display side by side

        hexdag_panel = Panel(
            Syntax(hexdag_code, "yaml", theme="monokai"), title="hexDAG", border_style="green"
        )
        langchain_panel = Panel(
            Syntax(langchain_code, "python", theme="monokai", line_numbers=True),
            title="LangChain",
            border_style="red",
        )
        self.console.print(Columns([hexdag_panel, langchain_panel], equal=True, expand=True))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            # hexDAG handling
            self.console.print("\n[bold green]hexDAG Automatic Fallback:[/bold green]")
            task = progress.add_task("[green]1. Schema validation fails...", total=1)
            await asyncio.sleep(0.5)
            progress.update(task, completed=1)

            task = progress.add_task(
                "[green]2. Auto-switching to o3-mini (YAML: fallback_model)...", total=1
            )
            await asyncio.sleep(0.7)
            progress.update(task, completed=1)

            task = progress.add_task("[green]3. o3-mini returns valid JSON...", total=1)
            await asyncio.sleep(0.5)
            progress.update(task, completed=1)
            self.console.print("  ✅ [green]Success with fallback_model[/green]")

            # LangChain handling
            self.console.print("\n[bold red]LangChain Manual Fallback:[/bold red]")
            task = progress.add_task("[red]1. OutputFixingParser attempt...", total=1)
            await asyncio.sleep(0.5)
            progress.update(task, completed=1)
            self.console.print("  ❌ [red]Failed to parse[/red]")

            task = progress.add_task("[red]2. Manual exception handling...", total=1)
            await asyncio.sleep(0.5)
            progress.update(task, completed=1)
            self.console.print("  ⚠️  [yellow]Developer must write fallback code[/yellow]")

            task = progress.add_task("[red]3. Trying o3-mini (60+ lines of code)...", total=1)
            await asyncio.sleep(0.7)
            progress.update(task, completed=1)
            self.console.print("  ❓ [yellow]Hope the budget allows it[/yellow]")

        table = Table(title="\nLLM Fallback Comparison")
        table.add_column("Aspect", style="cyan", width=25)
        table.add_column("hexDAG", style="green", width=35)
        table.add_column("LangChain", style="red", width=35)

        table.add_row("Configuration", "fallback_model: o3-mini", "60+ lines of fallback logic")
        table.add_row("Trigger", "Automatic on validation failure", "Manual in except block")
        table.add_row("Cost Control", "Only when needed, automatic", "Hope dev handles it properly")
        table.add_row(
            "Code Required",
            "1 line: fallback_model: o3-mini",
            "60+ lines with loops, retries, logging",
        )

        self.console.print(table)

    async def run_demo(self):
        """Main demo flow"""
        self.console.clear()

        # Title
        self.console.print(
            Panel.fit(
                "[bold cyan]hexDAG Demo[/bold cyan]\n"
                + "[yellow]'Because 80% of LLM workflows just need to work'[/yellow]",
                border_style="bold",
            )
        )

        # Test case
        ticket_text = "My enterprise application crashes during report generation. Critical for quarterly review!"
        customer_tier = "enterprise"

        self.console.print(f"\n📧 [bold]Incoming ticket:[/bold] {ticket_text}")
        self.console.print(f"👤 [bold]Customer tier:[/bold] {customer_tier}\n")

        self.console.print("\n[italic]Press Enter to see the implementations...[/italic]", end="")
        input()

        # Show code previews
        self.show_manifest_preview()
        self.show_langchain_preview()

        self.console.print("\n[italic]Press Enter to run both workflows...[/italic]", end="")
        input()

        # Run workflows
        self.console.print("\n[bold green]▶ Running hexDAG workflow[/bold green]")
        hexdag_result = await self.simulate_hexdag_workflow(ticket_text, customer_tier)

        self.console.print("\n[bold red]▶ Running LangChain workflow[/bold red]")
        langchain_result = await self.simulate_langchain_workflow(ticket_text, customer_tier)

        # Show results
        self.show_comparison_table(hexdag_result, langchain_result)

        # Show error scenario
        self.console.print(
            "\n[italic]Press Enter to see what happens when things go wrong...[/italic]", end=""
        )
        input()
        await self.simulate_error_scenario()

        # Show philosophy
        self.console.print("\n[italic]Press Enter to see the hexDAG philosophy...[/italic]", end="")
        input()
        self.show_manifest_principles()

        # Final message
        self.console.print(
            "\n[bold green]✨ hexDAG:[/bold green] [italic]Boring infrastructure for boring AI that boringly works.[/italic]"
        )
        self.console.print("[italic]Every. Single. Time.[/italic]\n")


async def main():
    runner = DemoPitchRunner()
    await runner.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
