"""
LangChain Implementation: Same Customer Support Ticket Analyzer
'The path of a thousand imports and a prayer-based error handling strategy'
"""

import asyncio
import json
import logging
import traceback
from datetime import datetime
from typing import Any

from langchain.callbacks import AsyncCallbackHandler

# Welcome to import hell
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, validator
from tenacity import retry, stop_after_attempt, wait_exponential


# Output schemas (at least Pydantic works here too)
class TicketParserOutput(BaseModel):
    issue_type: str
    technical_details: str
    customer_emotion: str

    @validator("issue_type")
    def validate_issue_type(cls, v):
        valid = ["bug", "feature_request", "billing", "onboarding", "other"]
        if v not in valid:
            # Good luck debugging this in production
            raise ValueError(f"Invalid issue_type: {v}")
        return v


class EnterpriseAnalysisOutput(BaseModel):
    priority: str
    action_items: list[str]
    executive_summary: str


class StandardAnalysisOutput(BaseModel):
    priority: str
    suggested_response: str


# Custom callback handler for "observability" (aka print statements with extra steps)
class TicketAnalyzerCallback(AsyncCallbackHandler):
    def __init__(self):
        self.events = []
        self.errors = []

    async def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs):
        # Hope you like unstructured logs
        self.events.append({
            "type": "llm_start",
            "time": datetime.now().isoformat(),
            "prompts": prompts[:100],  # Truncate because why not
        })

    async def on_llm_error(self, error: Exception, **kwargs):
        # Error handling: catch it, log it, pray it doesn't happen in prod
        self.errors.append({
            "type": "llm_error",
            "error": str(error),
            "traceback": traceback.format_exc(),
        })

    async def on_chain_error(self, error: Exception, **kwargs):
        # Chain errors are special, they get their own handler
        self.errors.append({"type": "chain_error", "error": str(error)})


class CustomerSupportAnalyzer:
    def __init__(self):
        # Initialize all the models (hope you budgeted for this)
        self.parser_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            request_timeout=10,  # This might work
            max_retries=3,  # Or might not
        )

        self.enterprise_llm = ChatOpenAI(model="gpt-4", temperature=0.3, max_tokens=1000)

        self.standard_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=500)

        # Set up parsers (with fixing parser because LLMs never follow schemas)
        self.ticket_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=TicketParserOutput), llm=self.parser_llm
        )

        self.enterprise_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=EnterpriseAnalysisOutput),
            llm=self.enterprise_llm,
        )

        self.standard_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=StandardAnalysisOutput),
            llm=self.standard_llm,
        )

        # Callbacks for "observability"
        self.callbacks = [TicketAnalyzerCallback()]

        # Manual schema validation because we trust nobody
        self.validated_inputs = []
        self.validated_outputs = []

    def validate_input(self, ticket_text: str, customer_tier: str) -> tuple[bool, str]:
        """Manual input validation because decorators are for quitters"""
        if not ticket_text or len(ticket_text) < 10:
            return False, "Ticket text too short"
        if len(ticket_text) > 5000:
            return False, "Ticket text too long"
        if customer_tier not in ["bronze", "silver", "gold", "enterprise"]:
            return False, f"Invalid customer tier: {customer_tier}"
        return True, "Valid"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def parse_ticket(self, ticket_text: str, customer_tier: str) -> dict[str, Any]:
        """Step 1: Parse the ticket (with retry decorator because it will fail)"""
        prompt = PromptTemplate(
            input_variables=["customer_tier", "ticket_text"],
            template="""Extract the following from this support ticket:
Customer Tier: {customer_tier}
Ticket: {ticket_text}

Return as JSON with keys: issue_type, technical_details, customer_emotion

{format_instructions}""",
        )

        chain = LLMChain(llm=self.parser_llm, prompt=prompt, callbacks=self.callbacks)

        try:
            # Run chain and pray
            result = await chain.arun(
                customer_tier=customer_tier,
                ticket_text=ticket_text,
                format_instructions=self.ticket_parser.get_format_instructions(),
            )

            # Parse and pray harder
            parsed = self.ticket_parser.parse(result)
            return parsed.dict()
        except Exception as e:
            # When all else fails, return something
            logging.error(f"Parsing failed: {e}")
            return {
                "issue_type": "other",
                "technical_details": "Error parsing ticket",
                "customer_emotion": "frustrated",
            }

    async def route_by_tier(
        self, customer_tier: str, parsed_ticket: dict[str, Any]
    ) -> dict[str, Any]:
        """Step 2: Conditional routing (if-else with extra steps)"""
        if customer_tier == "enterprise":
            return await self.analyze_enterprise(parsed_ticket)
        return await self.analyze_standard(parsed_ticket)

    async def analyze_enterprise(self, parsed_ticket: dict[str, Any]) -> dict[str, Any]:
        """Step 3A: Enterprise analysis (expensive and slow)"""
        prompt = ChatPromptTemplate.from_template("""ENTERPRISE CUSTOMER ALERT!
Issue Type: {issue_type}
Technical Details: {technical_details}
Emotion: {customer_emotion}

Provide:
1. Root cause analysis
2. Immediate mitigation steps
3. Long-term solution
4. Executive summary for account manager

{format_instructions}""")

        chain = LLMChain(llm=self.enterprise_llm, prompt=prompt, callbacks=self.callbacks)

        try:
            result = await chain.arun(
                issue_type=parsed_ticket["issue_type"],
                technical_details=parsed_ticket["technical_details"],
                customer_emotion=parsed_ticket["customer_emotion"],
                format_instructions=self.enterprise_parser.get_format_instructions(),
            )

            parsed = self.enterprise_parser.parse(result)
            return parsed.dict()
        except Exception as e:
            logging.error(f"Enterprise analysis failed: {e}")
            # Default response when things go wrong
            return {
                "priority": "high",
                "action_items": ["Investigate issue", "Contact customer"],
                "executive_summary": "Analysis failed, manual intervention required",
            }

    async def analyze_standard(self, parsed_ticket: dict[str, Any]) -> dict[str, Any]:
        """Step 3B: Standard analysis (cheap and cheerful)"""
        prompt = ChatPromptTemplate.from_template("""Issue: {issue_type}
Details: {technical_details}

Assign priority and suggest response template.

{format_instructions}""")

        chain = LLMChain(llm=self.standard_llm, prompt=prompt, callbacks=self.callbacks)

        try:
            result = await chain.arun(
                issue_type=parsed_ticket["issue_type"],
                technical_details=parsed_ticket["technical_details"],
                format_instructions=self.standard_parser.get_format_instructions(),
            )

            parsed = self.standard_parser.parse(result)
            return parsed.dict()
        except Exception as e:
            logging.error(f"Standard analysis failed: {e}")
            return {
                "priority": "medium",
                "suggested_response": "Thank you for contacting support. We're looking into this.",
            }

    def generate_response(self, tier: str, parser_output: dict, analysis: dict) -> dict[str, Any]:
        """Step 4: Generate response (no LLM needed, just logic)"""
        import random
        import string
        import time

        # Generate ticket ID using timestamp and random suffix (not for security purposes)
        # nosec B311 - This is for demo ticket IDs only, not security-sensitive
        timestamp = int(time.time() * 1000) % 1000000
        suffix = "".join(random.choices(string.digits, k=3))  # nosec B311
        ticket_id = f"TKT-{timestamp:06d}-{suffix}"

        # Build response based on tier
        if tier == "enterprise":
            response_text = f"Dear valued enterprise customer, {analysis.get('executive_summary', 'We are addressing your issue with highest priority.')}"
            assigned_team = "enterprise-support"
        else:
            response_text = analysis.get("suggested_response", "Thank you for your patience.")
            assigned_team = "standard-support"

        return {
            "ticket_id": ticket_id,
            "response_text": response_text,
            "internal_notes": f"{parser_output.get('technical_details', 'No details')}",
            "assigned_team": assigned_team,
        }

    async def send_notifications(self, response: dict[str, Any], priority: str):
        """Step 5: Send notifications (fake async for that enterprise feel)"""
        tasks = []

        # Slack notification
        async def notify_slack():
            await asyncio.sleep(0.1)  # Simulate API call
            print(f"Slack: New {response['assigned_team']} ticket: {response['ticket_id']}")
            return True

        # CRM update
        async def update_crm():
            await asyncio.sleep(0.15)  # Simulate API call
            print(f"CRM: Updated ticket {response['ticket_id']} with priority {priority}")
            return True

        # Jira ticket
        async def create_jira():
            await asyncio.sleep(0.2)  # Simulate API call
            print(f"Jira: Created ticket for {response['ticket_id']}")
            return True

        # Run in parallel (the only good part)
        tasks = [notify_slack(), update_crm(), create_jira()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for failures (but don't do anything about them)
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            logging.warning(f"Some notifications failed: {failures}")

        return all(r for r in results if not isinstance(r, Exception))

    def validate_output(self, output: dict[str, Any]) -> tuple[bool, str]:
        """Manual output validation (because we can't trust anything)"""
        import re

        if "ticket_id" not in output:
            return False, "Missing ticket_id"

        if not re.match(r"^TKT-[0-9]{6}$", output["ticket_id"]):
            return False, f"Invalid ticket_id format: {output['ticket_id']}"

        if "processing_time_ms" in output and output["processing_time_ms"] > 30000:
            return False, f"SLA violation: {output['processing_time_ms']}ms > 30000ms"

        return True, "Valid"

    async def process_ticket(self, ticket_text: str, customer_tier: str) -> dict[str, Any]:
        """Main orchestration function (where dreams come to die).

        Raises
        ------
        ValueError
            If input validation fails
        """
        start_time = datetime.now()

        # Input validation
        valid, message = self.validate_input(ticket_text, customer_tier)
        if not valid:
            raise ValueError(f"Input validation failed: {message}")

        try:
            # Step 1: Parse ticket
            parsed_ticket = await self.parse_ticket(ticket_text, customer_tier)

            # Step 2 & 3: Route and analyze
            analysis = await self.route_by_tier(customer_tier, parsed_ticket)

            # Step 4: Generate response
            response = self.generate_response(customer_tier, parsed_ticket, analysis)

            # Step 5: Send notifications
            notifications_sent = await self.send_notifications(
                response, analysis.get("priority", "medium")
            )

            # Calculate processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Build final output
            output = {
                "ticket_id": response["ticket_id"],
                "response_sent": notifications_sent,
                "processing_time_ms": processing_time_ms,
                "response": response,
                "analysis": analysis,
            }

            # Output validation
            valid, message = self.validate_output(output)
            if not valid:
                logging.error(f"Output validation failed: {message}")
                # But return it anyway because what else can we do?

            return output

        except Exception as e:
            # Global exception handler (the last line of defense)
            logging.error(f"Critical error in process_ticket: {e}")
            logging.error(traceback.format_exc())

            # Return something so the caller doesn't crash
            return {
                "ticket_id": "TKT-000000",
                "response_sent": False,
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "error": str(e),
            }


# Usage example (pray it works)
async def main():
    analyzer = CustomerSupportAnalyzer()

    # Test case
    result = await analyzer.process_ticket(
        ticket_text="My enterprise application keeps crashing when I try to generate reports. This is affecting our quarterly review and we need this fixed ASAP!",
        customer_tier="enterprise",
    )

    print(json.dumps(result, indent=2))

    # Check callbacks for "observability"
    if analyzer.callbacks[0].errors:
        print("\n⚠️ Errors occurred during processing:")
        for error in analyzer.callbacks[0].errors:
            print(f"  - {error['type']}: {error['error']}")


if __name__ == "__main__":
    # Run and hope for the best
    asyncio.run(main())
