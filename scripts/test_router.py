"""
Mock script to test Semantic Router decision accuracy on the 'Golden Dataset'.
"""

import sys
import asyncio
from typing import TypedDict
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.core.semantic_router import SemanticRouter, RoutedAgent

class TestCase(TypedDict):
    query: str
    department: str
    expected: RoutedAgent

GOLDEN_DATASET: list[TestCase] = [
    {
        "query": "What is the maternity leave policy?",
        "department": "Sales",
        "expected": RoutedAgent.RAG
    },
    {
        "query": "Show me the top selling SKUs in Angola this month compared to target.",
        "department": "BID",
        "expected": RoutedAgent.POWER_BI
    },
    {
        "query": "Has Purchase Order 12345 been approved?",
        "department": "ERP",
        "expected": RoutedAgent.ERP_API
    },
    {
        "query": "Show the maintenance history for packaging line 3 and any associated QMS incidents.",
        "department": "Manufacturing",
        "expected": RoutedAgent.QMS
    },
    {
        "query": "Find the Q3 marketing presentation that John sent last week.",
        "department": "Sales",
        "expected": RoutedAgent.DOCUMENT_SEARCH
    },
    {
        "query": "What is John's delivery route today?",
        "department": "GTM",
        "expected": RoutedAgent.GTM_API
    },
    {
        "query": "I want to complain about the food in the cafeteria.",
        "department": "OPS",
        "expected": RoutedAgent.RAG  # Usually HR/Admin handles these via RAG policy
    }
]

async def run_tests():
    print("Initializing Semantic Router...")
    router = SemanticRouter()
    
    success_count = 0
    total = len(GOLDEN_DATASET)
    
    print("\nStarting tests...\n" + "="*50)
    for i, test in enumerate(GOLDEN_DATASET, 1):
        print(f"Test {i}/{total}: '{test['query']}'")
        decision = await router.route_query(test['query'], test['department'])
        
        if decision.routed_to == test['expected']:
            print(f"✅ PASS: Routed to {decision.routed_to.value} (Conf: {decision.confidence_score:.2f})")
            success_count += 1
        else:
            print(f"❌ FAIL: Expected {test['expected'].value}, got {decision.routed_to.value}")
            print(f"Reasoning provided: {decision.reasoning}")
        print("-" * 50)
        
    print(f"\nFinal Score: {success_count}/{total} ({(success_count/total)*100:.1f}%)")
    
    if success_count < total:
        print("\nWARNING: Router accuracy is below 100% on the golden dataset.")
        sys.exit(1)
    else:
        print("\nSUCCESS: Router accuracy is 100%. Ready for Integration!")

if __name__ == "__main__":
    asyncio.run(run_tests())
