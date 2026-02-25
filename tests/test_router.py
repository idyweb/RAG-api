"""
Integration tests for the Semantic Router.

This tests the router against a 'Golden Dataset' of queries to ensure 
that intent classification remains accurate and does not hallucinate
or misroute queries spanning different departmental contexts.
"""

import pytest
from api.core.semantic_router import SemanticRouter, RoutedAgent

# The "Golden Dataset" of core queries to ensure router accuracy
GOLDEN_DATASET = [
    (
        "What is the maternity leave policy?",
        "Sales",
        RoutedAgent.RAG
    ),
    (
        "Show me the top selling SKUs in Angola this month compared to target.",
        "BID",
        RoutedAgent.POWER_BI
    ),
    (
        "Has Purchase Order 12345 been approved?",
        "ERP",
        RoutedAgent.ERP_API
    ),
    (
        "Show the maintenance history for packaging line 3 and any associated QMS incidents.",
        "Manufacturing",
        RoutedAgent.QMS
    ),
    (
        "Find the Q3 marketing presentation that John sent last week.",
        "Sales",
        RoutedAgent.DOCUMENT_SEARCH
    ),
    (
        "What is John's delivery route today?",
        "GTM",
        RoutedAgent.GTM_API
    )
]

@pytest.mark.asyncio
async def test_semantic_router_golden_dataset():
    """
    Test the Semantic Router against the golden dataset.
    This must pass 100% in CI/CD pipeline before any prompt updates are merged.
    """
    router = SemanticRouter()
    
    for query, department, expected_agent in GOLDEN_DATASET:
        decision = await router.route_query(query, department)
        
        # Verify the routing decision
        assert decision.routed_to == expected_agent, \
            f"Failed on query: '{query}'. Expected {expected_agent}, got {decision.routed_to}. Reasoning: {decision.reasoning}"
        
        # Verify confidence score is populated and reasonable
        assert 0.0 <= decision.confidence_score <= 1.0
        
        # Verify reasoning is populated
        assert len(decision.reasoning) > 0
