#!/usr/bin/env python3
"""Example RAG evaluation script for HashiCorp documentation.

This script demonstrates how to evaluate RAG retrieval quality and
compare reranking configurations using the llm-api-server eval framework.

Usage:
    # From the llm-api-server directory:
    uv run python examples/rag_eval_hashicorp.py

    # Or from Ivan directory (where index exists):
    cd ~/Developer/Ivan
    uv run python -m examples.rag_eval_hashicorp

Requirements:
    - llm-api-server with RAG extra: uv sync --extra rag
    - An existing HashiCorp doc index (or it will build one)
"""

import sys
from pathlib import Path

# Add parent to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_tools_server.eval import RAGEvaluator, RAGTestCase, save_test_cases
from llm_tools_server.rag import DocSearchIndex, RAGConfig


# =============================================================================
# HASHICORP DOC TEST CASES
# =============================================================================
# These test cases cover common queries across HashiCorp products.
# Each test case has:
#   - query: The search query
#   - description: What we're testing
#   - relevant_urls: Ground truth - URLs that SHOULD be in top results
#   - relevant_keywords: Alternative - keywords that should appear in results
#
# To add more test cases, you can:
# 1. Think of common user queries
# 2. Manually find the authoritative doc page URL
# 3. Add it as a RAGTestCase below

HASHICORP_TEST_CASES = [
    # =========================================================================
    # VAULT
    # =========================================================================
    RAGTestCase(
        query="vault namespace configuration",
        description="Vault Enterprise namespaces setup",
        relevant_urls=[
            "https://developer.hashicorp.com/vault/tutorials/enterprise/namespaces",
            "https://developer.hashicorp.com/vault/tutorials/get-started-hcp-vault-dedicated/vault-namespaces",
            "https://developer.hashicorp.com/vault/api-docs/system/namespaces",
        ],
        metadata={"product": "vault", "category": "enterprise"},
    ),
    RAGTestCase(
        query="vault raft storage backend",
        description="Vault integrated storage (Raft)",
        relevant_urls=[
            "https://developer.hashicorp.com/vault/docs/configuration/storage/raft",
            "https://developer.hashicorp.com/vault/docs/concepts/integrated-storage",
            "https://developer.hashicorp.com/vault/tutorials/raft/raft-storage",
            "https://developer.hashicorp.com/vault/tutorials/day-one-raft/raft-deployment-guide",
        ],
        metadata={"product": "vault", "category": "storage"},
    ),
    RAGTestCase(
        query="vault kubernetes auth method",
        description="Vault Kubernetes authentication",
        relevant_urls=[
            "https://developer.hashicorp.com/vault/docs/auth/kubernetes",
            "https://developer.hashicorp.com/vault/api-docs/auth/kubernetes",
            "https://developer.hashicorp.com/vault/tutorials/kubernetes/kubernetes-sidecar",
            "https://developer.hashicorp.com/vault/docs/deploy/kubernetes/helm/examples/kubernetes-auth",
        ],
        metadata={"product": "vault", "category": "auth"},
    ),
    RAGTestCase(
        query="vault transit secrets engine encrypt decrypt",
        description="Vault Transit encryption as a service",
        relevant_urls=[
            "https://developer.hashicorp.com/vault/docs/secrets/transit",
            "https://developer.hashicorp.com/vault/api-docs/secret/transit",
            "https://developer.hashicorp.com/vault/tutorials/encryption-as-a-service/eaas-transit",
        ],
        metadata={"product": "vault", "category": "secrets"},
    ),
    RAGTestCase(
        query="vault disaster recovery replication",
        description="Vault DR replication setup",
        relevant_urls=[
            "https://developer.hashicorp.com/vault/tutorials/enterprise/disaster-recovery-replication-failover",
        ],
        relevant_keywords=["disaster", "recovery", "replication", "failover"],
        metadata={"product": "vault", "category": "enterprise"},
    ),
    # =========================================================================
    # TERRAFORM
    # =========================================================================
    RAGTestCase(
        query="terraform state locking",
        description="Terraform state file locking",
        relevant_urls=[
            "https://developer.hashicorp.com/terraform/language/state/locking",
        ],
        metadata={"product": "terraform", "category": "state"},
    ),
    RAGTestCase(
        query="terraform workspace management",
        description="Terraform workspaces for environment separation",
        relevant_urls=[
            "https://developer.hashicorp.com/terraform/tutorials/cloud-get-started/cloud-workspace-create",
            "https://developer.hashicorp.com/terraform/tutorials/cloud-get-started/cloud-workspace-configure",
        ],
        relevant_keywords=["workspace", "environment", "state"],
        metadata={"product": "terraform", "category": "state"},
    ),
    RAGTestCase(
        query="terraform module best practices",
        description="Terraform module design patterns",
        relevant_urls=[
            "https://developer.hashicorp.com/terraform/tutorials/modules/pattern-module-creation",
            "https://developer.hashicorp.com/terraform/tutorials/modules/module",
        ],
        relevant_keywords=["module", "source", "version", "outputs"],
        metadata={"product": "terraform", "category": "modules"},
    ),
    RAGTestCase(
        query="terraform cloud agents",
        description="TFE/TFC agent pools setup",
        relevant_urls=[
            "https://developer.hashicorp.com/terraform/cloud-docs/agents",
            "https://developer.hashicorp.com/terraform/cloud-docs/agents/agents",
            "https://developer.hashicorp.com/terraform/cloud-docs/agents/agent-pools",
            "https://developer.hashicorp.com/terraform/tutorials/cloud/cloud-agents",
        ],
        metadata={"product": "terraform", "category": "enterprise"},
    ),
    # =========================================================================
    # CONSUL
    # =========================================================================
    RAGTestCase(
        query="consul service mesh connect",
        description="Consul Connect service mesh",
        relevant_urls=[
            "https://developer.hashicorp.com/consul/docs/connect/proxy/mesh",
        ],
        relevant_keywords=["connect", "sidecar", "proxy", "intentions", "mesh"],
        metadata={"product": "consul", "category": "connect"},
    ),
    RAGTestCase(
        query="consul acl bootstrap tokens",
        description="Consul ACL system setup",
        relevant_urls=[
            "https://developer.hashicorp.com/consul/docs/secure/acl",
            "https://developer.hashicorp.com/consul/docs/secure/acl/bootstrap",
            "https://developer.hashicorp.com/consul/commands/acl/bootstrap",
        ],
        metadata={"product": "consul", "category": "security"},
    ),
    RAGTestCase(
        query="consul gossip encryption",
        description="Consul gossip protocol encryption",
        relevant_urls=[
            "https://developer.hashicorp.com/consul/docs/secure/encryption",
            "https://developer.hashicorp.com/consul/docs/secure/encryption/gossip/enable",
            "https://developer.hashicorp.com/consul/docs/concept/gossip",
        ],
        metadata={"product": "consul", "category": "security"},
    ),
    # =========================================================================
    # NOMAD
    # =========================================================================
    RAGTestCase(
        query="nomad job specification",
        description="Nomad job file format",
        relevant_urls=[
            "https://developer.hashicorp.com/nomad/tutorials/manage-jobs/jobs",
        ],
        relevant_keywords=["job", "group", "task", "driver"],
        metadata={"product": "nomad", "category": "jobs"},
    ),
    RAGTestCase(
        query="nomad acl policies tokens",
        description="Nomad ACL system",
        relevant_urls=[
            "https://developer.hashicorp.com/nomad/docs/secure/acl/policies",
            "https://developer.hashicorp.com/nomad/docs/secure/acl/tokens",
            "https://developer.hashicorp.com/nomad/tutorials/access-control/access-control",
        ],
        metadata={"product": "nomad", "category": "security"},
    ),
    # =========================================================================
    # BOUNDARY
    # =========================================================================
    RAGTestCase(
        query="boundary target host catalog",
        description="Boundary targets and hosts",
        relevant_urls=[
            "https://developer.hashicorp.com/boundary/docs/domain-model/targets",
            "https://developer.hashicorp.com/boundary/docs/domain-model/host-catalogs",
            "https://developer.hashicorp.com/boundary/tutorials/host-management/aws-host-catalogs",
            "https://developer.hashicorp.com/boundary/tutorials/community-administration/community-manage-targets",
        ],
        metadata={"product": "boundary", "category": "concepts"},
    ),
    # =========================================================================
    # CROSS-PRODUCT / VALIDATED DESIGNS
    # =========================================================================
    RAGTestCase(
        query="vault terraform enterprise integration",
        description="Using Vault with Terraform Enterprise",
        relevant_keywords=["vault", "terraform", "provider", "secrets"],
        metadata={"product": "cross", "category": "integration"},
    ),
    RAGTestCase(
        query="hashicorp validated design operating guide",
        description="HashiCorp validated designs overview",
        relevant_urls=[
            "https://developer.hashicorp.com/validated-designs/vault-operating-guides-adoption",
            "https://developer.hashicorp.com/validated-designs/terraform-operating-guides-adoption",
        ],
        relevant_keywords=["validated", "design", "operating", "guide"],
        metadata={"product": "cross", "category": "validated-designs"},
    ),
]


def main():
    """Run RAG evaluation on HashiCorp docs."""
    # Configuration - adjust paths as needed
    # Try Ivan's index first, fall back to local
    ivan_index = Path.home() / "Developer" / "Ivan" / "hashicorp_docs_index"
    local_index = Path("./hashicorp_docs_index")

    if ivan_index.exists():
        cache_dir = ivan_index
        print(f"Using existing Ivan index: {cache_dir}")
    elif local_index.exists():
        cache_dir = local_index
        print(f"Using local index: {cache_dir}")
    else:
        print("No existing index found. Building a new one...")
        print("This will crawl HashiCorp docs and may take a while.")
        cache_dir = local_index

    # Create RAG config
    config = RAGConfig(
        base_url="https://developer.hashicorp.com/validated-designs",
        cache_dir=str(cache_dir),
        manual_urls=[
            # Add some key doc pages as manual URLs
            "https://developer.hashicorp.com/vault/docs",
            "https://developer.hashicorp.com/terraform/docs",
            "https://developer.hashicorp.com/consul/docs",
            "https://developer.hashicorp.com/nomad/docs",
            "https://developer.hashicorp.com/boundary/docs",
        ],
        max_pages=500,  # Limit for faster testing
    )

    # Load or build index
    print("\nInitializing index...")
    index = DocSearchIndex(config)

    if cache_dir.exists() and (cache_dir / "chunks.json").exists():
        print("Loading existing index from cache...")
        index.load_index()
    else:
        print("Building new index (this may take several minutes)...")
        index.crawl_and_index()

    print(f"\nIndex ready with {len(index.chunks)} chunks")

    # Create evaluator
    evaluator = RAGEvaluator(index)

    # Run basic evaluation
    print("\n" + "=" * 70)
    print(" Running RAG Evaluation")
    print("=" * 70)

    results = evaluator.run_tests(HASHICORP_TEST_CASES)
    evaluator.print_summary(results, title="HashiCorp Docs RAG Evaluation")

    # Show per-test details for failed tests
    print("\nFailed Tests Details:")
    print("-" * 70)
    failed = [r for r in results if not r.passed]
    if not failed:
        print("All tests passed!")
    else:
        for r in failed:
            print(f"\nQuery: {r.test_case.query}")
            print(f"Description: {r.test_case.description}")
            print(f"Expected URLs: {r.test_case.relevant_urls}")
            print(f"Got URLs: {[res['url'] for res in r.retrieved_results[:3]]}")
            print(f"Metrics: Recall={r.recall:.2%}, MRR={r.mrr:.3f}")

    # A/B comparison: reranking on vs off
    print("\n" + "=" * 70)
    print(" A/B Test: Reranking ON vs OFF")
    print("=" * 70)

    comparison = evaluator.run_ab_comparison(
        HASHICORP_TEST_CASES,
        config_a={"rerank_enabled": True},
        config_b={"rerank_enabled": False},
    )
    evaluator.print_ab_comparison(comparison)

    # Save test cases for future use
    save_test_cases(HASHICORP_TEST_CASES, "hashicorp_rag_test_cases.json")

    print("\nDone! Test cases saved to hashicorp_rag_test_cases.json")


if __name__ == "__main__":
    main()
