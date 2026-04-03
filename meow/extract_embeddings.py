"""
Meow Embedding Extraction - Extract embeddings from text using HuggingFace models.

Generates training data for codebook learning by encoding agent-relevant
text through a transformer model and saving the hidden-state embeddings.

Usage:
    python -m meow.extract_embeddings                              # defaults
    python -m meow.extract_embeddings --model bert-base-uncased    # custom model
    python -m meow.extract_embeddings --corpus path/to/texts.txt   # custom corpus
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


# --- Built-in agent-relevant text corpus ---
# Covers: code tasks, reasoning, planning, debugging, collaboration
AGENT_CORPUS = [
    # Code tasks
    "Refactor the authentication module to use Redis for session storage",
    "Fix the race condition in the connection pool implementation",
    "Add input validation to the user registration endpoint",
    "Optimize the database query that scans the full orders table",
    "Migrate the legacy REST API endpoints to GraphQL",
    "Implement rate limiting middleware for the public API",
    "Write unit tests for the payment processing service",
    "Debug the memory leak in the WebSocket handler",
    "Add retry logic with exponential backoff to the HTTP client",
    "Refactor the monolithic service into microservices",
    "Update the CI/CD pipeline to support parallel test execution",
    "Implement a circuit breaker pattern for external API calls",
    "Add structured logging with correlation IDs across services",
    "Fix the deadlock in the concurrent cache implementation",
    "Create a database migration for the new user permissions schema",
    "Implement OAuth 2.0 authorization code flow",
    "Add pagination support to the search results endpoint",
    "Optimize the image processing pipeline for batch uploads",
    "Fix the CORS configuration for cross-origin requests",
    "Implement WebSocket-based real-time notifications",

    # Reasoning and analysis
    "The error occurs because the transaction isolation level allows dirty reads",
    "This approach has O(n^2) time complexity which becomes a bottleneck at scale",
    "The root cause is a race condition between the cache invalidation and the read path",
    "We should use eventual consistency here because strong consistency would add 200ms latency",
    "The memory usage spikes because the stream isn't being consumed incrementally",
    "This design violates the single responsibility principle",
    "The test is flaky because it depends on wall clock timing rather than logical ordering",
    "Using a bloom filter here would reduce the false positive rate to 1%",
    "The performance regression is caused by the N+1 query pattern in the ORM",
    "We need to handle the partial failure case when one of the downstream services is unavailable",

    # Planning and architecture
    "First we need to define the API contract, then implement the service layer",
    "The migration should be done in three phases: schema change, backfill, cleanup",
    "We'll use feature flags to gradually roll out the new recommendation engine",
    "The system should handle 10,000 requests per second with p99 latency under 100ms",
    "Let's start with a monolith and extract services as the domain boundaries become clear",
    "The data pipeline should be idempotent so we can safely replay failed batches",
    "We need to add an index on created_at before deploying the new query",
    "The rollback plan is to revert the deployment and restore from the last backup",
    "Split the frontend bundle into lazy-loaded chunks to improve initial load time",
    "Use event sourcing for the order lifecycle to maintain a complete audit trail",

    # Debugging and investigation
    "The stack trace shows a null pointer exception at line 142 in UserService.java",
    "The logs indicate the connection was reset by the upstream proxy after 30 seconds",
    "The CPU profile shows 60% of time is spent in JSON serialization",
    "The error rate increased from 0.1% to 5% after the last deployment",
    "Memory usage grows linearly with the number of active connections",
    "The request is timing out because the DNS resolution is taking 3 seconds",
    "The test fails on CI but passes locally due to a timezone difference",
    "The deadlock occurs when thread A holds lock X and waits for lock Y while thread B does the opposite",
    "The heap dump shows 500MB of unreachable string objects from the template engine",
    "Packet capture shows the TLS handshake fails because the certificate chain is incomplete",

    # Collaboration and coordination
    "I'll handle the backend changes while you update the frontend components",
    "The API changes are backward compatible so we can deploy independently",
    "Let's merge the feature branch after the integration tests pass",
    "I've identified three files that need to change: auth.py, middleware.py, and config.py",
    "The dependency update requires coordinated changes across three repositories",
    "Please review the schema migration before I run it in staging",
    "The shared library update will affect all downstream consumers",
    "I'm blocked on the API design review, can you take a look?",
    "The refactoring is safe to merge because all existing tests still pass",
    "We should pair on this because it touches both the caching layer and the database",

    # Multi-agent specific
    "Agent A should analyze the codebase structure while Agent B reviews the test coverage",
    "The partial results from the first analysis suggest the bug is in the auth module",
    "Combining hypothesis 1 and hypothesis 3 explains all the observed failures",
    "My confidence in this diagnosis is high based on the stack trace evidence",
    "I need the output from your code analysis to proceed with the fix",
    "The distributed trace shows the bottleneck is in the third service in the chain",
    "Let me verify your finding by checking the database logs independently",
    "Your proposed fix addresses the symptom but not the root cause",
    "I agree with the approach but suggest we also add a regression test",
    "The combined analysis points to a configuration drift between staging and production",

    # Uncertainty and hypotheses
    "This could be caused by either a network partition or a clock skew issue",
    "I'm 80% confident the bug is in the serialization layer",
    "There are two possible explanations: a race condition or a stale cache entry",
    "The evidence supports hypothesis A more than hypothesis B",
    "I'm uncertain whether this optimization will help without profiling data",
    "The fix might introduce a regression in the edge case where the queue is empty",
    "We need more data to determine if this is a systematic issue or a one-off",
    "Preliminary analysis suggests the issue is related to connection pooling",
    "The correlation is suggestive but we can't establish causation without an experiment",
    "Given the constraints, the best tradeoff is to optimize for latency over throughput",

    # Short functional messages
    "Ready",
    "Done",
    "Failed: timeout",
    "Tests passed",
    "Build succeeded",
    "Deploying to staging",
    "Rolling back",
    "Investigating",
    "Confirmed",
    "Blocked",
    "In progress",
    "Needs review",
    "Approved",
    "Merged",
    "Reverted",

    # Complex multi-step instructions
    "Step 1: Create the new table. Step 2: Backfill data from the old table. Step 3: Update the application to read from the new table. Step 4: Drop the old table.",
    "First, run the linter to check for style issues. Then, run the unit tests. If they pass, run the integration tests. Finally, deploy to the staging environment.",
    "The fix requires changes in three layers: update the database schema, modify the repository layer to use the new columns, and update the API response format.",
    "To reproduce: send a POST request with a payload larger than 10MB, wait for the timeout, then immediately send a GET request to the same endpoint.",
    "The deployment sequence is: apply database migration, deploy the new backend version, warm the caches, then switch the load balancer to the new instances.",

    # Data structures and algorithms
    "Use a priority queue to process tasks in order of urgency",
    "The trie structure gives us O(m) lookup where m is the key length",
    "Implement a skip list for the sorted index instead of a balanced BST",
    "The graph has a cycle, use topological sort to detect it",
    "Apply dynamic programming: the subproblems overlap along the diagonal",
    "The hash map has too many collisions, switch to cuckoo hashing",
    "A segment tree would support range queries in O(log n)",
    "The B-tree index is fragmented, rebuild it to improve scan performance",
    "Use union-find to detect connected components in the network graph",
    "The sliding window approach reduces this from O(n^2) to O(n)",
    "Implement an LRU cache with a doubly linked list and hash map",
    "The radix sort is faster here because the keys are bounded integers",
    "Use a monotone stack to find the next greater element efficiently",
    "The Dijkstra implementation needs a decrease-key operation",
    "Apply reservoir sampling to get a uniform random sample from the stream",

    # Infrastructure and DevOps
    "Scale the Kubernetes pods horizontally when CPU exceeds 80%",
    "The Terraform plan shows 3 resources to add and 1 to modify",
    "Configure the Nginx reverse proxy to handle WebSocket upgrades",
    "The Docker build cache is invalidated because the COPY layer changed",
    "Set up a blue-green deployment to minimize downtime during releases",
    "The Prometheus alert is firing because the error rate exceeded the SLO threshold",
    "Configure the S3 bucket lifecycle policy to transition to Glacier after 90 days",
    "The load balancer health check is failing because the endpoint returns 503",
    "Set up mTLS between the API gateway and the backend services",
    "The CloudWatch logs show the Lambda function is hitting the 15-minute timeout",
    "Configure the Redis cluster with 3 shards and 2 replicas per shard",
    "The Helm chart values need to be updated for the new environment",
    "Set up a VPN tunnel between the AWS VPC and the on-premises network",
    "The Kafka consumer group is lagging behind by 50,000 messages",
    "Configure the CDN to cache static assets with a 24-hour TTL",

    # Security
    "The SQL injection vulnerability is in the search parameter handler",
    "Rotate the API keys and invalidate all existing sessions",
    "The CSRF token is missing from the form submission",
    "Implement Content Security Policy headers to prevent XSS attacks",
    "The JWT token expiry is set to 30 days which is too long",
    "Add rate limiting to the login endpoint to prevent brute force attacks",
    "The S3 bucket policy allows public read access which should be restricted",
    "Encrypt the sensitive fields in the database at rest using AES-256",
    "The dependency audit found 3 critical vulnerabilities in transitive dependencies",
    "Implement RBAC for the admin API endpoints",
    "The OAuth redirect URI validation allows open redirect attacks",
    "Add HSTS headers with a max-age of one year including subdomains",
    "The password hashing uses MD5 which is insecure, migrate to bcrypt",
    "Implement certificate pinning for the mobile API client",
    "The audit log shows unauthorized access attempts from an internal IP",

    # Machine learning and data
    "The model accuracy dropped after retraining on the new dataset",
    "Apply L2 regularization to prevent overfitting on the small training set",
    "The feature importance analysis shows the top 3 features explain 80% of variance",
    "Use stratified sampling to maintain class balance in the test set",
    "The embedding space shows clear clustering after applying t-SNE",
    "Fine-tune the learning rate schedule: warm up for 1000 steps then cosine decay",
    "The batch normalization layers are causing issues during inference",
    "Apply gradient clipping to prevent exploding gradients in the RNN",
    "The attention weights show the model focuses on the wrong tokens",
    "Use mixed precision training to reduce memory usage and speed up training",
    "The confusion matrix shows high false positive rate for class 3",
    "Implement early stopping based on validation loss with patience of 10 epochs",
    "The data pipeline has a bottleneck in the preprocessing step",
    "Apply data augmentation: random crop, horizontal flip, and color jitter",
    "The model quantization reduced accuracy by only 0.5% but halved inference time",

    # Database operations
    "The query planner chooses a sequential scan instead of using the index",
    "Add a composite index on (user_id, created_at) for the activity feed query",
    "The connection pool is exhausted because long-running transactions hold connections",
    "Implement optimistic locking using a version column to prevent lost updates",
    "The database replication lag is 5 seconds which causes stale reads",
    "Partition the events table by month to improve query performance",
    "The vacuum process is blocked by a long-running idle transaction",
    "Use a materialized view to precompute the dashboard statistics",
    "The foreign key constraint is causing cascade delete performance issues",
    "Implement a write-ahead log for crash recovery in the custom storage engine",
    "The deadlock graph shows a cycle between the orders and inventory tables",
    "Use a cursor-based pagination instead of offset-based for large result sets",
    "The JSON column query is slow, extract the frequently queried fields into columns",
    "Implement change data capture to sync the database with the search index",
    "The database migration took 3 hours because it locked the table for ALTER",

    # API design
    "The REST endpoint should return 201 Created with a Location header",
    "Implement cursor-based pagination: return next_cursor in the response",
    "The GraphQL resolver has an N+1 query problem, use DataLoader",
    "Version the API using the URL path prefix: /api/v2/resources",
    "The webhook payload should include a signature header for verification",
    "Implement idempotency keys for the payment creation endpoint",
    "The gRPC service definition needs a streaming RPC for real-time updates",
    "Return proper error responses with error codes and human-readable messages",
    "The OpenAPI spec is out of date, regenerate it from the route definitions",
    "Implement request/response compression using gzip for payloads over 1KB",
    "The batch endpoint should accept up to 100 items and return partial results",
    "Add ETag headers for conditional requests to reduce bandwidth",
    "The rate limit response should include Retry-After and X-RateLimit headers",
    "Implement field selection: allow clients to specify which fields to return",
    "The long-polling endpoint should timeout after 30 seconds and return empty",

    # Testing
    "The integration test needs a database fixture with seed data",
    "Mock the external payment gateway to avoid real charges in tests",
    "The property-based test found an edge case with empty strings",
    "Add a smoke test that verifies all critical endpoints after deployment",
    "The test coverage is 45% for the auth module, aim for at least 80%",
    "Use a test container for the PostgreSQL dependency in CI",
    "The snapshot test needs updating after the UI component change",
    "Implement contract testing between the frontend and backend teams",
    "The load test shows p99 latency spikes to 2 seconds under 5000 RPS",
    "Add a fuzz test for the input parser to catch malformed payloads",
    "The E2E test is flaky because it depends on animation timing",
    "Use mutation testing to verify the test suite catches actual bugs",
    "The benchmark shows a 30% regression in the serialization path",
    "Add chaos engineering tests: randomly kill pods during integration tests",
    "The test matrix should cover Python 3.10, 3.11, and 3.12",

    # Frontend and UI
    "The React component re-renders 15 times on each keystroke",
    "Implement virtual scrolling for the list with 10,000 items",
    "The CSS animation jank is caused by layout thrashing in the scroll handler",
    "Lazy load the chart library to reduce initial bundle size by 200KB",
    "The form validation should show errors inline as the user types",
    "Implement optimistic UI updates for the like button interaction",
    "The accessibility audit found 12 missing aria labels on interactive elements",
    "Use intersection observer for lazy loading images below the fold",
    "The state management is scattered across 5 different stores, consolidate",
    "Implement skeleton screens instead of spinners for content loading",
    "The responsive layout breaks at 768px because of a fixed-width container",
    "Use CSS custom properties for the theme system instead of JS runtime styling",
    "The Web Vitals report shows LCP of 4.2 seconds, target is under 2.5",
    "Implement keyboard navigation for the dropdown menu component",
    "The service worker cache is stale, add a version-based cache busting strategy",

    # Distributed systems
    "The consensus algorithm needs at least 3 of 5 nodes to agree",
    "Implement a distributed lock using Redis with a TTL of 30 seconds",
    "The event bus guarantees at-least-once delivery, handle duplicates idempotently",
    "Use consistent hashing to distribute keys across the cache cluster",
    "The split-brain scenario caused both partitions to accept writes",
    "Implement a saga pattern for the multi-service order workflow",
    "The vector clock shows a concurrent update conflict on the same key",
    "Use gossip protocol for cluster membership and failure detection",
    "The eventual consistency window is 500ms which is acceptable for this use case",
    "Implement a circuit breaker with 5 failure threshold and 60 second reset",
    "The leader election uses Raft consensus with a 150ms election timeout",
    "Add request tracing with distributed trace IDs across all services",
    "The CRDT merge resolves the conflict automatically without coordination",
    "Implement backpressure: the consumer signals the producer to slow down",
    "The partition tolerance requires sacrificing either consistency or availability",

    # Code review feedback
    "This function does too many things, split it into smaller focused functions",
    "The variable name 'x' is not descriptive, rename it to 'userCount'",
    "Consider extracting this repeated pattern into a shared utility function",
    "The error handling swallows the exception silently, at least log it",
    "This SQL query is vulnerable to injection, use parameterized queries",
    "The magic number 86400 should be a named constant SECONDS_PER_DAY",
    "The nested ternary is hard to read, use an if-else block instead",
    "This test doesn't assert anything meaningful, it just checks existence",
    "The import cycle between these two modules needs to be broken",
    "Consider using an enum instead of string literals for the status field",
    "The recursive function has no base case for the empty input",
    "This global mutable state will cause issues in concurrent execution",
    "The callback nesting is 5 levels deep, refactor to async/await",
    "The API response includes internal database IDs which is a security risk",
    "The regular expression is too greedy, add a non-greedy quantifier",

    # Performance optimization
    "Profile shows 40% of CPU time in the JSON serialization hot path",
    "The N+1 query fires 500 individual SELECTs instead of a single JOIN",
    "Use connection pooling to avoid the TCP handshake overhead per request",
    "The garbage collector pauses for 200ms every 30 seconds under load",
    "Batch the Redis pipeline commands to reduce round trips from 100 to 1",
    "The memory allocation pattern causes severe heap fragmentation",
    "Use SIMD instructions for the vector similarity computation",
    "The file I/O is blocking the event loop, move it to a worker thread",
    "Cache the compiled regex pattern instead of recompiling per request",
    "The string concatenation in the loop creates O(n^2) intermediate objects",
    "Use a bloom filter to avoid expensive database lookups for missing keys",
    "The hot function is called 10M times, inline it to eliminate call overhead",
    "Pre-allocate the buffer to the expected size to avoid repeated resizing",
    "The lock contention on the shared counter causes 80% of threads to wait",
    "Use memory-mapped I/O for reading the large config file at startup",
]


def load_corpus(path: Optional[str] = None) -> List[str]:
    """Load text corpus from file or use built-in agent corpus."""
    if path is not None:
        with open(path, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts
    return AGENT_CORPUS


def extract_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: Optional[str] = None,
    augment_repeats: int = 1,
) -> np.ndarray:
    """
    Extract mean-pooled embeddings from a HuggingFace model.

    Args:
        texts: Input texts
        model_name: HuggingFace model name
        batch_size: Batch size for inference
        device: Device (auto-detect if None)
        augment_repeats: Repeat corpus N times (each pass may differ slightly due to dropout)

    Returns:
        embeddings: (N, hidden_dim) numpy array
    """
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []

    for repeat in range(augment_repeats):
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**encoded)

            # Mean pooling over non-padding tokens
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts

            all_embeddings.append(mean_pooled.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus
    texts = load_corpus(args.corpus)
    print(f"Corpus: {len(texts)} texts")

    # Extract embeddings
    print(f"Model: {args.model}")
    print(f"Augment repeats: {args.augment_repeats}")
    embeddings = extract_embeddings(
        texts,
        model_name=args.model,
        batch_size=args.batch_size,
        augment_repeats=args.augment_repeats,
    )
    print(f"Embeddings: {embeddings.shape} (samples × dim)")

    # Save
    pt_path = output_dir / "embeddings.pt"
    torch.save(torch.from_numpy(embeddings), pt_path)
    print(f"Saved: {pt_path}")

    # Save metadata
    meta = {
        "model": args.model,
        "num_texts": len(texts),
        "num_embeddings": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "augment_repeats": args.augment_repeats,
    }
    with open(output_dir / "embeddings_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Total samples: {embeddings.shape[0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract embeddings for Meow codebook training")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--corpus", type=str, default=None, help="Path to text file (one text per line)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augment-repeats", type=int, default=1, help="Repeat corpus N times")
    parser.add_argument("--output-dir", type=str, default="data")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
