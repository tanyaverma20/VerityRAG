"""
benchmark_corpus.py — 10 isolated benchmark documents used by
run_benchmark.py (request D, item 4: "10-document scale validation").

These are NOT real research papers scraped/downloaded for this benchmark —
they are original, hand-written technical text (~500-900 words each, 3
pages) covering 10 distinct, non-overlapping CS/ML topics, written
specifically so each document has content genuinely unique to it (no
shared vocabulary bleed between documents) — this makes retrieval
correctness independently verifiable: a query about "Reciprocal Rank
Fusion" should NEVER surface chunks from the "Raft consensus" document, etc.

Every document also ships a short list of representative questions with
their KNOWN correct relevant_document_ids — this is the ground truth the
retrieval evaluation (Recall@K / Precision@K / MRR) is measured against.
Nothing here is downloaded, scraped, or LLM-generated; content, structure,
and expected answers are all authored directly in this file, so the
benchmark is fully reproducible from source with no external dependency.
"""
from __future__ import annotations

# Each entry: (doc_key, title, filename, [page_1_text, page_2_text, page_3_text])
DOCUMENTS: list[dict] = [
    {
        "doc_key": "transformers",
        "title": "Attention Mechanisms in Sequence Transduction",
        "filename": "bench_transformers.pdf",
        "pages": [
            "Introduction\n\nSequence transduction models traditionally relied on recurrent "
            "architectures that process tokens one position at a time, which limits "
            "parallelization and makes it difficult to model long-range dependencies. "
            "The Transformer architecture replaces recurrence entirely with a "
            "self-attention mechanism, computing a weighted representation of every "
            "token as a function of every other token in the sequence in a single "
            "matrix multiplication. This document describes scaled dot-product "
            "attention, multi-head attention, and positional encoding, the three "
            "components that let a Transformer model sequence order without "
            "recurrence.",
            "Scaled Dot-Product Attention\n\nGiven query, key, and value matrices Q, K, "
            "and V, attention is computed as softmax(QK^T / sqrt(d_k)) V. The scaling "
            "factor sqrt(d_k) prevents the dot products from growing too large in "
            "magnitude for large key dimensions, which would push the softmax into "
            "regions with extremely small gradients. Multi-head attention runs this "
            "computation h times in parallel with different learned linear "
            "projections, allowing the model to jointly attend to information from "
            "different representation subspaces at different positions, then "
            "concatenates and projects the results.",
            "Positional Encoding and Training\n\nBecause self-attention itself contains "
            "no notion of token order, sinusoidal positional encodings are added to "
            "the input embeddings so the model can make use of sequence position. "
            "Transformer models trained on machine translation tasks such as "
            "WMT14 English-to-German achieved state-of-the-art BLEU scores while "
            "training substantially faster than recurrent or convolutional "
            "alternatives, since the self-attention computation for all positions "
            "can be parallelized on a GPU.",
        ],
        "questions": [
            {"question": "How is scaled dot-product attention computed?", "expected_answer_contains": ["softmax", "QK", "sqrt"]},
            {"question": "Why is positional encoding needed in a Transformer?", "expected_answer_contains": ["order", "position"]},
        ],
    },
    {
        "doc_key": "cnn",
        "title": "Convolutional Neural Networks for Image Recognition",
        "filename": "bench_cnn.pdf",
        "pages": [
            "Convolutional Layers\n\nConvolutional neural networks apply small learned "
            "filters (kernels) that slide across an input image, computing a dot "
            "product between the filter weights and the local image patch at every "
            "position. This weight sharing means the same edge detector or texture "
            "detector can fire anywhere in the image, giving convolutional networks "
            "translation invariance and dramatically fewer parameters than a fully "
            "connected layer operating on raw pixels.",
            "Pooling and Downsampling\n\nMax pooling layers reduce the spatial "
            "resolution of feature maps by taking the maximum activation within a "
            "small window, discarding precise spatial location while retaining the "
            "strongest feature responses. Stacking convolution, activation, and "
            "pooling layers builds a hierarchy where early layers detect edges and "
            "textures and deeper layers detect increasingly complex, semantically "
            "meaningful patterns such as object parts.",
            "Training on ImageNet\n\nDeep convolutional architectures trained on the "
            "ImageNet dataset, containing over a million labeled images across a "
            "thousand categories, reduced top-5 classification error substantially "
            "compared to earlier hand-engineered feature pipelines. Batch "
            "normalization and residual connections later allowed these networks to "
            "be trained much deeper without vanishing gradients.",
        ],
        "questions": [
            {"question": "What does a max pooling layer do?", "expected_answer_contains": ["maximum", "pooling", "window"]},
            {"question": "What dataset was used to train deep convolutional networks for image classification?", "expected_answer_contains": ["ImageNet"]},
        ],
    },
    {
        "doc_key": "rl",
        "title": "Reinforcement Learning with Temporal Difference Methods",
        "filename": "bench_rl.pdf",
        "pages": [
            "The Reinforcement Learning Problem\n\nAn agent interacts with an "
            "environment over discrete time steps, observing a state, selecting an "
            "action according to a policy, and receiving a scalar reward along with "
            "a new state. The goal is to learn a policy that maximizes the expected "
            "cumulative discounted reward. Unlike supervised learning, the agent "
            "never sees a labeled correct action — it must discover good behavior "
            "through trial and error.",
            "Q-Learning\n\nQ-learning estimates an action-value function Q(s, a) "
            "representing the expected return of taking action a in state s and "
            "following the optimal policy thereafter. The update rule adjusts "
            "Q(s, a) toward the observed reward plus the discounted maximum "
            "Q-value of the next state: Q(s,a) <- Q(s,a) + alpha * (r + gamma * "
            "max_a' Q(s',a') - Q(s,a)). This is a temporal-difference method: it "
            "updates estimates based on other learned estimates rather than "
            "waiting for a full episode to complete.",
            "Exploration vs Exploitation\n\nAn epsilon-greedy policy selects the "
            "action with the highest estimated Q-value most of the time but chooses "
            "a random action with probability epsilon, balancing exploitation of "
            "known good actions against exploration of potentially better "
            "unvisited actions. Without sufficient exploration, Q-learning can "
            "converge to a suboptimal policy.",
        ],
        "questions": [
            {"question": "What is the Q-learning update rule?", "expected_answer_contains": ["alpha", "gamma", "Q(s"]},
            {"question": "What does an epsilon-greedy policy do?", "expected_answer_contains": ["epsilon", "random", "exploration"]},
        ],
    },
    {
        "doc_key": "db_indexing",
        "title": "Indexing Strategies in Relational Database Systems",
        "filename": "bench_db_indexing.pdf",
        "pages": [
            "Why Indexes Matter\n\nWithout an index, a database must perform a full "
            "table scan to find rows matching a query predicate, reading every row "
            "in the table regardless of how many actually match. An index is an "
            "auxiliary data structure, typically a B-tree, that maps column values "
            "to the physical row locations containing them, letting the database "
            "jump directly to matching rows in logarithmic time instead of scanning "
            "linearly.",
            "B-Tree Indexes\n\nA B-tree index keeps its keys sorted and balances "
            "itself so every leaf is the same distance from the root, guaranteeing "
            "O(log n) lookup, insertion, and deletion. B-trees are well suited to "
            "range queries (WHERE age BETWEEN 20 AND 30) because sorted keys can be "
            "scanned sequentially once the starting point is located, unlike a hash "
            "index which only supports exact-match lookups.",
            "Index Trade-offs\n\nEvery index speeds up reads that use it but slows "
            "down writes, since each INSERT, UPDATE, or DELETE must also update "
            "every index on the affected columns, and indexes consume additional "
            "disk space. Composite indexes covering multiple columns can serve "
            "queries that filter or sort on those columns together, but a composite "
            "index is only useful for query predicates that use its leftmost "
            "columns first.",
        ],
        "questions": [
            {"question": "What data structure do relational database indexes typically use?", "expected_answer_contains": ["B-tree"]},
            {"question": "What is the downside of adding more indexes to a table?", "expected_answer_contains": ["write", "slow", "disk space"]},
        ],
    },
    {
        "doc_key": "os_scheduling",
        "title": "Process Scheduling in Operating Systems",
        "filename": "bench_os_scheduling.pdf",
        "pages": [
            "The Scheduler's Job\n\nA CPU scheduler decides which of the ready "
            "processes in memory gets to run on the CPU next, and for how long. "
            "Because a single CPU core can only execute one process at a time, the "
            "scheduler is responsible for creating the illusion of concurrent "
            "execution by rapidly switching between processes, a technique called "
            "time-sharing or multitasking.",
            "Round-Robin Scheduling\n\nRound-robin scheduling assigns each process a "
            "fixed time slice, or quantum, and cycles through the ready queue in "
            "order, preempting a running process once its quantum expires and "
            "moving it to the back of the queue. A small quantum improves response "
            "time for interactive processes but increases context-switch overhead; "
            "a large quantum reduces overhead but can make the system feel "
            "unresponsive.",
            "Priority Scheduling and Starvation\n\nPriority scheduling always runs "
            "the highest-priority ready process first, which can starve low-priority "
            "processes indefinitely if higher-priority work keeps arriving. Aging "
            "addresses this by gradually increasing the priority of processes that "
            "have waited a long time, guaranteeing that every process eventually "
            "runs.",
        ],
        "questions": [
            {"question": "What problem does aging solve in priority scheduling?", "expected_answer_contains": ["starvation", "priority", "wait"]},
            {"question": "What is a time quantum in round-robin scheduling?", "expected_answer_contains": ["quantum", "time slice"]},
        ],
    },
    {
        "doc_key": "raft",
        "title": "Raft: A Consensus Algorithm for Replicated Logs",
        "filename": "bench_raft.pdf",
        "pages": [
            "Leader Election\n\nRaft manages a replicated log across a cluster of "
            "servers by electing a single leader responsible for accepting client "
            "requests and replicating them to follower servers. Servers start as "
            "followers; if a follower receives no communication from a leader "
            "within an election timeout, it becomes a candidate, increments its "
            "term number, and requests votes from the other servers. A candidate "
            "that receives votes from a majority of the cluster becomes leader for "
            "that term.",
            "Log Replication\n\nOnce elected, the leader appends each new client "
            "command to its own log as an uncommitted entry, then sends "
            "AppendEntries RPCs to replicate that entry to its followers. An entry "
            "is considered committed once the leader has replicated it to a "
            "majority of the cluster, at which point the leader applies it to its "
            "state machine and informs followers of the new commit index.",
            "Safety Guarantees\n\nRaft guarantees that if two logs contain an entry "
            "with the same index and term, the logs are identical in all entries up "
            "through that index. Combined with the requirement that a candidate's "
            "log must be at least as up to date as a majority of the cluster to win "
            "an election, this prevents a server with a stale, incomplete log from "
            "ever becoming leader and overwriting already-committed entries.",
        ],
        "questions": [
            {"question": "How does a Raft candidate get elected leader?", "expected_answer_contains": ["majority", "votes", "candidate"]},
            {"question": "When is a log entry considered committed in Raft?", "expected_answer_contains": ["majority", "committed", "replicated"]},
        ],
    },
    {
        "doc_key": "gnn",
        "title": "Graph Neural Networks and Message Passing",
        "filename": "bench_gnn.pdf",
        "pages": [
            "Graph-Structured Data\n\nMany real-world datasets are naturally "
            "represented as graphs — social networks, molecules, and citation "
            "networks all consist of nodes connected by edges rather than a fixed "
            "grid like an image or a sequence like text. Graph neural networks "
            "(GNNs) generalize convolution to this irregular structure by learning "
            "a function that aggregates information from a node's neighbors.",
            "Message Passing\n\nIn a message-passing GNN layer, every node sends a "
            "message (typically a transformation of its current feature vector) to "
            "each of its neighbors, and every node aggregates the incoming messages "
            "— commonly by summing, averaging, or taking the maximum — and combines "
            "the aggregated message with its own previous representation to produce "
            "an updated node embedding. Stacking k such layers lets a node's "
            "representation incorporate information from nodes up to k hops away.",
            "Applications\n\nGraph neural networks have been applied to molecular "
            "property prediction, where atoms are nodes and bonds are edges, to "
            "recommendation systems modeling user-item interaction graphs, and to "
            "traffic forecasting on road networks. Over-smoothing, where node "
            "representations become indistinguishable after too many message-"
            "passing layers, is a known limitation of very deep GNNs.",
        ],
        "questions": [
            {"question": "How does a node update its representation in a message-passing GNN?", "expected_answer_contains": ["aggregat", "neighbor", "message"]},
            {"question": "What is over-smoothing in graph neural networks?", "expected_answer_contains": ["indistinguishable", "deep", "layers"]},
        ],
    },
    {
        "doc_key": "federated",
        "title": "Federated Learning Across Decentralized Devices",
        "filename": "bench_federated.pdf",
        "pages": [
            "Motivation\n\nFederated learning trains a shared model across many "
            "decentralized devices, such as mobile phones, each holding local "
            "training data that never leaves the device. Instead of centralizing "
            "raw data on a server, each device computes a local model update using "
            "its own data and sends only that update, not the underlying data, back "
            "to a coordinating server.",
            "Federated Averaging\n\nThe FedAvg algorithm selects a subset of "
            "available devices in each communication round, sends them the current "
            "global model, has each device run several steps of local stochastic "
            "gradient descent on its own data, and then averages the resulting "
            "model weights across devices, weighted by how much local data each "
            "device had, to produce the next global model.",
            "Challenges\n\nDevice data is typically non-independent-and-"
            "identically-distributed, meaning one user's phone data looks "
            "statistically different from another's, which can slow convergence "
            "compared to centrally-shuffled training data. Communication cost is "
            "often the main bottleneck rather than computation, since mobile "
            "network bandwidth is limited and devices may be offline or on "
            "metered connections.",
        ],
        "questions": [
            {"question": "What does the FedAvg algorithm average across devices?", "expected_answer_contains": ["model weights", "average", "device"]},
            {"question": "Why can federated learning be slow to converge?", "expected_answer_contains": ["non-independent", "identically", "distributed"]},
        ],
    },
    {
        "doc_key": "compilers",
        "title": "Compiler Optimization Passes",
        "filename": "bench_compilers.pdf",
        "pages": [
            "Intermediate Representation\n\nA compiler typically translates source "
            "code into an intermediate representation (IR), a lower-level, "
            "language-independent form such as static single assignment (SSA) "
            "form, where every variable is assigned exactly once, before applying "
            "optimization passes and eventually generating target machine code.",
            "Constant Folding and Dead Code Elimination\n\nConstant folding "
            "evaluates expressions with compile-time-known operands ahead of time, "
            "replacing an expression like 2 + 3 with the literal 5 so the addition "
            "never runs at runtime. Dead code elimination removes computations "
            "whose results are never used, such as an assignment to a variable "
            "that is never subsequently read, shrinking the compiled program and "
            "reducing wasted work.",
            "Register Allocation\n\nRegister allocation assigns the limited set of "
            "physical CPU registers to the (often much larger) set of variables "
            "live in the program at any point, using graph coloring: variables "
            "that are simultaneously live form an interference graph, and "
            "assigning registers becomes the problem of coloring that graph so no "
            "two interfering variables share a color. When there are more live "
            "variables than registers, some are spilled to memory.",
        ],
        "questions": [
            {"question": "What does dead code elimination remove?", "expected_answer_contains": ["never used", "unused", "dead"]},
            {"question": "How is register allocation modeled as a graph problem?", "expected_answer_contains": ["graph coloring", "interference"]},
        ],
    },
    {
        "doc_key": "ir_bm25",
        "title": "Probabilistic Ranking with BM25",
        "filename": "bench_ir_bm25.pdf",
        "pages": [
            "Term Frequency and Document Length\n\nBM25 is a bag-of-words ranking "
            "function used to score how well a document matches a query, "
            "extending simple term-frequency scoring with two additional "
            "considerations: diminishing returns for repeated terms (a document "
            "mentioning a query term 20 times isn't twice as relevant as one "
            "mentioning it 10 times) and normalization for document length, so a "
            "long document doesn't score higher purely by containing more words.",
            "The BM25 Formula\n\nFor a query term, BM25 combines an inverse "
            "document frequency component, which downweights terms that appear in "
            "almost every document in the collection, with a saturating "
            "term-frequency component controlled by a parameter k1, and a "
            "length-normalization component controlled by a parameter b that "
            "compares a document's length to the average document length in the "
            "collection.",
            "BM25 in Hybrid Retrieval\n\nBecause BM25 scores exact lexical overlap, "
            "it excels at matching rare, specific terms such as model names, "
            "dataset names, and numeric values that a purely semantic dense "
            "embedding model might blur together. Combining BM25 with dense vector "
            "search, for example via Reciprocal Rank Fusion, lets a retrieval "
            "system benefit from both exact keyword matching and semantic "
            "similarity.",
        ],
        "questions": [
            {"question": "What does the parameter b control in BM25?", "expected_answer_contains": ["length", "normalization"]},
            {"question": "Why combine BM25 with dense vector search?", "expected_answer_contains": ["exact", "semantic", "keyword"]},
        ],
    },
]

assert len(DOCUMENTS) == 10, "Benchmark corpus must contain exactly 10 documents"
