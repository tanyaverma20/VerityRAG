import json
from pathlib import Path

# Mapping of file names to document IDs based on the ingestion manifest
DOCS = {
    "bert": "5692a5514787a8c6",      # 1810.04805v2.pdf (BERT)
    "t5": "ad07e47cedf437a4",        # 1910.10683v4.pdf (T5)
    "realm": "0ec1a7ab3de5514c",     # 2002.08909v1.pdf (REALM)
    "dpr": "3e67fc1a9977715a",       # 2004.04906v3.pdf (DPR / Dense Passage Retrieval)
    "rag": "23e3249e9a1e7541",       # 2005.11401v4.pdf (RAG)
    "gpt3": "97fd272f1fdfc186",      # 2005.14165v4.pdf (GPT-3)
    "moco": "f1ee7e927c3fb31b",      # 2007.01282v2.pdf (MoCo / SimCLR / Contrastive - assuming vision/contrastive)
    "mistral": "653b29619eae2ae4",   # 2307.03172v3.pdf (Mistral)
    "llama2": "1df284ce95f78300",    # 2307.09288v2.pdf (Llama 2)
    "transformer": "bdfaa68d8984f0dc",# attention.pdf (Attention Is All You Need)
    "edu": "a9b78223f7f2e3fb",       # EJ1172284.pdf (Educational / Pedagogy)
}

questions = []
q_id = 1

def add_q(question, q_type, expected, docs, comparison):
    global q_id
    questions.append({
        "id": f"q{q_id:03d}",
        "question": question,
        "question_type": q_type,
        "expected_answer_contains": expected,
        "relevant_document_ids": [DOCS[d] for d in docs],
        "relevant_chunk_ids": [],
        "comparison_required": comparison
    })
    q_id += 1


# 1. Single-paper factual (20)
add_q("What is the Transformer architecture and how does the attention mechanism work?", "factual", ["attention", "encoder", "decoder", "multi-head"], ["transformer"], False)
add_q("How is BERT pre-trained and what are the two main pre-training tasks?", "factual", ["masked", "next sentence", "pre-training"], ["bert"], False)
add_q("Describe the Text-to-Text Transfer Transformer (T5) framework.", "factual", ["text-to-text", "transfer learning", "transformer"], ["t5"], False)
add_q("How does REALM integrate retrieval during language model pre-training?", "factual", ["retrieval", "pre-training", "knowledge", "augment"], ["realm"], False)
add_q("What is the difference between RAG-Sequence and RAG-Token models?", "factual", ["sequence", "token", "generation", "marginalize"], ["rag"], False)
add_q("How does Dense Passage Retrieval (DPR) construct its dual-encoder architecture?", "factual", ["dual-encoder", "dense", "passage", "question"], ["dpr"], False)
add_q("What is the parameter count and architecture of GPT-3?", "factual", ["175 billion", "transformer", "autoregressive"], ["gpt3"], False)
add_q("Describe the sliding window attention mechanism used in Mistral 7B.", "factual", ["sliding window", "attention", "efficiency", "cache"], ["mistral"], False)
add_q("What are the key safety and alignment techniques used for Llama 2 Chat?", "factual", ["rlhf", "safety", "alignment", "reward"], ["llama2"], False)
add_q("How does grouped-query attention (GQA) improve inference in Llama 2?", "factual", ["grouped-query", "inference", "speed", "memory"], ["llama2"], False)
add_q("What is the primary evaluation metric used in the educational pedagogy paper?", "factual", ["students", "learning", "assessment"], ["edu"], False)
add_q("How does BERT represent input sequences using WordPiece embeddings?", "factual", ["wordpiece", "embeddings", "token", "cls", "sep"], ["bert"], False)
add_q("What is the 'Colossal Clean Crawled Corpus' (C4) dataset used in T5?", "factual", ["c4", "corpus", "web", "clean"], ["t5"], False)
add_q("How does the REALM model update its knowledge retriever during fine-tuning?", "factual", ["fine-tuning", "retriever", "update", "gradients"], ["realm"], False)
add_q("What are the main components of the RAG generator model?", "factual", ["generator", "bart", "seq2seq"], ["rag"], False)
add_q("How are positive and negative passages selected for training DPR?", "factual", ["positive", "negative", "bm25", "gold"], ["dpr"], False)
add_q("What is the impact of few-shot learning demonstrated by GPT-3?", "factual", ["few-shot", "in-context", "zero-shot", "learning"], ["gpt3"], False)
add_q("Describe the self-attention formula used in the original Transformer.", "factual", ["query", "key", "value", "softmax", "sqrt"], ["transformer"], False)
add_q("What is the vocabulary size used in the Mistral 7B tokenizer?", "factual", ["vocabulary", "token", "bpe"], ["mistral"], False)
add_q("What are the safety reward models used in the Llama 2 RLHF pipeline?", "factual", ["safety", "reward model", "helpful", "rlhf"], ["llama2"], False)

# 2. Architecture / Methodology Comparisons (22)
add_q("Compare the architectures of the Transformer model and BERT.", "architecture_comparison", ["encoder", "decoder", "bidirectional", "attention"], ["transformer", "bert"], True)
add_q("How do the pre-training objectives of BERT and T5 differ?", "methodology_comparison", ["masked", "text-to-text", "span", "objective"], ["bert", "t5"], True)
add_q("Compare the retrieval mechanisms in REALM, RAG, and DPR.", "methodology_comparison", ["retrieval", "dense", "generator", "pre-training"], ["realm", "rag", "dpr"], True)
add_q("How does the architecture of GPT-3 differ from BERT?", "architecture_comparison", ["autoregressive", "bidirectional", "decoder", "encoder"], ["gpt3", "bert"], True)
add_q("Compare the attention mechanisms in Mistral and Llama 2.", "architecture_comparison", ["sliding window", "grouped-query", "attention", "efficiency"], ["mistral", "llama2"], True)
add_q("What are the differences between standard multi-head attention and grouped-query attention?", "architecture_comparison", ["multi-head", "grouped-query", "keys", "values"], ["transformer", "llama2"], True)
add_q("How do RAG and REALM incorporate external knowledge differently?", "methodology_comparison", ["fine-tuning", "pre-training", "generator", "knowledge"], ["rag", "realm"], True)
add_q("Compare the dual-encoder architecture of DPR with cross-encoder models.", "architecture_comparison", ["dual-encoder", "cross-encoder", "representation", "scoring"], ["dpr"], True) # Mainly in DPR paper
add_q("How do the tokenization strategies differ between GPT-3 and Llama 2?", "methodology_comparison", ["bpe", "tokenizer", "byte", "vocabulary"], ["gpt3", "llama2"], True)
add_q("Compare the RLHF pipelines used in Llama 2 and other instruction-tuned models.", "methodology_comparison", ["rlhf", "ppo", "reward", "alignment"], ["llama2", "mistral"], True)
add_q("What are the differences in positional encoding between the original Transformer and later models like Llama 2?", "architecture_comparison", ["sinusoidal", "rotary", "rope", "positional"], ["transformer", "llama2"], True)
add_q("How does T5's approach to multitask learning compare to GPT-3's in-context learning?", "methodology_comparison", ["multitask", "text-to-text", "in-context", "few-shot"], ["t5", "gpt3"], True)
add_q("Compare the fine-tuning strategies of BERT and RAG.", "methodology_comparison", ["fine-tuning", "downstream", "generator", "retriever"], ["bert", "rag"], True)
add_q("How does Mistral's architecture optimize for faster inference compared to standard Transformers?", "architecture_comparison", ["sliding window", "inference", "memory", "cache"], ["mistral", "transformer"], True)
add_q("What are the differences in how REALM and DPR define their document indexes?", "methodology_comparison", ["index", "wikipedia", "dense", "passages"], ["realm", "dpr"], True)
add_q("Compare the scale of parameters between BERT-Large, T5-11B, and GPT-3.", "architecture_comparison", ["340m", "11b", "175b", "scale"], ["bert", "t5", "gpt3"], True)
add_q("How do the safety interventions in Llama 2 differ from those described in earlier language model papers?", "methodology_comparison", ["safety", "red teaming", "rlhf", "bias"], ["llama2", "gpt3", "bert"], True)
add_q("Compare the use of activation functions (e.g., ReLU vs SwiGLU) across the Transformer, T5, and Llama 2.", "architecture_comparison", ["relu", "swiglu", "activation", "gated"], ["transformer", "t5", "llama2"], True)
add_q("What are the differences in how RAG-Sequence and RAG-Token generate text?", "methodology_comparison", ["sequence", "token", "marginalize", "beam search"], ["rag"], True)
add_q("Compare the data filtering techniques used for pre-training corpora in T5 (C4) and Llama 2.", "methodology_comparison", ["filtering", "clean", "quality", "c4"], ["t5", "llama2"], True)
add_q("How do the sequence length limits compare across BERT, GPT-3, and Mistral?", "architecture_comparison", ["512", "2048", "8192", "context length"], ["bert", "gpt3", "mistral"], True)
add_q("Compare the embedding sizes used in the original Transformer versus GPT-3.", "architecture_comparison", ["512", "12288", "embedding", "dimension"], ["transformer", "gpt3"], True)


# 3. Dataset / Training Comparisons (18)
add_q("What datasets were used to train BERT, T5, and GPT-3?", "dataset_comparison", ["wikipedia", "bookcorpus", "c4", "common crawl"], ["bert", "t5", "gpt3"], True)
add_q("Compare the pre-training datasets of Llama 2 and Mistral.", "dataset_comparison", ["tokens", "trillion", "public", "data"], ["llama2", "mistral"], True)
add_q("What datasets are commonly used for Open-Domain QA evaluation in REALM, RAG, and DPR?", "dataset_comparison", ["natural questions", "webquestions", "triviaqa", "curatedtrec"], ["realm", "rag", "dpr"], True)
add_q("How many tokens were used to train GPT-3 compared to Llama 2?", "dataset_comparison", ["300 billion", "2 trillion", "tokens", "scale"], ["gpt3", "llama2"], True)
add_q("What machine translation datasets were used to evaluate the original Transformer and T5?", "dataset_comparison", ["wmt", "english-german", "english-french", "translation"], ["transformer", "t5"], True)
add_q("Compare the knowledge sources (e.g., Wikipedia dumps) used in REALM and RAG.", "dataset_comparison", ["wikipedia", "dump", "passages", "knowledge"], ["realm", "rag"], True)
add_q("What fine-tuning datasets were used for alignment in Llama 2?", "dataset_comparison", ["hh-rlhf", "helpfulness", "safety", "prompts"], ["llama2"], True)
add_q("How do the training durations and hardware setups compare for GPT-3 and Llama 2?", "training_comparison", ["v100", "a100", "gpu", "compute", "days"], ["gpt3", "llama2"], True)
add_q("What datasets were used to evaluate code generation capabilities in Llama 2 and Mistral?", "dataset_comparison", ["humaneval", "mbpp", "code", "benchmark"], ["llama2", "mistral"], True)
add_q("Compare the use of GLUE and SuperGLUE benchmarks in evaluating BERT and T5.", "dataset_comparison", ["glue", "superglue", "benchmark", "tasks"], ["bert", "t5"], True)
add_q("What conversational datasets were used to train the Chat versions of Llama 2 and Mistral?", "dataset_comparison", ["chat", "instruction", "dialogue", "turns"], ["llama2", "mistral"], True)
add_q("How was the C4 dataset created for T5?", "dataset_comparison", ["common crawl", "filtering", "heuristics", "deduplication"], ["t5"], False)
add_q("What negative sampling strategies were used to create training data for DPR?", "dataset_comparison", ["bm25", "hard negatives", "gold", "passages"], ["dpr"], False)
add_q("Compare the reasoning benchmarks (e.g., GSM8K) used to evaluate GPT-3, Llama 2, and Mistral.", "dataset_comparison", ["gsm8k", "math", "reasoning", "benchmark"], ["gpt3", "llama2", "mistral"], True)
add_q("What reading comprehension datasets (e.g., SQuAD) were used across BERT and T5?", "dataset_comparison", ["squad", "reading comprehension", "qa", "f1"], ["bert", "t5"], True)
add_q("How did the authors of Llama 2 filter out sensitive or PII data from their training corpus?", "dataset_comparison", ["pii", "sensitive", "filtering", "scrubbing"], ["llama2"], False)
add_q("What datasets were used to evaluate the educational pedagogy paper?", "dataset_comparison", ["students", "survey", "classroom", "data"], ["edu"], False)
add_q("Compare the sizes of the pre-training corpora for BERT (Books+Wiki) vs T5 (C4).", "dataset_comparison", ["16gb", "750gb", "size", "corpus"], ["bert", "t5"], True)


# 4. Results / Evaluation Metric Comparisons (16)
add_q("How do the reported BLEU scores of the Transformer compare to earlier models?", "results_comparison", ["bleu", "state-of-the-art", "translation", "score"], ["transformer"], False)
add_q("Compare the GLUE score improvements achieved by BERT and T5.", "results_comparison", ["glue", "score", "state-of-the-art", "average"], ["bert", "t5"], True)
add_q("What were the Exact Match (EM) scores for Open-Domain QA models REALM, RAG, and DPR on Natural Questions?", "results_comparison", ["exact match", "em", "natural questions", "score"], ["realm", "rag", "dpr"], True)
add_q("How did GPT-3 perform on zero-shot vs few-shot tasks?", "results_comparison", ["zero-shot", "few-shot", "accuracy", "prompt"], ["gpt3"], False)
add_q("Compare the performance of Mistral 7B and Llama 2 13B on standard benchmarks.", "results_comparison", ["mistral 7b", "llama 2 13b", "outperforms", "benchmark"], ["mistral", "llama2"], True)
add_q("What was the impact of scaling parameters from BERT-Base to BERT-Large?", "results_comparison", ["accuracy", "improvement", "large", "base"], ["bert"], False)
add_q("How did T5-11B perform on the SuperGLUE benchmark compared to human baselines?", "results_comparison", ["superglue", "human", "baseline", "outperform"], ["t5"], False)
add_q("Compare the generation quality (e.g., factuality, specificity) of RAG vs standard seq2seq models.", "results_comparison", ["factuality", "hallucination", "specific", "generation"], ["rag"], True)
add_q("What were the top-20 and top-100 retrieval accuracies reported for DPR compared to BM25?", "results_comparison", ["top-20", "top-100", "accuracy", "bm25", "outperform"], ["dpr"], True)
add_q("How did Llama 2 Chat perform in human evaluations against open-source and closed-source models?", "results_comparison", ["human evaluation", "win rate", "elo", "chatgpt"], ["llama2"], False)
add_q("What results were reported for GPT-3 on the TriviaQA dataset?", "results_comparison", ["triviaqa", "accuracy", "few-shot", "zero-shot"], ["gpt3"], False)
add_q("Compare the inference latency or throughput improvements reported for Mistral due to its architectural changes.", "results_comparison", ["throughput", "latency", "speed", "inference"], ["mistral"], True)
add_q("What safety evaluation metrics (e.g., violation rates) were reported for Llama 2?", "results_comparison", ["violation", "safety", "toxicity", "bias"], ["llama2"], False)
add_q("How did the pedagogical interventions affect student outcomes in the educational paper?", "results_comparison", ["improvement", "significant", "p-value", "score"], ["edu"], False)
add_q("Compare the perplexity scores achieved by different language models as parameter counts increased.", "results_comparison", ["perplexity", "decrease", "scale", "loss"], ["gpt3", "llama2"], True)
add_q("What was the impact of using masked language modeling versus standard left-to-right modeling in BERT?", "results_comparison", ["bidirectional", "improvement", "ablation", "context"], ["bert"], False)


# 5. Multi-paper Synthesis (15)
add_q("Summarize the evolution of language model architectures from the original Transformer to Llama 2.", "synthesis", ["encoder", "decoder-only", "scale", "attention"], ["transformer", "bert", "gpt3", "llama2"], True)
add_q("How has the scale of training data and parameters changed from BERT (2018) to Llama 2 (2023)?", "synthesis", ["millions", "billions", "trillions", "scale", "compute"], ["bert", "t5", "gpt3", "llama2"], True)
add_q("Synthesize the different approaches to Open-Domain QA proposed in REALM, RAG, and DPR.", "synthesis", ["pre-training", "generation", "retrieval", "dense"], ["realm", "rag", "dpr"], True)
add_q("What are the key themes in improving attention mechanism efficiency across the Transformer, Mistral, and Llama 2?", "synthesis", ["efficiency", "sliding window", "grouped-query", "memory"], ["transformer", "mistral", "llama2"], True)
add_q("How did the NLP community shift from fine-tuning (BERT/T5) to in-context learning (GPT-3)?", "synthesis", ["fine-tuning", "in-context", "few-shot", "prompt"], ["bert", "t5", "gpt3"], True)
add_q("Summarize the strategies for aligning large language models with human preferences.", "synthesis", ["rlhf", "alignment", "safety", "reward"], ["gpt3", "llama2", "mistral"], True)
add_q("What role does external knowledge retrieval play in modern language models based on REALM and RAG?", "synthesis", ["hallucination", "knowledge", "update", "retrieval-augmented"], ["realm", "rag"], True)
add_q("How have tokenization methods evolved across these language models?", "synthesis", ["wordpiece", "sentencepiece", "bpe", "byte"], ["bert", "t5", "gpt3", "llama2"], True)
add_q("Synthesize the findings on how model scaling affects zero-shot and few-shot capabilities.", "synthesis", ["scale", "emergent", "few-shot", "performance"], ["gpt3", "llama2"], True)
add_q("What are the common hardware challenges and compute requirements discussed across large model papers?", "synthesis", ["gpu", "tpu", "memory", "parallelism", "cost"], ["gpt3", "llama2", "t5"], True)
add_q("How do these papers approach the problem of context length limitations in Transformers?", "synthesis", ["context length", "extrapolation", "memory", "attention"], ["transformer", "bert", "gpt3", "mistral", "llama2"], True)
add_q("Summarize the shift toward open-source model releases seen in Llama 2 and Mistral compared to GPT-3.", "synthesis", ["open-source", "weights", "community", "closed"], ["gpt3", "llama2", "mistral"], True)
add_q("What common benchmarks are used to track the progress of NLP models across these papers?", "synthesis", ["glue", "superglue", "squad", "mmlu"], ["bert", "t5", "gpt3", "llama2"], True)
add_q("How do generative approaches (T5, GPT-3) differ in solving classification tasks compared to encoder-only approaches (BERT)?", "synthesis", ["text-to-text", "classification", "generation", "prompt"], ["bert", "t5", "gpt3"], True)
add_q("Synthesize the various methods used to reduce toxicity and bias in language models.", "synthesis", ["toxicity", "bias", "filtering", "alignment", "rlhf"], ["gpt3", "llama2"], True)


# 6. Limitations / Research-gap Questions (12)
add_q("What limitations of standard dense retrieval are highlighted in the RAG and REALM papers?", "limitations_comparison", ["static", "index", "update", "knowledge"], ["rag", "realm"], True)
add_q("What does the GPT-3 paper state about its limitations regarding common sense and logical reasoning?", "limitations_comparison", ["common sense", "logic", "physics", "reasoning"], ["gpt3"], False)
add_q("Describe the limitations of the original Transformer architecture regarding long sequences.", "limitations_comparison", ["quadratic", "complexity", "memory", "long"], ["transformer"], False)
add_q("What limitations of BERT did T5 attempt to address?", "limitations_comparison", ["generative", "tasks", "unified", "text-to-text"], ["bert", "t5"], True)
add_q("What research gaps in model safety and alignment are discussed in the Llama 2 paper?", "limitations_comparison", ["safety", "jailbreak", "adversarial", "bias"], ["llama2"], False)
add_q("How do the authors of DPR characterize the limitations of BM25?", "limitations_comparison", ["lexical", "synonym", "vocabulary", "sparse"], ["dpr"], False)
add_q("What does the GPT-3 paper say about the environmental impact and compute cost of large models?", "limitations_comparison", ["carbon", "energy", "compute", "cost"], ["gpt3"], False)
add_q("What limitations are associated with the RAG generator's inability to see the full context at once?", "limitations_comparison", ["marginalize", "fusion", "context", "sequence"], ["rag"], False)
add_q("What research gaps remain in open-source models compared to closed-source models according to Llama 2 and Mistral?", "limitations_comparison", ["gap", "closed-source", "proprietary", "performance"], ["llama2", "mistral"], True)
add_q("What limitations of the educational study are mentioned in the pedagogy paper?", "limitations_comparison", ["sample size", "generalizability", "control"], ["edu"], False)
add_q("What are the limitations of fine-tuning discussed in the GPT-3 paper when introducing in-context learning?", "limitations_comparison", ["overfitting", "spurious", "dataset", "task-specific"], ["gpt3"], False)
add_q("How do Mistral and Llama 2 address the memory bandwidth limitations during inference?", "limitations_comparison", ["memory bandwidth", "grouped-query", "kv cache", "sliding window"], ["mistral", "llama2"], True)


# 7. Contradiction / Disagreement (5)
add_q("How does the T5 paper's view on the necessity of unsupervised pre-training contrast with other approaches?", "contradiction", ["necessity", "unsupervised", "supervised", "transfer"], ["t5", "bert"], True)
add_q("Does DPR suggest that BM25 should be completely replaced, or do they still find it useful?", "contradiction", ["hybrid", "combine", "outperform", "lexical"], ["dpr"], False)
add_q("How does GPT-3's approach to task-specific fine-tuning contradict the paradigm established by BERT?", "contradiction", ["fine-tuning", "few-shot", "gradient", "in-context"], ["gpt3", "bert"], True)
add_q("Do Llama 2 and Mistral agree on the best attention mechanism for efficient inference?", "contradiction", ["grouped-query", "sliding window", "attention", "efficiency"], ["llama2", "mistral"], True)
add_q("How do RAG and REALM differ on whether the retriever should be updated during downstream fine-tuning?", "contradiction", ["update", "frozen", "retriever", "gradients"], ["rag", "realm"], True)


# 8. Cross-paper Evolution / Historical (5)
add_q("Trace the evolution of the self-attention mechanism from the 2017 Transformer to 2023's Mistral.", "evolution", ["multi-head", "sliding window", "efficiency", "context"], ["transformer", "mistral"], True)
add_q("How has the scale of 'large' language models evolved from 2018 (BERT) to 2020 (GPT-3)?", "evolution", ["340 million", "175 billion", "scale", "magnitude"], ["bert", "gpt3"], True)
add_q("Describe the evolution of retrieval-augmented generation from REALM to RAG.", "evolution", ["pre-training", "generation", "seq2seq", "bart"], ["realm", "rag"], True)
add_q("How have evaluation benchmarks evolved from GLUE (used in BERT) to MMLU (used in Llama 2)?", "evolution", ["glue", "mmlu", "saturation", "difficulty"], ["bert", "llama2"], True)
add_q("Trace the history of Open-Source LLMs from T5 to Llama 2.", "evolution", ["open-source", "weights", "t5", "llama 2", "commercial"], ["t5", "llama2"], True)

# Total Questions: 20 + 22 + 18 + 16 + 15 + 12 + 5 + 5 = 113 questions.

out_file = Path("../data/eval_set.json")
out_file.write_text(json.dumps(questions, indent=2))
print(f"Generated {len(questions)} evaluation questions and saved to {out_file}")
