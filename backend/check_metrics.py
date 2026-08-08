import json
import statistics

data = json.load(open('../data/eval_results_comprehensive.json'))
print('Total questions:', len(data['per_question_results']))

# Average over all 113 questions for the agentic graph baseline
ag_cit_validity = []
ag_cit_coverage = []
ag_cit_retrieval = []
ag_sup = []
ag_wsup = []
ag_unsup = []
ag_unavail = []

for q in data['per_question_results']:
    metrics = q['baselines']['agentic_graph']
    cit = metrics['citation_grounding']
    
    if cit['citation_validity'] != "NOT_AVAILABLE": ag_cit_validity.append(cit['citation_validity'])
    if cit['citation_coverage'] != "NOT_AVAILABLE": ag_cit_coverage.append(cit['citation_coverage'])
    if cit['citation_retrieval_accuracy'] != "NOT_AVAILABLE": ag_cit_retrieval.append(cit['citation_retrieval_accuracy'])
    if cit['support_rate'] != "NOT_AVAILABLE": ag_sup.append(cit['support_rate'])
    if cit['weak_support_rate'] != "NOT_AVAILABLE": ag_wsup.append(cit['weak_support_rate'])
    if cit['unsupported_claim_rate'] != "NOT_AVAILABLE": ag_unsup.append(cit['unsupported_claim_rate'])
    if cit['verification_unavailable_rate'] != "NOT_AVAILABLE": ag_unavail.append(cit['verification_unavailable_rate'])

def safe_mean(lst):
    return statistics.mean(lst) if lst else 0.0

print("Averages:")
print(f"Citation Coverage: {safe_mean(ag_cit_coverage):.4f}")
print(f"Citation Validity: {safe_mean(ag_cit_validity):.4f}")
print(f"Citation Retrieval Accuracy: {safe_mean(ag_cit_retrieval):.4f}")
print(f"Support Rate: {safe_mean(ag_sup):.4f}")
print(f"Weak Support Rate: {safe_mean(ag_wsup):.4f}")
print(f"Unsupported Rate: {safe_mean(ag_unsup):.4f}")
print(f"Verification Unavailable Rate: {safe_mean(ag_unavail):.4f}")
