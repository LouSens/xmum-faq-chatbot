"""
stress_test.py — Comprehensive ML model stress test for xmum-faq-chatbot
Probes: accuracy, confidence, edge cases, synonyms, typos, ambiguity, gaps
"""

import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlp.pipeline import load_pipeline, get_response

load_pipeline()

# ──────────────────────────────────────────────────────────────────────────────
# Test battery — (query, expected_intent, label)
# ──────────────────────────────────────────────────────────────────────────────
TESTS = [

    # ── STRONG POSITIVES (should be easy) ────────────────────────────────────
    ("How do I apply to XMUM?",                          "admissions_application",   "STRONG"),
    ("What programs does XMUM offer?",                   "programs_faculties",        "STRONG"),
    ("How much is the tuition fee?",                     "tuition_fees",             "STRONG"),
    ("Are there scholarships at XMUM?",                  "scholarships_financial_aid","STRONG"),
    ("Is there accommodation on campus?",                "campus_life_facilities",   "STRONG"),
    ("What are the entry requirements?",                 "entry_requirements",       "STRONG"),
    ("Where is XMUM located?",                          "location_transportation",  "STRONG"),
    ("How do I contact XMUM?",                          "contact_enquiries",        "STRONG"),
    ("Tell me about XMUM",                              "university_background",    "STRONG"),

    # ── SYNONYM / PARAPHRASE (should handle) ─────────────────────────────────
    ("What are the fees?",                              "tuition_fees",             "SYNONYM"),
    ("What does it cost to study here?",                "tuition_fees",             "SYNONYM"),
    ("How do I get in?",                                "entry_requirements",       "SYNONYM"),
    ("What do I need to enroll?",                       "admissions_application",   "SYNONYM"),
    ("Is there a dorm?",                                "campus_life_facilities",   "SYNONYM"),
    ("What hostel is available?",                       "campus_life_facilities",   "SYNONYM"),
    ("How do I reach XMUM?",                            "location_transportation",  "SYNONYM"),
    ("Do you have financial support?",                  "scholarships_financial_aid","SYNONYM"),
    ("I need help paying for school",                   "scholarships_financial_aid","SYNONYM"),
    ("Is there a medical center?",                      "campus_life_facilities",   "SYNONYM"),

    # ── PROGRAM-SPECIFIC (critical — many sub-topics lumped into 1 intent) ───
    ("Does XMUM have an AI degree?",                    "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("What is the artificial intelligence major?",      "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("Is there a machine learning course?",             "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("Tell me about the data science program",          "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("Is software engineering available?",              "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("Does XMUM offer law?",                            "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("Is there a medicine degree at XMUM?",             "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("What engineering courses are there?",             "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("Is there a business degree?",                     "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("Does XMUM have a psychology program?",            "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("Is there a nursing or pharmacy course?",          "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("Does XMUM have MBA?",                             "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("What is the accounting program like?",            "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("Is architecture offered at XMUM?",                "programs_faculties",       "PROGRAM_SPECIFIC"),
    ("Does XMUM have a film or media program?",         "programs_faculties",       "PROGRAM_SPECIFIC"),

    # ── ENTRY REQUIREMENTS SPECIFICS ─────────────────────────────────────────
    ("What CGPA do I need?",                            "entry_requirements",       "ENTRY_SPECIFIC"),
    ("Can I apply with SPM results?",                   "entry_requirements",       "ENTRY_SPECIFIC"),
    ("What are the A-Level requirements?",              "entry_requirements",       "ENTRY_SPECIFIC"),
    ("Do I need IELTS?",                                "entry_requirements",       "ENTRY_SPECIFIC"),
    ("What English language score is required?",        "entry_requirements",       "ENTRY_SPECIFIC"),
    ("Can international students apply?",               "entry_requirements",       "ENTRY_SPECIFIC"),
    ("What is the minimum GPA for engineering?",        "entry_requirements",       "ENTRY_SPECIFIC"),

    # ── FEE SPECIFICS ────────────────────────────────────────────────────────
    ("How much for a 4-year engineering degree?",       "tuition_fees",             "FEE_SPECIFIC"),
    ("What is the yearly tuition?",                     "tuition_fees",             "FEE_SPECIFIC"),
    ("How much does business cost per semester?",       "tuition_fees",             "FEE_SPECIFIC"),
    ("Are there additional costs besides tuition?",     "tuition_fees",             "FEE_SPECIFIC"),
    ("What is the registration fee?",                   "tuition_fees",             "FEE_SPECIFIC"),

    # ── TYPOS & INFORMAL ─────────────────────────────────────────────────────
    ("how mch is the tution fee",                       "tuition_fees",             "TYPO"),
    ("wat programmes does xmum have",                   "programs_faculties",       "TYPO"),
    ("scolarship at xmum?",                             "scholarships_financial_aid","TYPO"),
    ("how 2 apply xmum",                                "admissions_application",   "TYPO"),
    ("wher is xmum campus",                             "location_transportation",  "TYPO"),
    ("xmum entery requirement",                         "entry_requirements",       "TYPO"),
    ("dormitory availble?",                             "campus_life_facilities",   "TYPO"),

    # ── VERY SHORT / KEYWORD-ONLY ────────────────────────────────────────────
    ("fees",                                            "tuition_fees",             "SHORT"),
    ("scholarship",                                     "scholarships_financial_aid","SHORT"),
    ("apply",                                           "admissions_application",   "SHORT"),
    ("programs",                                        "programs_faculties",       "SHORT"),
    ("location",                                        "location_transportation",  "SHORT"),
    ("hostel",                                          "campus_life_facilities",   "SHORT"),
    ("XMUM",                                            "university_background",    "SHORT"),
    ("AI",                                              "programs_faculties",       "SHORT"),
    ("CS",                                              "programs_faculties",       "SHORT"),

    # ── AMBIGUOUS (hard cases) ────────────────────────────────────────────────
    ("What can I study?",                               "programs_faculties",       "AMBIGUOUS"),
    ("Tell me more",                                    "out_of_scope",             "AMBIGUOUS"),
    ("Is XMUM good?",                                   "university_background",    "AMBIGUOUS"),
    ("How long is the degree?",                         "programs_faculties",       "AMBIGUOUS"),
    ("When does semester start?",                       "admissions_application",   "AMBIGUOUS"),
    ("What is the deadline?",                           "admissions_application",   "AMBIGUOUS"),
    ("Do you accept transfers?",                        "admissions_application",   "AMBIGUOUS"),
    ("What is the GPA requirement for scholarship?",    "scholarships_financial_aid","AMBIGUOUS"),

    # ── OUT OF SCOPE (should trigger fallback) ────────────────────────────────
    ("What is the weather like today?",                 "out_of_scope",             "OOS"),
    ("Who is the president of Malaysia?",               "out_of_scope",             "OOS"),
    ("Write me a Python program",                       "out_of_scope",             "OOS"),
    ("What is 2 + 2?",                                  "out_of_scope",             "OOS"),
    ("Hi",                                              "out_of_scope",             "OOS"),
    ("Thank you",                                       "out_of_scope",             "OOS"),
    ("xyzabc123",                                       "out_of_scope",             "OOS"),
    ("I want to buy a car",                             "out_of_scope",             "OOS"),

    # ── OUT OF SCOPE EDGE CASES (Tricky OOS) ──────────────────────────────────
    ("Where can I buy a car near XMUM?",                "out_of_scope",             "OOS_EDGE"),
    ("How much is the fee for an online Coursera course?", "out_of_scope",          "OOS_EDGE"),
    ("Can I use my scholarship to buy a house?",        "out_of_scope",             "OOS_EDGE"),
    ("wat is the wether in sepang?",                    "out_of_scope",             "OOS_EDGE"),
    ("I want to apply for a job at XMUM",               "out_of_scope",             "OOS_EDGE"),
    ("What programs does Harvard offer?",               "out_of_scope",             "OOS_EDGE"),
    ("Do I need IELTS to travel to UK?",                "out_of_scope",             "OOS_EDGE"),
    ("how much to fix my laptop?",                      "out_of_scope",             "OOS_EDGE"),
    ("is the food at xmum canteen good?",               "out_of_scope",             "OOS_EDGE"),
    ("what happens if i fail my exam?",                 "out_of_scope",             "OOS_EDGE"),
    ("show me the map of malaysia",                     "out_of_scope",             "OOS_EDGE"),
    ("admisison 2 hospitl",                             "out_of_scope",             "OOS_EDGE"),
    ("how 2 get scholarship for primary school",        "out_of_scope",             "OOS_EDGE"),
    ("fees for gym membership",                         "out_of_scope",             "OOS_EDGE"),

    # ── CONFUSABLE INTENT PAIRS ───────────────────────────────────────────────
    ("How do I pay the fees?",                          "tuition_fees",             "CONFUSABLE"),
    ("How do I pay for accommodation?",                 "campus_life_facilities",   "CONFUSABLE"),
    ("What documents do I need to apply?",              "admissions_application",   "CONFUSABLE"),
    ("What documents do I need for scholarship?",       "scholarships_financial_aid","CONFUSABLE"),
    ("Is XMUM near KL?",                               "location_transportation",  "CONFUSABLE"),
    ("What is the campus like?",                        "campus_life_facilities",   "CONFUSABLE"),
    ("Does XMUM have a clinic?",                        "campus_life_facilities",   "CONFUSABLE"),
    ("Is XMUM a research university?",                  "university_background",    "CONFUSABLE"),

    # ── MULTILINGUAL / MALAY MIXED ────────────────────────────────────────────
    ("Berapa yuran pengajian di XMUM?",                "tuition_fees",             "MALAY"),
    ("Macam mana nak apply XMUM?",                     "admissions_application",   "MALAY"),
    ("Ada biasiswa tak?",                               "scholarships_financial_aid","MALAY"),
    ("Apa program ada kat XMUM?",                       "programs_faculties",       "MALAY"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Run tests
# ──────────────────────────────────────────────────────────────────────────────
results = []
categories = {}

print(f"\n{'='*100}")
print(f"{'XMUM FAQ CHATBOT — ML STRESS TEST':^100}")
print(f"{'='*100}")
print(f"{'Query':<52} {'Expected':<26} {'Got':<26} {'Conf':>6}  {'Match':>6}  {'Cat'}")
print(f"{'-'*100}")

correct = wrong = low_conf = fallback_wrong = 0

for query, expected_intent, category in TESTS:
    result = get_response(query)
    got_intent    = result.get("intent", "?")
    confidence    = result.get("confidence", 0.0)
    matched_q     = result.get("matched_question") or ""
    answer_snippet= (result.get("answer") or "")[:60]

    is_correct = (got_intent == expected_intent)
    is_low     = confidence < 0.40
    is_fallback= got_intent == "out_of_scope"

    if is_correct:
        correct += 1
        status = "✓"
    else:
        wrong += 1
        status = "✗"
        if expected_intent != "out_of_scope" and is_fallback:
            fallback_wrong += 1

    if is_low and expected_intent != "out_of_scope":
        low_conf += 1

    flag = ""
    if not is_correct: flag += "[WRONG]"
    if is_low and expected_intent != "out_of_scope": flag += "[LOW]"

    q_short = (query[:50] + "…") if len(query) > 51 else query
    print(f"{q_short:<52} {expected_intent:<26} {got_intent:<26} {confidence:>6.3f}  {status:>6}  {category}  {flag}")

    results.append({
        "query": query, "expected": expected_intent, "got": got_intent,
        "confidence": confidence, "matched_q": matched_q,
        "answer": answer_snippet, "correct": is_correct,
        "low_conf": is_low, "category": category
    })
    categories.setdefault(category, {"correct": 0, "wrong": 0, "total": 0, "low": 0})
    categories[category]["total"] += 1
    categories[category]["correct" if is_correct else "wrong"] += 1
    if is_low and expected_intent != "out_of_scope":
        categories[category]["low"] += 1

# ──────────────────────────────────────────────────────────────────────────────
# Category summary
# ──────────────────────────────────────────────────────────────────────────────
total = len(TESTS)
print(f"\n{'='*100}")
print(f"{'RESULTS BY CATEGORY':^100}")
print(f"{'='*100}")
print(f"{'Category':<22} {'Total':>6} {'Correct':>8} {'Wrong':>7} {'Acc%':>7} {'Low Conf':>9}")
print(f"{'-'*60}")
for cat, s in sorted(categories.items()):
    acc = s["correct"] / s["total"] * 100 if s["total"] else 0
    print(f"{cat:<22} {s['total']:>6} {s['correct']:>8} {s['wrong']:>7} {acc:>6.0f}%  {s['low']:>8}")

print(f"\n{'='*100}")
print(f"OVERALL: {correct}/{total} correct ({correct/total*100:.1f}%)")
print(f"WRONG INTENT: {wrong} | LOW CONFIDENCE (<0.40): {low_conf} | WRONG FALLBACKS: {fallback_wrong}")
print(f"{'='*100}")

# ──────────────────────────────────────────────────────────────────────────────
# Detailed wrong & low-confidence cases
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"{'FAILED / LOW CONFIDENCE CASES — DETAILED':^100}")
print(f"{'='*100}")
for r in results:
    if not r["correct"] or r["low_conf"]:
        flag = "WRONG" if not r["correct"] else "LOW CONF"
        print(f"\n[{flag}] [{r['category']}]")
        print(f"  Query    : {r['query']}")
        print(f"  Expected : {r['expected']}")
        print(f"  Got      : {r['got']}")
        print(f"  Conf     : {r['confidence']:.4f}")
        print(f"  Matched Q: {r['matched_q'][:80]}")
        print(f"  Answer   : {r['answer']}")

# ──────────────────────────────────────────────────────────────────────────────
# Confidence distribution
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"{'CONFIDENCE DISTRIBUTION (correct predictions only)':^100}")
print(f"{'='*100}")
bands = {"0.30-0.40": 0, "0.40-0.55": 0, "0.55-0.70": 0, "0.70-0.85": 0, "0.85-1.00": 0}
for r in results:
    if r["correct"] and r["expected"] != "out_of_scope":
        c = r["confidence"]
        if   c < 0.40: bands["0.30-0.40"] += 1
        elif c < 0.55: bands["0.40-0.55"] += 1
        elif c < 0.70: bands["0.55-0.70"] += 1
        elif c < 0.85: bands["0.70-0.85"] += 1
        else:          bands["0.85-1.00"] += 1
for band, n in bands.items():
    bar = "█" * n
    print(f"  {band}: {bar} ({n})")

print()
