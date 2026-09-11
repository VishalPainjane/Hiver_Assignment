import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
from src.domain.models import SupportCase
from src.classification.setfit_classifier import SetFitIntentClassifier
from src.classification.need_analyzer import NeedAnalyzer

classifier = SetFitIntentClassifier()
analyzer = NeedAnalyzer(classifier=classifier.classify)

# Test 1: Held-out real customer tweets from test split
test_split = json.load(open("data/processed/test_split.json", encoding="utf-8"))

# Pick 1 representative test case per class from test split
by_intent = {}
for item in test_split:
    intent = item["intent"]
    if intent not in by_intent:
        by_intent[intent] = item["text"]

print("\n" + "="*80)
print("PART 1: REAL HELD-OUT TEST CASES (NeedAnalyzer + SetFit on RTX 4050)")
print("="*80)
real_all_pass = True
for intent, text in by_intent.items():
    case = SupportCase(case_id="held_out", clean_text=text)
    res = analyzer.analyze(case)
    matched = res.primary_intent == intent
    real_all_pass = real_all_pass and matched
    status = "PASS" if matched else "FAIL"
    print(f"[{status}] Expected: {intent:<16} | Predicted: {res.primary_intent:<16} (Confidence: {res.confidence*100:.1f}%)")
    clean_display = text.replace("\n", " ")[:85]
    print(f"       Tweet: \"{clean_display}...\"\n")

print(f"Part 1 Result: {'ALL 6 INTENTS PASSED' if real_all_pass else 'SOME FAILED'}")

# Test 2: Natural colloquial customer inputs
natural_cases = [
    ("My iPhone battery dies within 2 hours of charging and gets super hot to touch", "BATTERY_DRAIN"),
    ("My iPhone shut off and keeps restarting every minute in a boot loop", "DEVICE_BOOT"),
    ("I cannot sign in to my iCloud account because verification passcode is locked", "ACCOUNT_ACCESS"),
    ("I see an unauthorized subscription charge of $9.99 on my credit card from iTunes store", "BILLING_DISPUTE"),
    ("My phone is stuck trying to verify iOS 17 software update and gives an install error", "UPDATE_FAILURE"),
    ("What are the store hours for the Apple Store in Regent Street today?", "UNKNOWN"),
]

print("\n" + "="*80)
print("PART 2: COLLOQUIAL REAL-WORLD QUERIES (NeedAnalyzer + SetFit on RTX 4050)")
print("="*80)
nat_all_pass = True
for text, expected in natural_cases:
    case = SupportCase(case_id="natural", clean_text=text)
    res = analyzer.analyze(case)
    matched = res.primary_intent == expected
    nat_all_pass = nat_all_pass and matched
    status = "PASS" if matched else "FAIL"
    print(f"[{status}] Expected: {expected:<16} | Predicted: {res.primary_intent:<16} (Confidence: {res.confidence*100:.1f}%)")
    print(f"       Query: \"{text}\"\n")

print(f"Part 2 Result: {'ALL 6 INTENTS PASSED' if nat_all_pass else 'SOME FAILED'}")
print("="*80 + "\n")
