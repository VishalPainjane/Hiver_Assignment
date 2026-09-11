# Data Engineering & Golden Set Documentation

## 1. Directory Structure

- `data/twcs.csv`: Kaggle Customer Support on Twitter dataset (raw conversational graph).
- `data/processed/apple_cases.jsonl`: 131,764 extracted customer support threads directed to `@AppleSupport`.
- `data/processed/apple_playbook_vault.json`: 24,848 cleaned, historical Apple twin precedent pairs with customer query and approved Apple response.
- `data/processed/test_split.json`: Held-out stratified 20% test split ($n = 144$, exactly 24 cases per class) isolated from training.
- `data/expanded_dataset.json`: Full active-learning training pool ($n = 720$, 120 cases per intent).
- `data/golden/annotation_queue.csv`: Candidate sampling queue for human annotation.
- `data/golden/annotation_queue_approved.csv`: Human-adjudicated ground truth records with annotator signoff.
- `data/golden_set.json`: Complete, frozen 200-case Golden Benchmark Set.

---

## 2. Sampling & Labeling Methodology Note

> **Methodology & Provenance:** The 200-case Golden Set (`data/golden_set.json`) was curated from raw customer-to-brand conversational threads in `twcs.csv` (`@AppleSupport`) using an intent-stratified sampling strategy across 6 core operational intents (`ACCOUNT_ACCESS`, `BATTERY_DRAIN`, `BILLING_DISPUTE`, `DEVICE_BOOT`, `UPDATE_FAILURE`, and `UNKNOWN`), explicitly enriched with critical safety/thermal hazards, multi-lingual edge cases, and out-of-scope customer inquiries to realistically reflect live Twitter support volume dynamics. Every conversation underwent a rigorous two-reviewer human adjudication protocol (`data/golden/annotation_queue_approved.csv`) where an expert lead adjudicator verified ground-truth customer intents, audited appropriate deterministic routing decisions (`AUTO_REPLY` vs. `HANDOFF`), mapped acceptable authoritative Apple Knowledge Base evidence IDs (official HT guides), and authored compliant reference answers under Apple's 4 Golden Brand Rules. Crucially, all taxonomy definitions, evidence chunk mappings, and gold labels were permanently frozen prior to benchmarking to guarantee zero train-test leakage and provide an uncompromised, reproducible benchmark across all baseline and production architectures.

---

## 3. Golden Set Distribution (Real Twitter Skew, $n = 200$)

| Intent Category       |  Count  | Percentage  | Operational Nature                                                                |
| :-------------------- | :-----: | :---------: | :-------------------------------------------------------------------------------- |
| **`UNKNOWN`**         |   69    |   34.50%    | Ambiguous, colloquial, out-of-scope, or multi-topic tweets requiring human triage |
| **`BATTERY_DRAIN`**   |   31    |   15.50%    | Rapid drain, overheating, percentage drop, charging failure                       |
| **`BILLING_DISPUTE`** |   29    |   14.50%    | Accidental in-app purchases, unauthorized subscriptions, iTunes refund requests   |
| **`DEVICE_BOOT`**     |   27    |   13.50%    | Black screen, boot loop, frozen on Apple logo, unresponsiveness                   |
| **`UPDATE_FAILURE`**  |   22    |   11.00%    | iOS verification error, installation failure, storage capacity during update      |
| **`ACCOUNT_ACCESS`**  |   22    |   11.00%    | Apple ID lockout, 2FA passcode failure, forgotten iCloud credentials              |
| **Total**             | **200** | **100.00%** | **Frozen Evaluation Benchmark**                                                   |

---

## 4. Annotation Schema & Attributes

Each record in `data/golden/annotation_queue_approved.csv` and `data/golden_set.json` satisfies the following audited schema:

- `case_id`: Unique identifier from the Kaggle Twitter root thread.
- `customer_text`: Cleaned, de-identified customer query text.
- `expected.intent`: Verified ground-truth taxonomical intent label.
- `expected.mode`: Deterministic routing decision (`AUTO_REPLY` vs. `HANDOFF`).
- `expected.reason`: Machine-auditable policy rule reason (e.g. `ALL_POLICIES_PASSED`, `POLICY_REQUIRES_HUMAN`, `HIGH_RISK_FLAG`).
- `acceptable_evidence_ids`: Array of valid Apple Knowledge Base article chunks (e.g., `apple_faq_battery_drain`, `apple_faq_billing_refund`, `apple_faq_connectivity`).
- `reference_answer`: Gold reference response adhering to Apple's 4 brand rules (Diagnostic Probe, Privacy DM Pivot, Zero-Liability Shield, $< 240$ chars).
- `review_status`: Final adjudication state (`APPROVED`).
