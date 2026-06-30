# Chapter 4: Results and Evaluation

## 4.1 Experimental Setup
The natural language processing (NLP) pipeline was evaluated to determine its effectiveness in classifying user intents and retrieving accurate FAQ answers. The core architecture consists of two main stages:
1. **Feature Extraction:** A Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer configured to extract unigrams and bigrams (`ngram_range=(1, 2)`), limited to a maximum of 5,000 features, and applying sublinear term frequency scaling (`sublinear_tf=True`) to dampen the effect of highly frequent terms.
2. **Intent Classification:** A Support Vector Machine (SVM) classifier utilizing a linear kernel (`C=1.0`) with probability estimation enabled.

The dataset consists of 1,464 labeled question variations across 10 distinct intents (plus an `out_of_scope` class). To ensure robust evaluation metrics and prevent overfitting, the system was tested using **5-fold Stratified Cross-Validation**, which maintains the proportional representation of each intent class in every fold.

## 4.2 Intent Classification Results

### 4.2.1 Per-Fold Accuracy
The 5-fold cross-validation demonstrated strong and consistent performance across all data splits. The accuracy for each fold is as follows:

*   **Fold 1:** 88.05%
*   **Fold 2:** 90.44%
*   **Fold 3:** 85.67%
*   **Fold 4:** 89.76%
*   **Fold 5:** 93.15%

**Mean CV Accuracy:** 89.42%  
**Standard Deviation:** 2.49%

The relatively low standard deviation (±2.49%) indicates that the model is stable and generalized well to unseen variations in phrasing, rather than memorizing the training set.

### 4.2.2 Classification Report
The detailed classification report below breaks down performance across all intents using Precision, Recall, and F1-Score:

| Intent Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `admissions_application` | 0.87 | 0.88 | 0.87 | 160 |
| `campus_life_facilities` | 0.79 | 0.89 | 0.84 | 160 |
| `contact_enquiries` | 0.94 | 0.76 | 0.84 | 120 |
| `entry_requirements` | 0.91 | 0.88 | 0.89 | 152 |
| `location_transportation` | 0.92 | 0.93 | 0.92 | 152 |
| `out_of_scope` | 1.00 | 0.97 | 0.98 | 120 |
| `programs_faculties` | 0.84 | 0.90 | 0.87 | 144 |
| `scholarships_financial_aid` | 0.97 | 0.95 | 0.96 | 160 |
| `tuition_fees` | 0.94 | 0.93 | 0.93 | 160 |
| `university_background` | 0.82 | 0.85 | 0.83 | 136 |
| **Accuracy** | | | **0.89** | **1464** |
| **Macro Avg** | 0.90 | 0.89 | 0.89 | 1464 |
| **Weighted Avg** | 0.90 | 0.89 | 0.89 | 1464 |

### 4.2.3 Confusion Matrix Analysis
*Note: A visual representation of the confusion matrix was generated during training and saved to `training/confusion_matrix.png`.*

Analysis of the misclassifications reveals several patterns:
*   **Highly Distinct Intents:** `out_of_scope`, `scholarships_financial_aid`, and `tuition_fees` perform exceptionally well (F1-scores ≥ 0.93). Their vocabulary (e.g., "money", "scholarship", "fee", "cost") rarely overlaps with other topics.
*   **Confusable Boundaries:** `campus_life_facilities` (Precision 0.79) occasionally overlaps with `location_transportation`, as questions about "where is the hostel" contain spatial vocabulary.
*   **The Default Fallback Effect:** `university_background` has a slightly lower precision (0.82). Because general queries often feature the word "XMUM" heavily but lack specific topical keywords, the model sometimes defaults to this background intent.

## 4.4 Confidence Threshold Analysis
The chatbot pipeline utilizes a minimum confidence threshold to determine whether to output an answer or trigger a fallback response ("Sorry, I could not find a relevant answer"). We evaluated the percentage of queries rejected (Fallback %) versus answered (Answered %) across different thresholds during cross-validation:

| Threshold | Fallback % (Rejected) | Answered % (Passed) |
| :--- | :--- | :--- |
| **0.3** | 22.9% | 77.1% |
| **0.4** | 43.0% | 57.0% |
| **0.5** | 66.9% | 33.1% |
| **0.6** | 81.2% | 18.8% |

This analysis highlights a critical sensitivity in the model: raising the threshold to just `0.4` causes 43% of test answers to be rejected as fallback. This indicates that while the SVM is highly accurate at picking the *correct* intent (89.4% accuracy), it frequently does so with mathematically low probability scores (between 0.3 and 0.5). A threshold of `0.3` provides the best balance, though it allows some low-confidence false positives through.

## 4.5 Query Test Analysis
To simulate real-world usage, a custom stress test of 90 diverse queries was constructed covering 11 categories, including typos, synonyms, short queries, ambiguous phrasing, and non-English (Malay) inputs. 

The overall stress test accuracy was **85.6% (77/90 correct)**.

**Performance by Category:**
*   **Canonical/Expected phrasing:** 100% (9/9)
*   **Fee & Entry Specifics:** 91% (11/12)
*   **Program Specifics (e.g., specific majors):** 93% (14/15)
*   **Synonyms & Paraphrasing:** 80% (8/10)
*   **Typos & Misspellings:** 57% (4/7)
*   **Malay Language:** 50% (2/4)

The test confirmed that the TF-IDF+SVM architecture handles canonical academic vocabulary perfectly. However, the drop in performance on typos and Malay language highlights the vulnerability of sparse vector models to out-of-vocabulary (OOV) terms.

### 4.5.1 Full List of Test Queries

**Strong Positives**
* "How do I apply to XMUM?"
* "What programs does XMUM offer?"
* "How much is the tuition fee?"
* "Are there scholarships at XMUM?"
* "Is there accommodation on campus?"
* "What are the entry requirements?"
* "Where is XMUM located?"
* "How do I contact XMUM?"
* "Tell me about XMUM"

**Synonyms & Paraphrasing**
* "What are the fees?"
* "What does it cost to study here?"
* "How do I get in?"
* "What do I need to enroll?"
* "Is there a dorm?"
* "What hostel is available?"
* "How do I reach XMUM?"
* "Do you have financial support?"
* "I need help paying for school"
* "Is there a medical center?"

**Program-Specific**
* "Does XMUM have an AI degree?"
* "What is the artificial intelligence major?"
* "Is there a machine learning course?"
* "Tell me about the data science program"
* "Is software engineering available?"
* "Does XMUM offer law?"
* "Is there a medicine degree at XMUM?"
* "What engineering courses are there?"
* "Is there a business degree?"
* "Does XMUM have a psychology program?"
* "Is there a nursing or pharmacy course?"
* "Does XMUM have MBA?"
* "What is the accounting program like?"
* "Is architecture offered at XMUM?"
* "Does XMUM have a film or media program?"

**Entry Requirements Specifics**
* "What CGPA do I need?"
* "Can I apply with SPM results?"
* "What are the A-Level requirements?"
* "Do I need IELTS?"
* "What English language score is required?"
* "Can international students apply?"
* "What is the minimum GPA for engineering?"

**Fee Specifics**
* "How much for a 4-year engineering degree?"
* "What is the yearly tuition?"
* "How much does business cost per semester?"
* "Are there additional costs besides tuition?"
* "What is the registration fee?"

**Typos & Informal**
* "how mch is the tution fee"
* "wat programmes does xmum have"
* "scolarship at xmum?"
* "how 2 apply xmum"
* "wher is xmum campus"
* "xmum entery requirement"
* "dormitory availble?"

**Very Short / Keyword-Only**
* "fees"
* "scholarship"
* "apply"
* "programs"
* "location"
* "hostel"
* "XMUM"
* "AI"
* "CS"

**Ambiguous**
* "What can I study?"
* "Tell me more"
* "Is XMUM good?"
* "How long is the degree?"
* "When does semester start?"
* "What is the deadline?"
* "Do you accept transfers?"
* "What is the GPA requirement for scholarship?"

**Out of Scope**
* "What is the weather like today?"
* "Who is the president of Malaysia?"
* "Write me a Python program"
* "What is 2 + 2?"
* "Hi"
* "Thank you"
* "xyzabc123"
* "I want to buy a car"

**Out of Scope (Tricky Edge Cases)**
* "Where can I buy a car near XMUM?"
* "How much is the fee for an online Coursera course?"
* "Can I use my scholarship to buy a house?"
* "wat is the wether in sepang?"
* "I want to apply for a job at XMUM"
* "What programs does Harvard offer?"
* "Do I need IELTS to travel to UK?"
* "how much to fix my laptop?"
* "is the food at xmum canteen good?"
* "what happens if i fail my exam?"
* "show me the map of malaysia"
* "admisison 2 hospitl"
* "how 2 get scholarship for primary school"
* "fees for gym membership"

**Confusable Intent Pairs**
* "How do I pay the fees?"
* "How do I pay for accommodation?"
* "What documents do I need to apply?"
* "What documents do I need for scholarship?"
* "Is XMUM near KL?"
* "What is the campus like?"
* "Does XMUM have a clinic?"
* "Is XMUM a research university?"

**Multilingual / Malay Mixed**
* "Berapa yuran pengajian di XMUM?"
* "Macam mana nak apply XMUM?"
* "Ada biasiswa tak?"
* "Apa program ada kat XMUM?"

## 4.6 Discussion

### 4.6.1 Strengths of the System
1.  **High Baseline Accuracy:** An 89.42% cross-validation accuracy proves the feature extraction (TF-IDF) and linear SVM approach is highly effective for this specific FAQ domain.
2.  **Robust Out-of-Scope Rejection:** The system achieved 100% precision and 97% recall on out-of-scope queries. It reliably recognizes when a user asks about unrelated topics (e.g., weather, chit-chat) and prevents the chatbot from hallucinating an answer.
3.  **Strong Topical Boundaries:** Intents with highly specific lexicons (Fees, Scholarships) isolate perfectly, meaning the most critical transactional queries are answered reliably.

### 4.6.2 Limitations and Weaknesses
1.  **Typo Sensitivity (The "XMUM" Collapse):** When users misspell key terms (e.g., "scolarship"), the PorterStemmer fails to process them. Consequently, all topical words are stripped during preprocessing, leaving only "XMUM". The model then defaults entirely to the `university_background` intent regardless of context.
2.  **Vocabulary Gaps & Acronyms:** Terms not present in the training corpus (e.g., "MBA") immediately become unknown tokens. Without semantic word embeddings (like Word2Vec or BERT), the TF-IDF model cannot infer that "MBA" relates to "postgraduate programs."
3.  **Lack of Multilingual Support:** The system relies entirely on English tokenization. Testing with Malay language queries ("Berapa yuran?") resulted in immediate failure or near-zero confidence fallbacks, as the vocabulary space is completely disjointed.
4.  **Low Confidence Calibration:** As shown in Section 4.4, many correctly classified intents are predicted with a confidence score below 0.5. This makes the system fragile to threshold tuning, as attempting to filter out bad predictions will inadvertently block many correct ones.
5.  **Vulnerability to Deceptive Keywords (Bag-of-Words Problem):** Because TF-IDF cannot understand semantic context, any out-of-scope question containing strongly weighted in-scope keywords (e.g., "scholarship", "fee", "apply") will incorrectly bypass the fallback mechanism. For instance, "fees for gym membership" will incorrectly trigger `tuition_fees`, and "What programs does Harvard offer?" will trigger `programs_faculties` because the model only sees the keywords and ignores the unseen entities.
