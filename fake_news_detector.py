# ============================================================
# PROJECT 3: Fake News Detector
# Skills: Python, NLP, TF-IDF, Passive Aggressive Classifier
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── 1. Sample Dataset ──
articles = [
    # REAL news
    ("Government announces new budget with focus on infrastructure development", "REAL"),
    ("Scientists discover new vaccine showing 90 percent efficacy in trials", "REAL"),
    ("Central bank raises interest rates by 25 basis points to control inflation", "REAL"),
    ("United Nations holds emergency session on climate change agreements", "REAL"),
    ("Tech giant reports record quarterly earnings driven by cloud services", "REAL"),
    ("Health ministry launches nationwide vaccination campaign for children", "REAL"),
    ("Prime Minister meets foreign delegates to strengthen trade relations", "REAL"),
    ("New study links regular exercise to reduced risk of heart disease", "REAL"),
    ("Stock markets reach all-time high amid strong economic data", "REAL"),
    ("City council approves new public transportation expansion project", "REAL"),
    ("Scientists confirm rising sea levels due to melting polar ice caps", "REAL"),
    ("University researchers develop new method to detect cancer earlier", "REAL"),
    ("Government passes education reform bill to improve rural schools", "REAL"),
    ("International summit results in agreement on carbon emission targets", "REAL"),
    ("New smartphone model breaks sales records in first week of launch", "REAL"),
    # FAKE news
    ("SHOCKING: Government secretly putting microchips in water supply!", "FAKE"),
    ("Doctors HATE him! Man cures diabetes in 3 days with this one trick.", "FAKE"),
    ("BREAKING: Aliens land in Nevada, government covering up evidence!", "FAKE"),
    ("Bill Gates admits vaccines contain tracking devices, insider reveals!", "FAKE"),
    ("Secret society controls world economy, leaked documents show!", "FAKE"),
    ("5G towers proven to cause cancer, experts silenced by big tech!", "FAKE"),
    ("Moon landing was FAKED! New footage proves NASA conspiracy!", "FAKE"),
    ("Scientists HIDE truth: Earth is actually flat, insider admits!", "FAKE"),
    ("COVID-19 cure found using household bleach, go viral now!", "FAKE"),
    ("URGENT SHARE: Eating this fruit daily reverses aging by 20 years!", "FAKE"),
    ("Deep state plots to overthrow elected government, whistleblower says", "FAKE"),
    ("Mainstream media exposed for hiding truth about secret underground bases", "FAKE"),
    ("Anonymous source confirms world leaders are reptilian shapeshifters!", "FAKE"),
    ("Miracle supplement Big Pharma does not want you to know about!", "FAKE"),
    ("Politician caught on secret tape admitting to global population control!", "FAKE"),
]

df = pd.DataFrame(articles, columns=['text', 'label'])

print("=" * 55)
print("       PROJECT 3: FAKE NEWS DETECTOR")
print("=" * 55)
print(f"\n📊 Dataset Overview:")
print(f"   Total articles : {len(df)}")
print(f"   Real news      : {(df['label']=='REAL').sum()}")
print(f"   Fake news      : {(df['label']=='FAKE').sum()}")

# ── 2. Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# ── 3. TF-IDF Vectorizer ──
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    ngram_range=(1, 2),
    max_features=1000
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── 4. Passive Aggressive Classifier ──
model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
model.fit(X_train_vec, y_train)

# ── 5. Evaluate ──
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Trained Successfully!")
print(f"   Accuracy : {acc * 100:.1f}%")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# ── 6. Live Predictions ──
print("📰 Detecting Fake vs Real News Headlines:")
headlines = [
    "Parliament debates new agricultural reform policies for farmers.",
    "SHOCKING: Secret lab creates zombie virus, CDC hiding the truth!",
    "Central bank holds rates steady amid global economic uncertainty.",
    "PROOF: Famous actor is actually a government clone sent to spy on us!",
    "Research team publishes findings on new renewable energy source.",
]

for headline in headlines:
    vec = vectorizer.transform([headline])
    pred = model.predict(vec)[0]
    label = "✅ REAL" if pred == "REAL" else "🚨 FAKE"
    print(f"   {label} → \"{headline[:55]}\"")
