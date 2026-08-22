# Smart Recruitment Assistant — AI Screening System
### خطة الـ Sprints الكاملة (3 أيام تقريبًا) — v1.1 (Notebook مشترك)

**الفكرة:** نظام ذكي يساعد الـ HR يقرر هل الـ Candidate يستاهل ينتقل للمرحلة الجاية في التوظيف ولا لأ، بالاعتماد على بيانات المتقدم (تعليم، خبرة، تدريب...)، مع Dashboard يوضح إحصائيات وأهم العوامل المؤثرة في القرار، ونظام Ranking اختياري لأفضل 10 مرشحين.

**الستاك:** Python, Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn, Joblib/Pickle, Streamlit

**أسلوب الشغل:** Notebook واحد مشترك (`Smart_Recruitment_Assistant.ipynb`) بيتشارك فيه الفريق كله عن طريق Google Colab / Jupyter + Git. كل شخص ليه Section واضح جوه الـ Notebook بعنوان باسمه، وما حدّش يعدّل في Section حد تاني من غير ما يقوله. آخر الشغل، الـ Notebook كله بيتحول لسكريبت واحد بسيط بيتحمّل في Streamlit.

**الفريق:**

| # | الاسم | المسؤولية الأساسية | الـ Section في الـ Notebook |
|---|-------|---------------------|------------------------------|
| 1 | Person 1 | Data Cleaning + Preprocessing | Section 2 |
| 2 | Person 2 | EDA + Visualizations | Section 3 |
| 3 | Person 3 | Logistic Regression + Evaluation | Section 4 |
| 4 | Person 4 | Random Forest + Model Comparison | Section 5 |
| 5 | Person 5 | Streamlit + Dashboard + Integration | Section 6 + app.py |

**قاعدة شغل من الأول:** كل Sprint لوحده، محدّش يبدأ Section بتاعه قبل ما اللي قبله (خصوصًا الـ Preprocessing) يخلص ويتأكد إنه شغال بشكل نضيف (خلايا مرتبة، من غير أخطاء، والـ Output ظاهر). كل عضو يعمل commit/push على الـ Notebook بعد ما يخلص Section بتاعه، وميلغيش خلايا حد تاني.

---

### 📂 الـ Project Structure النهائي (مبسّط)
```
smart-recruitment-assistant/
├── data/
│   └── recruitment_dataset.csv          # الداتا الأصلية
├── Smart_Recruitment_Assistant.ipynb    # النوتبوك المشترك (كل الأقسام)
├── models/
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
├── app/
│   └── app.py                            # Streamlit application
├── reports/
│   └── performance_comparison.md         # مقارنة الموديلات (نسخة من النوتبوك)
├── requirements.txt
└── README.md
```

---

## 📅 اليوم الأول — الداتا والفهم (Notebook: Section 1، 2، 3)

### ☐ Sprint 1 — Setup + Section 1: فهم الداتا (30-40 دقيقة، الفريق كله)
**هنعمل:**
- [ ] إنشاء الـ repo + رفع الداتا في `data/`
- [ ] إنشاء `Smart_Recruitment_Assistant.ipynb` وكتابة الـ Headers لكل الـ Sections من الأول (حتى لو فاضية) عشان الترتيب يبقى واضح من البداية
- [ ] **Section 0 — Setup:** استدعاء المكتبات (pandas, numpy, sklearn, matplotlib, seaborn, joblib)
- [ ] **Section 1 — Data Understanding:** `.info()`, `.describe()`, `.isnull().sum()`, وكتابة ملاحظات في خلية Markdown تحتها
- [ ] الاتفاق على تعريف الـ Target Variable وكتابته في خلية Markdown أول النوتبوك

**التولز:** Python, Pandas, Google Colab/Jupyter, Git

**المسؤول:** الفريق كله (كل واحد يجرب في نسخته الأول، وبعدين يتفقوا على نسخة واحدة نضيفة)

---

### ☐ Sprint 2 — Section 2: Data Cleaning + Preprocessing (ساعة - ساعة ونص)
**هنعمل:**
- [ ] معالجة الـ Missing Values (imputation مناسب لكل عمود)
- [ ] Encoding للـ Categorical Features (Education Level, Company Type, Employment History...)
- [ ] Feature Engineering بسيط (مثلاً تحويل experience من نص لرقم)
- [ ] Feature Scaling للأعمدة الرقمية
- [ ] Train/Test Split (مع مراعاة الـ Class Imbalance لو موجود)
- [ ] كل خلية فيها الكود ومعاها خلية Markdown صغيرة بتشرح إيه اللي اتعمل وليه

**التولز:** Pandas, Scikit-learn (LabelEncoder/OneHotEncoder, StandardScaler, train_test_split)

**المسؤول:** Person 1

**الهدف من الخطوة:** آخر خلية في الـ Section دي لازم تطلع `X_train, X_test, y_train, y_test` نضاف وجاهزين لأي حد في الفريق يكمل عليهم

---

### ☐ Sprint 3 — Section 3: EDA + Visualizations (ساعة - ساعة ونص)
**هنعمل:**
- [ ] توزيع الـ Target Variable (متوازن ولا لأ؟)
- [ ] العلاقة بين كل Feature والـ Target (Education vs Target, Company Type vs Target...)
- [ ] Correlation Heatmap بين الأعمدة الرقمية
- [ ] رسومات لأكثر صفات الـ Candidates شيوعًا (توزيع التعليم، الخبرة، ساعات التدريب)
- [ ] تلخيص أهم 3-4 ملاحظات في خلية Markdown آخر الـ Section (هتتفيد بيها في الـ Dashboard بعدين)

**التولز:** Matplotlib, Seaborn, Pandas

**المسؤول:** Person 2

**نهاية اليوم الأول:** الداتا نضيفة وجاهزة (Section 2) + فهم واضح لطبيعة المتقدمين (Section 3) ✅

---

## 📅 اليوم الثاني — الموديلات (Notebook: Section 4، 5)

### ☐ Sprint 4 — Section 4: Logistic Regression + Evaluation (ساعة - ساعة ونص)
**هنعمل:**
- [ ] تدريب Logistic Regression على `X_train, y_train` بتاعة Person 1
- [ ] حساب Accuracy, Precision, Recall, F1 Score
- [ ] رسم Confusion Matrix
- [ ] استخراج أهم الـ Coefficients (أهم العوامل المؤثرة حسب الموديل ده)
- [ ] حفظ الموديل بـ Joblib: `joblib.dump(model, "models/logistic_regression.pkl")`

**التولز:** Scikit-learn, Joblib

**المسؤول:** Person 3

---

### ☐ Sprint 5 — Section 5: Random Forest + Model Comparison (ساعة ونص - ساعتين)
**هنعمل:**
- [ ] تدريب Random Forest على نفس الـ Train/Test
- [ ] حساب نفس الـ Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
- [ ] استخراج Feature Importance
- [ ] حفظ الموديل بـ Joblib: `joblib.dump(model, "models/random_forest.pkl")`
- [ ] **جدول مقارنة** بين الموديلين على كل الـ Metrics في نفس الـ Section
- [ ] تحديد الموديل الأفضل واللي هيتحمّل في التطبيق
- [ ] (Bonus) خلية إضافية لحساب `predict_proba` وترتيب أفضل 10 Candidates تنازليًا حسب الـ Confidence

**التولز:** Scikit-learn, Joblib, Pandas

**المسؤول:** Person 4 (بالتعاون مع Person 3 في جزء المقارنة)

**نهاية اليوم الثاني:** موديلين متدربين، متقارنين، ومحفوظين في `models/` ✅

---

## 📅 اليوم الثالث — Dashboard + التطبيق (Notebook: Section 6 + app.py)

### ☐ Sprint 6 — Section 6: تجهيز الـ Insights للـ Dashboard (45 دقيقة - ساعة)
**هنعمل:**
- [ ] تجميع أهم العوامل المؤثرة (من Feature Importance بتاع الموديلين) في جدول واحد
- [ ] تجميع إحصائيات المتقدمين النهائية (من ملاحظات Section 3) في شكل جاهز للعرض
- [ ] تجهيز جدول مقارنة أداء الموديلات بشكل نهائي (نفس بتاع Sprint 5) عشان يتنقل بسهولة لـ Streamlit

**التولز:** Pandas, Matplotlib/Seaborn

**المسؤول:** Person 2 + Person 5

---

### ☐ Sprint 7 — app.py: Streamlit Prediction Interface (ساعة - ساعة ونص)
**هنعمل:**
- [ ] صفحة رئيسية فيها وصف مختصر عن المشروع (Home Page description)
- [ ] فورم إدخال بيانات الـ Candidate (نفس الأعمدة اللي اتدرب عليها الموديل بالظبط)
- [ ] تحميل الموديل المحفوظ بـ Joblib عند بدء التطبيق
- [ ] زرار "Predict Candidate" يطلع النتيجة:
  - 🟢 Recommended for next stage — Confidence: XX%
  - 🔴 Not recommended — Confidence: XX%
- [ ] التأكد إن الـ Input بيتعامل بنفس طريقة الـ Preprocessing بتاعة Section 2 بالظبط (نفس الـ Encoding/Scaling)

**التولز:** Streamlit, Joblib

**المسؤول:** Person 5

---

### ☐ Sprint 8 — إضافة الـ Dashboard + Top 10 داخل التطبيق (45 دقيقة - ساعة)
**هنعمل:**
- [ ] إضافة Tab/Sidebar للـ Dashboard (باستخدام الـ Insights من Sprint 6)
- [ ] إضافة Tab اختيارية لعرض Top 10 Candidates (لو Section 5 اتعمل فيها الجزء ده)
- [ ] اختبار التطبيق كامل من البداية للنهاية (End-to-End Test)

**التولز:** Streamlit

**المسؤول:** Person 5 (بمراجعة سريعة من Person 2 و Person 4 للأرقام المعروضة)

---

### ☐ Sprint 9 — README + التوثيق النهائي (30-45 دقيقة، الفريق كله)
**هنعمل:**
- [ ] كتابة فكرة المشروع والهدف منه في README
- [ ] رابط/مسار الـ Notebook المشترك + شرح مختصر لكل Section ومين عملها
- [ ] ذكر الـ Tech Stack كامل
- [ ] خطوات التشغيل (فتح الـ Notebook للتدريب + تشغيل Streamlit)
- [ ] لقطات شاشة من التطبيق
- [ ] كل عضو يكتب فقرة قصيرة عن مساهمته (مطلوب رسميًا إن كل عضو يساهم في التحضير، الموديل، التقييم، التوثيق والعرض)

**التولز:** Markdown, أداة screenshot

**المسؤول:** الفريق كله

---

### ☐ (Bonus) Sprint 10 — Deployment
- [ ] رفع التطبيق على Streamlit Community Cloud
- [ ] هنعملها لو فضل وقت، مش أساسية لو الوقت ضيق

**المسؤول:** Person 5

---

## 🎯 الـ Deliverables النهائية (Checklist سريع للمراجعة قبل التسليم)
- [ ] Candidate Screening Model (موديلين + الأفضل منهم محدد بوضوح)
- [ ] Performance Comparison (جدول واضح داخل الـ Notebook)
- [ ] Recruitment Dashboard (داخل Streamlit)
- [ ] Hiring Recommendations (insights مكتوبة)
- [ ] Final Presentation
- [ ] Saved Model (Joblib/Pickle)
- [ ] Streamlit Recruitment Screening Application
- [ ] Candidate Prediction Interface (input + prediction + confidence)
- [ ] Top-10 Candidate Recommendation Page (Bonus)

---

**قاعدة أساسية طول الوقت:** الـ Notebook المشترك ده مسؤولية الكل — محدّش يمسح أو يعدّل في Section حد تاني من غير ما يتفقوا. كل واحد يخلص Section بتاعه، يتأكد إنه شغال من غير Errors، وبعدين يعمل commit واضح باسمه.

**تذكير خاص بالـ Deployment (Sprint 7):** لازم الـ Input في Streamlit يتعامل بنفس طريقة الـ Preprocessing اللي اتعملت في Section 2 بالظبط، وإلا الـ Prediction هيبقى غلط حتى لو الموديل نفسه كويس.
