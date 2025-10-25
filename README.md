# British Airways Data Science Simulation
## Lounge Demand Forecasting & Customer Booking Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-red.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

> A comprehensive data science project addressing two critical business challenges: lounge capacity planning and customer booking prediction with business-optimized thresholds.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [Task 1: Lounge Demand Modeling](#task-1-lounge-demand-modeling)
- [Task 2: Predictive Booking Model](#task-2-predictive-booking-model)
- [Technical Stack](#technical-stack)
- [Results & Impact](#results--impact)
- [Key Learnings](#key-learnings)
- [Project Structure](#project-structure)

---

## 🎯 Project Overview

This project tackles two critical business problems for British Airways:

1. **Lounge Demand Forecasting**: Creating a scalable lookup table to estimate lounge eligibility across different flight groupings
2. **Customer Booking Prediction**: Building a machine learning model to predict booking completion with business-optimized thresholds

---

## 💼 Business Context

### Problem Statement

British Airways needs to:
- **Plan lounge capacity** at Heathrow Terminal 3 without exact flight details
- **Predict customer bookings** to optimize marketing spend and maximize revenue
- **Balance precision vs recall** based on real business costs (£1,000 revenue per booking, £10 marketing cost)

### Why This Matters

- Lounge overcrowding damages premium customer experience
- Poor booking predictions lead to either wasted marketing spend or missed revenue
- Business decisions require interpretable, calibrated models

---

## 📊 Task 1: Lounge Demand Modeling

### Objective
Create a generalized lookup table estimating lounge eligibility percentages without relying on specific flight numbers or aircraft types.

### Approach

**Grouping Strategy:**
- **Time of Day**: Morning, Lunchtime, Afternoon, Evening
- **Haul Type**: Long-haul (outside Europe) vs Short-haul (within Europe)

**Rationale:**
- Scalable to future schedules
- Reflects actual customer behavior patterns
- Simple enough for operational teams to use

### Key Findings

| Tier | Percentage | Strategic Implication |
|------|-----------|----------------------|
| **Tier 3 (Club Lounge)** | 78% | Requires maximum scalability and space |
| **Tier 2 (First Lounge)** | 20% | Focus on service quality for business travelers |
| **Tier 1 (Concorde Room)** | 1-2% | Premium experience, small but critical segment |

**Critical Insight**: Distribution is remarkably stable across time of day and haul type, which simplifies capacity planning significantly.

### Deliverables

✅ Lookup table with tier percentages by flight grouping  
✅ Justification document explaining grouping choices  
✅ Visualizations showing eligibility patterns  
✅ Actionable recommendations for lounge investments

---

## 🤖 Task 2: Predictive Booking Model

### The Challenge

**Dataset**: 5,000 customer booking records  
**Class Distribution**: 85% no booking, 15% booking complete  
**Key Challenge**: Severe class imbalance makes traditional accuracy metrics misleading

### Exploratory Data Analysis

#### Feature Analysis Summary

**Numerical Features:**
- Purchase lead time, length of stay, flight hour/duration, number of passengers
- Applied distribution analysis, skewness testing, and Q-Q plots

**Categorical Features:**
- Sales channel, trip type, booking origin (104 countries), route (799 unique), flight day
- Evaluated using chi-square tests, Cramér's V, and mutual information

**Binary Features:**
- Extra baggage, preferred seat, in-flight meal selections

#### Feature Selection Decisions

| Feature | Decision | Rationale |
|---------|----------|-----------|
| `booking_origin` | ✅ **KEEP** | Top 10 origins cover 91% of data, MI = 0.061 |
| `route` | ❌ **DROP** | 799 categories, 73% have <50 samples, excessive noise |
| `flight_day` | ❌ **DROP** | MI ≈ 0, only 2% booking rate variation |
| `purchase_lead` | ❌ **DROP** | Weak signal despite multiple transformation attempts |
| Binary features | ❌ **DROP** | All showed MI < 0.005, driven by necessity not intent |

### Feature Engineering

#### Encoding Strategies

**Low-Cardinality Features:**
- Label encoding for `sales_channel`, `trip_type`

**High-Cardinality Features:**
- Frequency encoding for `booking_origin`, `route`
- Out-of-Fold (OOF) smoothed target encoding for `booking_origin`
  - Smoothing parameter k=10 to regularize rare categories
  - Prevents data leakage through proper CV splits

**Temporal Features:**
- Cyclical encoding for `flight_day` and `flight_hour`
- Sine/cosine transformations to preserve periodicity
- Ensures hour 23 is mathematically close to hour 0

**Numerical Features:**
- Winsorization at 1st/99th percentile for outlier handling
- Preserves data while reducing extreme value influence

#### Interaction Features

Created meaningful interactions to capture complex patterns:
- `fhourbucket_x_routefreq`: Flight timing × route popularity
- `lenstay_x_fduration`: Stay length × flight duration
- `originTE_x_fhourbucket`: Geographic × temporal patterns

### Modeling Approach

#### Models Evaluated

| Model | Configuration | Purpose |
|-------|--------------|---------|
| **Random Forest (Baseline)** | `class_weight='balanced'` | Establish baseline, feature importance |
| **Random Forest + SMOTE** | Oversampling minority class | Test synthetic data approach |
| **XGBoost** | `scale_pos_weight=5.69` | Production model with better performance |
| **Stacked Ensemble** | XGBoost + base predictions | Final optimized model |

#### Handling Class Imbalance

**Techniques Evaluated:**

1. **SMOTE** (Synthetic Minority Over-sampling)
   - Result: Minimal improvement, slight AUC drop (0.75 → 0.738)
   - Decision: ❌ Not adopted

2. **Class Weights**
   - Implementation: `class_weight='balanced'` (RF), `scale_pos_weight` (XGBoost)
   - Result: ✅ Stable performance without synthetic data complications
   - Decision: ✅ Adopted

3. **Threshold Tuning**
   - Result: ✅ Critical for business optimization
   - Decision: ✅ Primary strategy for production deployment

### Model Performance

#### Cross-Validation Results

**5-Fold Stratified Cross-Validation:**
- Individual models saved as `xgb_fold1.pkl` through `xgb_fold5.pkl`
- Ensemble predictions from all folds used for final evaluation

**Test Set Metrics:**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.766 | Strong discriminative power; good separation between bookers and non-bookers |
| **Accuracy** | 0.70 | Balanced overall performance despite 15% class imbalance |
| **Recall (Positive Class)** | 0.71 | High ability to capture actual bookers - crucial for revenue potential |
| **Precision (Positive Class)** | 0.29 | Lower due to class imbalance; acceptable given priority on Recall |
| **Brier Score** | 0.112 | Excellent probability calibration; scores reliably reflect true likelihood |

#### Strategic Justification

**Why prioritize Recall over Precision?**

The business cost of missing a potential booker is higher than the cost of occasionally targeting a non-booker. A 71% Recall ensures we identify the vast majority of high-value conversion opportunities, maximizing revenue potential despite moderate precision.

#### Feature Importance (SHAP Analysis)

**Top 5 Predictive Drivers:**

| Rank | Feature | Importance | Business Insight |
|------|---------|-----------|------------------|
| 1 | **Booking Origin TE** | Highest | Geographic conversion rates - historical booking patterns by origin strongly predict likelihood |
| 2 | **Booking Origin Frequency** | High | High-traffic origins indicate established demand and higher booking propensity |
| 3 | **Flight Duration (Medium)** | Medium | Non-extreme flight durations (medium-haul) show optimal conversion rates vs very short or very long flights |
| 4 | **Length of Stay (Winsorized)** | Medium | Longer planned stays indicate higher commitment level, increasing booking odds |
| 5 | **Stay × Flight Duration Interaction** | Medium | Nonlinear interaction captures nuanced booking behavior - the relationship between trip length and flight time reveals complex customer intent |

**Key Finding**: The model effectively identifies booking patterns driven by **origin market conversion history** and **customer travel parameters** (duration, stay length), providing clear levers for marketing and conversion strategy optimization.

**Calibration Quality**: The reliability plot shows the model follows perfect calibration closely, with predicted probabilities reliably reflecting actual booking likelihoods across all probability ranges.

### Model Calibration

Comprehensive calibration analysis across multiple methods:

| Method | Brier Score | ECE | Assessment |
|--------|-------------|-----|------------|
| Raw Model | 0.1120 | 0.0109 | Excellent - already well calibrated |
| Platt Scaling | 0.1119 | 0.0041 | Undesired sharp probability drops |
| Temperature Scaling | 0.1120 | 0.0106 | T=1 indicates no adjustment needed |
| Isotonic Regression | 0.1115 | 0.0000 | Marginal gain but potential overfitting |

**Decision**: Deploy raw model - probabilities are already reliable for business decisions.

### Business-Optimized Threshold Selection

#### Strategic Goal

**Primary Objective**: Identify high-potential bookers early in the funnel to enable highly actionable and profitable targeted engagement.

#### Model Target

**What We're Predicting**: The probability that an individual customer will proceed to complete a booking transaction.

**Why This Matters**: Early identification allows for:
- Personalized marketing interventions
- Resource allocation optimization
- Proactive customer engagement strategies

#### Algorithm & Training

- **Model**: XGBoost (Regularized + CV)
- **Training Data**: 40,000 samples
- **Class Distribution**: 15% positive (booking completion)
- **Validation**: 5-Fold Stratified Cross-Validation with saved models per fold

#### Performance at Business-Optimal Threshold

The threshold was selected to **maximize recall** while maintaining acceptable precision, reflecting the business reality that missing a potential booking is significantly more costly than occasional false positives.

**Strategic Justification**: 
- 71% Recall ensures we identify the vast majority of high-value conversion opportunities
- Moderate precision (29%) is acceptable given the low cost of targeting non-converters relative to missing actual bookers
- This strategy maximizes total revenue potential over false positive minimization

---

## 🏆 Results & Impact
<br>
<img width="705" height="713" alt="Screenshot 2025-10-25 124534" src="https://github.com/user-attachments/assets/60450035-b8e6-4c99-8c44-ff252050e160" />
<img width="704" height="711" alt="Screenshot 2025-10-25 124616" src="https://github.com/user-attachments/assets/4653559e-e78a-43e6-bb50-d19e3c7c2679" />



### Model Performance Summary

| Metric | Score | Business Interpretation |
|--------|-------|------------------------|
| **ROC-AUC** | 0.766 | Strong discriminative ability to separate bookers from non-bookers |
| **Accuracy** | 0.70 | Balanced performance considering 15% class imbalance |
| **Recall** | 0.71 | Captures 71% of actual bookers - maximizes revenue opportunity |
| **Precision** | 0.29 | Acceptable tradeoff given low cost of false positives |
| **Brier Score** | 0.112 | Excellent probability calibration for reliable predictions |

### Business Takeaway

The model effectively identifies booking patterns driven by **origin market conversion history** and **customer travel parameters** (duration, stay length), providing clear levers for marketing and conversion strategy optimization.

**Strategic Advantage**: Early funnel identification enables highly actionable and profitable targeted engagement with high-potential customers.

### Lounge Planning Impact

**Operational Benefits:**
- Clear capacity planning targets: 78% Tier 3, 20% Tier 2, 2% Tier 1
- Prevents Club Lounge (Tier 3) overcrowding
- Right-sizing for business traveler volumes
- Maintains Tier 1 exclusivity without over-investment

### Key Achievements

✅ **Modular Encoding Architecture**: Built custom encoding modules (`categorical_encoding.py`, `numerical_encoding.py`, `routeEncoding.py`, `clusterEncoder.py`) for reusable feature transformations  
✅ **Proper ML Practices**: Stratified 5-fold CV with saved models per fold, prevented data leakage in OOF encoding  
✅ **Model Calibration**: Brier Score = 0.112 with reliability plot confirming excellent calibration  
✅ **Business Alignment**: Recall-optimized strategy aligned with real conversion funnel economics  
✅ **Interpretability**: SHAP analysis provides clear feature attribution and business insights  
✅ **Production-Ready**: Five trained XGBoost models saved as pickle files for ensemble deployment

---

## 💡 Key Learnings

### What Worked

✅ **OOF Target Encoding**: Properly implemented CV-safe encoding prevented leakage while regularizing rare categories  
✅ **Business-First Optimization**: Threshold tuning based on real costs significantly outperformed generic metric optimization  
✅ **Feature Selection Discipline**: Removing weak features (`wants_*`, `purchase_lead`) improved model stability  
✅ **Calibration Focus**: Well-calibrated probabilities enabled confident threshold decisions  

### What Didn't Work

❌ **SMOTE**: Added noise rather than signal, decreased overall performance  
❌ **Over-Engineering `purchase_lead`**: Multiple transformations and interactions showed no improvement  
❌ **Default Thresholds**: Standard 0.5 threshold completely missed business objectives  
❌ **Binary Add-on Features**: Customer extras (meals, bags, seats) showed almost zero predictive power  

### Surprising Findings

💡 **Geographic Dominance**: 44% of model power from booking origin alone - far stronger than expected  
💡 **Stable Tier Distribution**: Lounge eligibility barely varies by time/haul (simplifies planning dramatically)  
💡 **Purchase Lead Weakness**: Booking lead time showed minimal signal despite strong business intuition  
💡 **Calibration Quality**: Model achieved excellent calibration without explicit calibration techniques  

---

## 🛠️ Technical Stack

### Core Libraries

- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Visualization**: `matplotlib`, `seaborn`
- **Model Interpretation**: `shap`
- **Statistical Analysis**: `scipy`

### Key Techniques

- **Encoding**: Label, Frequency, OOF Target, Cyclical (sine/cosine)
- **Validation**: Stratified K-Fold Cross-Validation
- **Imbalance Handling**: Class Weighting, SMOTE (evaluated)
- **Calibration**: Brier Score, ECE, Reliability Diagrams
- **Interpretation**: SHAP Values, Feature Importance

---

## 📁 Project Structure

```
british-airways-simulation/
│
├── data/
│   └── raw/
│       ├── British Airways Summer Schedule.csv
│       └── customer_booking.csv
│
├── notebooks/
│   ├── categorical_encoding.py       # Custom categorical encoders
│   ├── clusterEncoder.py             # Clustering-based encoding
│   ├── eda.ipynb                     # Exploratory Data Analysis
│   ├── feature_iter_encoding.py      # Iterative feature encoding
│   ├── numerical_encoding.py         # Numerical feature transformations
│   ├── prediction_task1.ipynb        # Lounge demand modeling
│   └── routeEncoding.py              # Route-specific encoding logic
│
├── README.md
└── requirements.txt
```

---

## 📚 Documentation

This repository includes:

- **Jupyter Notebooks**: Step-by-step analysis with detailed explanations
- **PDF Documentation**: Complete 64-page technical report covering all iterations
- **Visualizations**: Comprehensive plots for EDA, model evaluation, and business impact

---

## 📧 Contact

**Author**: Adithi Sreenath
**LinkedIn**: https://www.linkedin.com/in/adithi-sreenath/ 
**Email**: aditisjadhav5@gmail.com

---

## 📄 License

This project is for educational and portfolio purposes. All data and scenarios are part of the British Airways virtual simulation program.

---

⭐ If you found this project interesting or useful, consider starring the repository!

