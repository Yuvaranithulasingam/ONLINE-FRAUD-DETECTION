# ONLINE-FRAUD-DETECTION WEBSITE

### OBJECTIVE  

The goal of this project is to develop a web-based platform where users can input payment details, and the system will analyze the transaction data to predict whether it is fraudulent or legitimate. The platform will use **machine learning models** to detect fraud and provide users with instant feedback on transaction safety.  

### INTRODUCTION 

Online transactions are increasingly vulnerable to fraud due to various cyber threats like **identity theft, credit card fraud, and phishing attacks**. Many users unknowingly fall victim to these scams, resulting in financial losses.  

To address this issue, this project focuses on building a **user-friendly website** that allows users to enter transaction details such as **card number, transaction amount, location, and IP address**. The system will process this information using a **fraud detection machine learning model** and provide an **instant prediction** on whether the transaction is safe or fraudulent.  

### PROBLEM STATEMENT  

With the rise in online payments, users face **uncertainty** about whether their transactions are secure. Many fraud detection mechanisms operate behind the scenes and do not provide **real-time feedback** to users. The main challenges include:  

- **Lack of instant fraud detection** for users before making a transaction.  
- **High false positives** in fraud detection, leading to unnecessary transaction rejections.  
- **Evolving fraud patterns** that make traditional rule-based detection ineffective.  
- **User awareness and security** – many users are unaware of potential fraud risks when entering payment details.  

### KEY QUESTION TO ADDRESS  

- How can users get real-time fraud detection insights?  
- What **machine learning model** can accurately predict fraud based on transaction details?  
- How can we **ensure data security** while processing sensitive payment information?  

### PROPOSED SOLUTION  

This project will develop a **fraud detection website** with the following key features:  

#### **1. User Input Form for Payment Details**  

- Users enter details such as:  
  - **Card number (hashed for security)**  
  - **Transaction amount**  
  - **Transaction type (online purchase, fund transfer, etc.)**  
  - **User location & IP address**  
  - **Time of transaction**  

#### **2. Fraud Detection Model Integration**  
- The system will use a **pre-trained machine learning model** to analyze transaction details.  
- Possible algorithms for fraud detection:  
  - **Logistic Regression** – for quick binary classification.  
  - **Random Forest / Decision Trees** – for better accuracy.  
  - **Gradient Boosting (XGBoost, LightGBM)** – for detecting complex fraud patterns.  
  - **Neural Networks (if needed for advanced analysis).**  

#### **3. Real-Time Prediction Display**  
- The website will display an **instant result** after processing the transaction details:  
  - ✅ **Legitimate Transaction** – If the transaction is safe.  
  - ❌ **Fraudulent Transaction** – If the system detects risk.  

#### **4. Secure Web Interface**  
- The website will be built using **HTML, CSS, JavaScript, Flask/Django** (backend).  
- **User authentication (optional)** – to allow secure logins for frequent users.  
- **API Integration** – for future payment gateway connections.  

#### **5. Data Storage and Analytics (Future Scope)**  
- **Database (MongoDB, MySQL, Firebase, etc.)** for transaction records.  
- **Visualization (Power BI, Matplotlib, etc.)** for fraud trend analysis.

### MODEL TRAINING

### Import necessary libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
```

### Load dataset
```
df = pd.read_csv("fraud_dataset.csv")  # Replace with your actual dataset file
```

![{BB8BF96F-C210-4A13-B31E-2A24DFDF7BD3}](https://github.com/user-attachments/assets/8d445f26-3c5f-4bb5-b0cd-cd472f8091e0)

### Step 1: Handle Missing Values
```
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill missing values with median
```

### Step 2: Encode Categorical Features
```
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])  # Encoding transaction type
```

### Step 3: Feature Engineering
```
df["balance_change"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["recipient_balance_change"] = df["newbalanceDest"] - df["oldbalanceDest"]
df["transaction_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)  # Avoid division by zero
```

### Selecting features and target variable
```
features = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
            "balance_change", "recipient_balance_change", "transaction_ratio"]
X = df[features]
y = df["isFraud"]
```

### Step 4: Normalize Numerical Features
```
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### Step 5: Handle Class Imbalance using SMOTE
```
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### Step 6: Split Data
```
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
```

### Step 7: Train Model (Random Forest)
```
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

![{BF2CE880-EE34-4FFE-B8A0-58AA267FF838}](https://github.com/user-attachments/assets/8fbe7f1b-4285-4c74-8fef-b9f803b13506)

### Step 8: Evaluate Model
```
y_pred = rf_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

![{875EC6C7-8620-4B69-80D1-3C3D4C5B0219}](https://github.com/user-attachments/assets/2660a71e-47dc-49db-b0e7-8a2d1d1701ae)

### Step 9: Hyperparameter Tuning
```
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)
```

![{0304D713-4800-410D-9F0F-230AECBC3665}](https://github.com/user-attachments/assets/0ad0a63c-c02c-4845-932d-9fea790f953d)

### Print best parameters
```
print("Best Parameters:", grid_search.best_params_)
```

![{EE4AFDB0-4A90-4FFC-9144-F4376C2F3F9F}](https://github.com/user-attachments/assets/117f79fc-329e-4f82-ab76-409f82e4eb16)

### Data Visualization
```
sns.set_style("darkgrid")
```

### 🔹 1. Bar Chart - Fraud vs. Non-Fraud Transactions
```
plt.figure(figsize=(6,4))
sns.countplot(x="isFraud", data=df, palette="coolwarm")
plt.title("Fraud vs. Non-Fraud Transactions")
plt.xlabel("Is Fraud (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()
```

![{3EDE6E45-C302-4805-9E3D-7A32E9DB0257}](https://github.com/user-attachments/assets/32a34c56-f4a4-4115-8038-55e3d26d1f81)

### 🔹 2. Scatter Plot - Amount vs. Old Balance Origin (Colored by Fraud)
```
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["amount"], y=df["oldbalanceOrg"], hue=df["isFraud"], palette="coolwarm", alpha=0.5)
plt.title("Amount vs. Old Balance Origin")
plt.xlabel("Transaction Amount")
plt.ylabel("Old Balance Origin")
plt.legend(title="Fraud", loc="upper right")
plt.show()
```

![{33A19E9E-6467-44B2-A8A8-FBE3AC960328}](https://github.com/user-attachments/assets/f86bf3de-8323-4673-81fe-fa757f8094e8)

### 🔹 3. Box Plot - Amount Distribution for Fraud vs. Non-Fraud
```
plt.figure(figsize=(8,5))
sns.boxplot(x="isFraud", y="amount", data=df, palette="coolwarm")
plt.title("Transaction Amount Distribution by Fraud Status")
plt.xlabel("Is Fraud (0 = No, 1 = Yes)")
plt.ylabel("Transaction Amount")
plt.yscale("log")  # Log scale for better visualization
plt.show()
```

![{42D57F31-71AD-4179-862E-7E5DC8B2C0EB}](https://github.com/user-attachments/assets/2b314d5e-23e1-4d5a-8f7c-9b0a9135e42b)

### 🔹 4. Heatmap - Correlation Between Features
```
plt.figure(figsize=(10,6))
corr = df.corr(numeric_only=True)  # Compute correlation matrix
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
```

![{5CB56264-E4AF-4EB9-9327-1304F645ADBE}](https://github.com/user-attachments/assets/0504df51-64cb-41c6-86aa-7f3077231aa1)

### 🔹 5. Histogram - Transaction Amount Distribution
```
plt.figure(figsize=(8,5))
sns.histplot(df["amount"], bins=50, kde=True, color="blue")
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Count")
plt.xscale("log")  # Log scale for better visualization
plt.show()
```

![{2917AE3F-F3F5-4BC9-80DA-97AF5388B7DB}](https://github.com/user-attachments/assets/6f5e892e-17e4-4c64-b319-e72cc3a66b85)

### 🔹 6. Pie Chart - Fraud vs. Non-Fraud Transactions
```
fraud_counts = df["isFraud"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(fraud_counts, labels=["Non-Fraud", "Fraud"], autopct="%1.1f%%", colors=["lightblue", "red"], startangle=90)
plt.title("Fraud vs. Non-Fraud Transaction Distribution")
plt.show()
```

![{411DD628-41FA-46D4-86A3-A510A33B211E}](https://github.com/user-attachments/assets/d95ae80e-b4ec-4c99-afe2-368fddd05fb6)

### EXISTING SYSTEM 

Currently, most fraud detection mechanisms work in the background and do not offer **real-time feedback** to users.  

#### **Limitations of Existing Methods**  

1. **Bank-Based Fraud Detection**  
   - Operates **behind the scenes**, users only get alerts after transactions.  
   - May **block legitimate transactions** due to high false positives.  

2. **Rule-Based Fraud Detection**  
   - Uses **fixed rules** (e.g., blocking large transactions from unknown locations).  
   - Cannot adapt to **new fraud techniques**.  

3. **Machine Learning-Based Detection (Limited Access to Users)**  
   - Used mainly by **banks and payment gateways**, not directly accessible to users.  

### APPLICATION  

This fraud detection website can be useful in:  

- **E-Commerce Websites:** Buyers can check the safety of their transactions before checkout.  
- **Freelancers & Online Payments:** Verify transactions before accepting payments.  
- **Personal Finance Management:** Users can check for fraud risks before sending money.  
- **Educational Purposes:** Understanding how fraud detection models work in real-time.  

### FUTURE ADVANCEMENT  

To enhance the system further, the following improvements can be made:  

#### **1. Payment Gateway Integration**  
- Connect with **Stripe, PayPal, Razorpay, or Google Pay** to verify real transactions.  

#### **2. AI-Powered Fraud Prevention**  
- Implement **adaptive AI models** that **learn from new fraud trends** automatically.  

#### **3. Blockchain-Based Security**  
- Use **blockchain technology** to create a **tamper-proof** transaction log.  

#### **4. Multi-Factor Authentication (MFA)**  
- Implement **OTP verification or biometric authentication** for transaction validation.  

#### **5. Fraud Alerts and Notifications**  
- Send alerts via **email or SMS** for high-risk transactions.  

## CONCLUSION  

This project aims to develop a **fraud detection website** where users can input payment details and receive **real-time fraud predictions**. By integrating **machine learning models**, the system will help users identify potential fraud risks before completing transactions. Future advancements will focus on **AI-driven fraud prevention, secure payment integration, and enhanced user authentication** to make online transactions safer.  

