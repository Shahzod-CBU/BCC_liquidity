# **Time Series Forecasting Framework**

This repository provides a **comprehensive forecasting framework** that integrates **econometric models, machine learning techniques, and ensemble learning methods** to predict **banking system liquidity**. The framework is designed to be **scalable, robust, and efficient**, leveraging **multiprocessing** for high-performance execution.

---

## **📌 Key Features**

### **1️⃣ Model Implementations**
This framework includes a diverse range of models from econometrics and machine learning.

#### **📊 Econometric Models**
- **SARIMA (Seasonal Autoregressive Integrated Moving Average)**  
  - Automatically selects optimal hyperparameters using `pmdarima.auto_arima`.  
  - Uses **power-transformed** data for stability.  

- **Exponential Smoothing (ETS - Error-Trend-Seasonality)**  
  - Implements **Holt-Winters' method** with additive trend & seasonality.  
  - Forecast intervals are derived using residual variance analysis and **GARCH modeling**.

- **Prophet (Facebook/Meta)**  
  - Handles missing values and outliers effectively.  
  - Includes monthly seasonality & Uzbekistan-specific holiday effects.  
  - Confidence intervals generated from posterior simulations.

#### **🤖 Machine Learning Models**
- **Random Forest with HP Filtering**  
  - Decomposes series into **trend** and **cycle components** using the **Hodrick-Prescott filter**.  
  - **Trend** is forecasted using **SARIMA**, while **cycle** is predicted using **Random Forest**.  
  - **Hyperparameter tuning** follows a **two-step approach**:
    - **GridSearchCV** (`TimeSeriesSplit`) for initial parameter selection.
    - **Optuna** refinement using `enqueue_trial` to enhance parameters.
  - **Final forecast** is an **average of top-performing models**.

- **GRU-Based Neural Network**  
  - Implements a **Seq2Seq GRU** architecture.  
  - Multiple look-back windows tested for sequence dependency optimization.

---

### **2️⃣ Ensemble Learning for Improved Forecasting Accuracy**
- **Weighted Blending:**  
  - Combines **SARIMA, Prophet, and Random Forest** forecasts.  
  - Weights determined using **train MAE**.

- **Stacking:**  
  - Uses **Linear Regression** as a **meta-model** to optimally combine forecasts.
 
- **Inverse of expected variance:**
  - Weights of ensemble members are determined based on the inverse of their predicted model variances.

---

### **3️⃣ Confidence Intervals & Uncertainty Estimation**
- **Econometric Models**:  
  - **GARCH modeling** of residual variance for heteroskedastic data.  
  - **Standard residual variance approach** when no ARCH effects are detected.

- **Random Forest & Tree-Based Models**:  
  - Confidence intervals computed using the **percentile method** based on tree predictions (2.5th & 97.5th percentiles).

---

## **⚡ Multiprocessing for High Performance**
To handle large-scale forecasting efficiently, the framework leverages **multiprocessing**:
- Uses **ProcessPoolExecutor** to **parallelize model training** across different scalers and transformations.  
- Runs **independent processes** to significantly reduce runtime for large datasets.  
- **Hyperparameter tuning with Optuna** is executed **concurrently** for faster optimization.

---

## **📂 Project Structure**
```plaintext
📦 project-folder
│-- 📄 ud_classes1.py      # Core classes for data handling, models, evaluation, and tuning
│-- 📄 app_multi1.py       # Implements multiprocessing pipeline for model training
│-- 📂 data/               # Directory for dataset storage
│-- 📂 results/            # Directory for storing model results & visualizations
│-- 📜 README.md           # Project documentation
```

## **📌 Installation & Usage**
1. **Clone the repository**
   ```sh
   git clone https://github.com/Shahzod-CBU/BCC_liquidity.git
   cd BCC_liquidity
   ```

2. **Install required dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the main script**
   ```sh
   python app_multi1.py
   ```

---

Contributions are welcome! Feel free to open an issue or submit a pull request.
