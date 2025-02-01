#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 17:23:38 2024

@author: shahzod

Contains user defined classes and relevant functions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# Standard Library Imports
import json
import numbers
import warnings
from functools import partial

# Numerical & Statistical Libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import boxcox, yeojohnson, t
from scipy.special import inv_boxcox

# Machine Learning & Preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, 
                             mean_squared_error, r2_score)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# Time Series & Forecasting Models
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet

# Deep Learning Frameworks
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Plotting & Visualization
import matplotlib.pyplot as plt

# Hyperparameter Optimization
import optuna


plt.rcParams["font.family"] = "Times New Roman"


class DataHandler:
    """
    Handles all data-related operations such as loading, preprocessing, and splitting.
    """
 
    def __init__(self, title):
        self.title = title
        self.y_raw = None
        self.y_train = None
        self.y_test = None
        self.features = None
        self.scaler = None
        self.test_ratio = None
        self.non_negative = ['log', 'sqrt', 'boxcox']

    def load_data(self, filepath):
        """
        Loads data from a CSV file and sets it to y_raw.
        """
        print(f"Loading data from {filepath}...")
        self.y_raw = pd.read_csv(filepath, parse_dates=True, index_col=0)

        print(f"Data loaded with shape: {self.y_raw.shape}")
        
    def apply_transformation(self, data, transformation):
        if transformation == 'none':
            self.inverse_power_transform = linear
        elif isinstance(transformation, numbers.Number):
            data /= transformation
            self.inverse_power_transform = partial(product, transformation)
        elif transformation == 'log':
            data = np.log1p(data)
            self.inverse_power_transform = np.expm1
        elif transformation == 'sqrt':
            data = np.sqrt(data)
            self.inverse_power_transform = square
        elif transformation == 'boxcox':
            # finds the value of lmbda that maximizes the log-likelihood function and returns it as the second output argument
            data, lam = boxcox(data.ravel() + 1e-5) # Add a small number to avoid log(0) issues
            # data = pd.DataFrame(data_np, index=data.index, columns=data.columns)
            self.boxcox_lambda = lam
            self.inverse_power_transform = partial(inverse_boxcox, lam)
        elif transformation == 'yeo':
            # Unlike boxcox, yeojohnson does not require the input data to be positive
            data, lam = yeojohnson(data)
            self.yeojohnson_lambda = lam
            self.inverse_power_transform = partial(inverse_yeo_johnson, lam)
        else:
            raise ValueError(f"Unknown transformation: {transformation}")
        self.transformation = transformation
        # return pd.DataFrame(data, index=data.index, columns=data.columns)
        return data
        
    def apply_scaler(self, data, scaler_type):
        if scaler_type == 'none':
            self.scaler = IdentityScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler_type}")

        data = self.scaler.fit_transform(data)
        self.scaler_type = scaler_type
        # return pd.DataFrame(data, index=data.index, columns=data.columns)
        return data

    def preprocess_data(self, transformation='none', scaler_type='none', test_ratio=0.05):
        """
        Applies transformation and scaling to the raw data.
        :param transformation: One of transformations (e.g., 'log', 'sqrt').
        :param scaler_type: Type of scaler to apply (e.g., 'minmax', 'standard', 'robust').
        """
        print("Preprocessing data...")
        if self.y_raw is None:
            raise ValueError("Raw data not loaded. Use `load_data` first.")

        data = self.y_raw.copy()
        # Check if the data has a negative value 
        if transformation in self.non_negative and (data.values < 0).any() and scaler_type != 'minmax':
            raise ValueError('The data has negative entries, which the given transformation does \
                             not support. Try a minmax scaler')
        
        # Split, scale then transform
        y_train, y_test = self.split_data(data, test_ratio)
        
        y_train = self.apply_scaler(y_train, scaler_type)
        y_test = self.scaler.transform(y_test) 

        self.y_train = self.apply_transformation(y_train, transformation)
        self.y_test = self.apply_transformation(y_test, transformation)     
        
        x = self.scaler.transform(data)
        x = self.apply_transformation(x, transformation)     
        self.processed_y = pd.DataFrame(x, index=data.index, columns=data.columns)
        print("Data preprocessing complete.")
        
    def split_data(self, data, test_ratio):
        """
        Splits the data into training and testing sets.
        :param test_ratio: Proportion of data to be used for testing.
        """
        self.test_ratio = test_ratio
        
        split_index = int(len(data) * (1 - test_ratio))
        y_train = data.iloc[:split_index]
        y_test = data.iloc[split_index:]
        self.y_raw_train = self.y_raw.iloc[:split_index]
        self.y_raw_test = self.y_raw.iloc[split_index:]
        self.test_size = len(self.y_raw_test)
        print(f"Data split into train (shape: {y_train.shape}) and test (shape: {y_test.shape}).")
        
        return y_train, y_test
    
    def update_data(self, new_data):
        """
        Updates the raw data with new records and re-applies preprocessing and scaling.
        :param new_data: New data as a DataFrame or array.
        """
        print("Updating data with new records...")
        if self.y_raw is None:
            raise ValueError("Raw data not loaded. Use `load_data` first.")

        # Concatenate new data to raw data
        new_data = pd.DataFrame(new_data, columns=self.y_raw.columns)
        self.y_raw = pd.concat([self.y_raw, new_data])

        # Reapply preprocessing
        self.preprocess_data(transformations=None, scaler_type=None)
        print("Data updated and preprocessed.")

    def inverse_scaler_transform(self, data):
        """
        Applies inverse scaling to the data, if a scaler was used.
        :param data: Scaled data to inverse transform.
        """
        if self.scaler is None:
            raise ValueError("No scaler applied during preprocessing. Cannot inverse transform.")
        return self.scaler.inverse_transform(data)
        # return self.scaler.inverse_transform(data).ravel()
    
    def data_postprocessing(self, data):
        # data = self.inverse_scaler_transform(data)
        # return self.inverse_power_transform(data)
        data = self.inverse_power_transform(data)
        return self.inverse_scaler_transform(data)

    def calculate_statistics(self, data):
        """
        Calculate and return mean and variance of the time series.
        """
        mean = data.mean()
        variance = data.var()
        return {"mean": mean, "variance": variance}

    def visualize_data(self, data, line=True, hist=True, save_file=False):
        """
        Generate visualizations: full series plot and histogram of the distribution.
        """
        name = f'{self.title}_{self.transformation}_{self.scaler_type}_'
        if line:
            plt.figure(figsize=(10, 5))
            plt.plot(data)
            # plt.title(self.title)
            # plt.legend()
            if save_file: save_plot(plt, name + 'line')
            plt.show()
            

        if hist:
            plt.figure(figsize=(6, 5))
            plt.hist(data, color="skyblue", edgecolor="black")
            # plt.title("Histogram of Distribution")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            if save_file: save_plot(plt, name + 'hist')
            plt.show()

    def handle_outliers(self):
        """
        Handle outliers by replacing them with the lower or upper bound.
        """
        Q1 = self.y_raw.quantile(0.25)
        Q3 = self.y_raw.quantile(0.75)
        IQR = Q3 - Q1
        # lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # POTENTIAL BUG: NO data ATTRIBUTE
        cleaned_data = self.data.copy()
        # cleaned_data[cleaned_data < lower_bound] = lower_bound
        cleaned_data[cleaned_data > upper_bound] = upper_bound

        return cleaned_data

    def detrend(self, data, method, plot=False, save_file=False):
        """
        Perform detrending by regressing the data on a constant and time trend.
        """

        if method == 'GLS':
            # Detrending by regressing the data on a constant and time trend
            time = np.arange(len(data.index))
    
            #  Fit an OLS model to estimate residuals
            X = sm.add_constant(time)  # Add constant for intercept
            ols_model = sm.OLS(data, X).fit()
            residuals = ols_model.resid
    
            # Estimate AR(1) autocorrelation
            ar_model = sm.tsa.ARIMA(residuals, order=(1, 0, 0)).fit()
            rho = ar_model.params['ar.L1']
    
            # Convert X to a DataFrame to use .diff()
            X_df = pd.DataFrame(X, columns=["const", "time"], index=data.index)
    
            # GLS Transformation
            y_transformed = data.diff() - rho * data.shift(1).diff()
            X_transformed = X_df.diff() - rho * X_df.shift(1).diff()
    
            # Drop NaN values due to lagging
            y_transformed = y_transformed.dropna()
            X_transformed = X_transformed.iloc[2:]  # Drop NaN
    
            # Fit GLS model
            gls_model = sm.OLS(y_transformed, X_transformed).fit()
    
            # Extract fitted trend and detrend
            fitted_trend = gls_model.predict(X)
            detrended_series = data - fitted_trend
        elif method == 'HP':
            # Perform HP filtering to extract the trend and cycle components.
            detrended_series, fitted_trend = hpfilter(data, lamb=1600*3**4)
        else:
            raise ValueError(f"Unknown scaler: {method}")
            
        if plot:
            # Plot results
            # if method == 'GLS':
            #     fitted_trend += data[0]
                
            # if method == 'HP':
            #     detrended_series += data[0]
                
            plt.figure(figsize=(10, 6))
            plt.plot(data.index, data, label="Original Series")
            plt.plot(data.index, fitted_trend, label="Fitted Trend", linestyle="--")
            plt.plot(data.index, detrended_series, label="Detrended Series", color='green')
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc="lower center", ncol=3)
            plt_title = f"{method}_Detrending_with_{self.transformation}"
            if save_file: 
                save_plot(plt, plt_title)
            
            plt.title(plt_title)
            plt.show()
            
        return detrended_series, fitted_trend

    def stationarity_test(self, data):
        """
        Perform stationarity test using the Augmented Dickey-Fuller test.
        """
        result = adfuller(data)
        return {
            "ADF Statistic": result[0],
            "p-value": result[1],
            "Critical Values": result[4],
            "Stationary": result[1] < 0.05,
        }

    def heteroskedasticity_test(self, data):
        """
        Perform a heteroskedasticity test (ARCH test).
        """
        result = het_arch(data)
        return {"Test Statistic": result[0], 
                "p-value": result[1], 
                "Heteroskedastic": result[1] < 0.05}

    def get_features(self):
        """
        Add month number and decade number to the data.
        """
        data = self.y_raw
        features = pd.DataFrame(index=data.index)
        features["Year"] = data.index.year
        features["Month"] = data.index.month
        features["Decade"] = (data.index.day + 10) // 10
        # self.features = features
        
        # Fourier terms: Use Fourier transforms to represent seasonality in a parsimonious way:
        # time_index = np.arange(len(data))
        # seasonal_period = 3 # num of decades in a month
        # features['sin_term'] = np.sin(2 * np.pi * time_index / seasonal_period),
        # features['cos_term'] = np.cos(2 * np.pi * time_index / seasonal_period),
        
        return features


# Define custom Theil's U function
def theils_u(actual, predicted):
    numerator = np.sqrt(np.mean((predicted - actual) ** 2))
    denominator = np.sqrt(np.mean(actual ** 2)) + np.sqrt(np.mean(predicted ** 2))
    return numerator / denominator


class ModelManager:
    def __init__(self, data_handler, model_save_path):
        """Initialize with time series data."""
        self.data_handler = data_handler
        self.model_save_path = model_save_path
        # self.data = data_handler.y_train
        self.models = {}
        self.point_forecasts = {}
        self.point_forecasts_scaled = {}
        self.fitted_values = {}
        self.fitted_values_scaled = {}
        self.residuals = {}
        self.residuals_scaled = {}
        self.confidence_intervals = {}
        self.variances = {}
        self.ensemble_model_weights = {}
        self.performance_metrics = {}
        self.is_heteroskedastic = False
        os.makedirs(self.model_save_path, exist_ok=True)

    def evaluate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        theil_u = theils_u(y_true.values, y_pred)
        # return {
        #     'MAPE': f'{mape:.4f}',
        #     'R2': f'{r2:.4f}', 
        #     'Theils U': f'{theil_u:.4f}'
        # }

        return {
            'MAPE': mape,
            'R2': r2, 
            'Theils U': theil_u
        }

    def save_model_performance(self, model_name):
        model = self.models[model_name]
        fitted_values_scaled = model.fittedvalues.reshape(-1, 1)
        # When SARIMA, there is jump in the beginning of period
        if model_name.startswith("SARIMA"):
            fitted_values_scaled[:2] = fitted_values_scaled[2]
        self.fitted_values_scaled[model_name] = fitted_values_scaled        
        self.fitted_values[model_name] = self.data_handler.data_postprocessing(
            fitted_values_scaled)
        resid_scaled = model.resid.reshape(-1, 1)
        self.residuals_scaled[model_name] = resid_scaled
        # self.residuals[model_name] = self.data_handler.y_train - self.fitted_values[model_name]#.ravel()
        self.residuals[model_name] = self.data_handler.data_postprocessing(
            resid_scaled)
    
    def fit_sarima(self, data, model_params, model_name):
        """Suppress Warnings including related to frequency"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
            sarima_model = SARIMAX(data, **model_params)
            fitted_model = sarima_model.fit(maxiter=1000, disp=False, cov_type="robust")
        
        self.models[model_name] = fitted_model
        self.save_model_performance(model_name)
        self.save_model(model_params, model_name)
        return fitted_model
    
    def train_sarima(self, y_train, seasonal, seasonal_period=12, name_prefix=''):
        """Train a SARIMA model with auto_arima for parameter selection and save found best orders."""
        print("Training SARIMA model...", flush=True)
        model_name = f'SARIMA_{name_prefix}'
        y_train = np.asarray(y_train)
        
        sarima_model = auto_arima(y_train, 
            start_p=0, start_q=0, max_p=6, max_q=6,
            # start_p=2, start_q=3, max_p=2, max_q=3,
            # d=None,  # Let the model determine the optimal differencing order
            seasonal=True,
            m=seasonal_period,  # Seasonal period length
            start_P=0, start_Q=0, max_P=6, max_Q=6,
            # start_P=1, start_Q=1, max_P=1, max_Q=1,
            D=None,  # Let the model determine the seasonal differencing
            trace=True,  # Prints the testing of different models
            # n_jobs=-1, 
            error_action='ignore',  # Ignore if a model fails
            suppress_warnings=True,
            stepwise=True  # Use a stepwise search to minimize computation
        )
        print(sarima_model.summary(), flush=True)
        model_params = {'order': sarima_model.order, 
                        'seasonal_order': sarima_model.seasonal_order,
                        'trend': 'c' if sarima_model.with_intercept else None}

        fitted_model = self.fit_sarima(y_train, model_params, model_name)
        
        return fitted_model

    def forecast_sarima(self, steps, name_prefix=''):
        """Forecast using the SARIMA model."""
        
        model_name = f'SARIMA_{name_prefix}'
        if model_name not in self.models:
            model_params = self.load_model(model_name)
            self.models[model_name] = self.fit_sarima(self.data_handler.y_train, model_params, model_name)
        
        forecast_scaled = self.models[model_name].forecast(steps=steps).reshape(-1, 1)
        forecast = self.data_handler.data_postprocessing(forecast_scaled)
        self.point_forecasts[model_name] = forecast
        self.point_forecasts_scaled[model_name] = forecast_scaled
        return forecast
    
    def calculate_forecast_variance(self, fitted_model, steps):
        """
        Calculate forecast variance and standard error using a fitted SARIMA model.
    
        Parameters:
        - fitted_model: Fitted SARIMA model object from statsmodels
        - steps: Number of steps ahead to forecast
    
        Returns:
        - forecast_variance: Variance for each forecasted step
        - forecast_se: Standard error for each forecasted step
        """
        # Get residual variance (sigma^2)
        sigma2 = fitted_model.sigma2  # Variance of the residuals
    
        # Extract AR coefficients (phi)
        ar_params = fitted_model.polynomial_ar  # Array of AR coefficients
    
        # Initialize variance accumulation
        forecast_variance = []
    
        for h in range(1, steps + 1):
            # Calculate variance up to step h
            variance_h = sigma2 * (1 + np.sum([np.power(ar_params, 2 * k) for k in range(1, h)]))
            forecast_variance.append(variance_h)
    
        # Convert to numpy array for convenience
        forecast_variance = np.array(forecast_variance)
    
        # Calculate standard error (square root of variance)
        forecast_se = np.sqrt(forecast_variance)
    
        return forecast_variance, forecast_se
    
    def SARIMA_ci(self, alpha=0.05, name_prefix=''):
        """Calculate lower and upper bound of the confidence interval  
        based on heteroskedasticity of the residuals"""
        
        model_name = f'SARIMA_{name_prefix}'
        mean_forecast = self.point_forecasts[model_name].ravel()
        
        critical_values = {0.10: 1.645, 0.05: 1.96, 0.01: 2.575}
        steps = len(self.data_handler.y_test)
        model = self.models[model_name]
        
        conf_int = model.get_forecast(steps).conf_int(alpha=alpha).T
        
        lower_bound = conf_int[0].reshape(-1,1)
        upper_bound = conf_int[1].reshape(-1,1)
        
        lower_bound = self.data_handler.data_postprocessing(lower_bound).ravel()
        upper_bound = self.data_handler.data_postprocessing(upper_bound).ravel()
        
        residual_variance = ((upper_bound - lower_bound) / (2 * critical_values[alpha]))**2
        forecast_var = np.full(len(self.data_handler.y_raw_test), residual_variance)
        
        self.confidence_intervals[model_name] = [lower_bound, upper_bound]
        self.variances[model_name] = forecast_var
        
        return lower_bound, upper_bound
        
    def train_exponential_smoothing(self, y_train, seasonal_periods=12, model_type='triple'):
        """Train an Exponential Smoothing model (double or triple)."""
        print(f"Training Exponential Smoothing model ({model_type})...")
        
        model_name = 'ExponentialSmoothing'
        if model_type == 'double':
            es_model = ExponentialSmoothing(y_train, trend='add', seasonal=None).fit()
        elif model_type == 'triple':
            # Holt-Winters model
            es_model = ExponentialSmoothing(y_train, trend='add', seasonal='add', 
                                            seasonal_periods=seasonal_periods).fit()
        else:
            raise ValueError("Invalid model type. Use 'double' or 'triple'.")
            
        self.models[model_name] = es_model
        self.save_model_performance(model_name)
        
        file_path = os.path.join(self.model_save_path, f"{model_name}.pkl")
        es_model.save(file_path)
        return es_model

    def forecast_exponential_smoothing(self, steps):
        """Forecast using the Exponential Smoothing model."""
        model_name = 'ExponentialSmoothing'
        if model_name not in self.models:
            file_path = os.path.join(self.model_save_path, f"{model_name}.pkl")
            self.models[model_name] = ExponentialSmoothing.load_from_path(file_path)

        forecast_scaled = self.models[model_name].forecast(steps).reshape(-1, 1)
        forecast = self.data_handler.data_postprocessing(forecast_scaled)
        self.point_forecasts[model_name] = forecast
        self.point_forecasts_scaled[model_name] = forecast_scaled
        
        return forecast
    
    def exponentail_smoothing_ci(self, alpha=0.05):
        model_name = 'ExponentialSmoothing'
        # es_fittedvalues = self.fitted_values[model_name]
        es_forecast = self.point_forecasts[model_name].ravel()

        # # Load residuals from mean model
        # residuals = model.resid
        residuals = self.residuals[model_name]
        
        # Remove the outlier in the first period
        # residuals.iloc[0] = residuals.iloc[1:].mean()
        residuals[0] = residuals[1:].mean()
        
        scaling_factor = residuals.std()
        
        # To suppress the issues with scaling in het_arch
        residuals /= scaling_factor
        
        # Check for ARCH effects
        stat, p_value, _, _ = het_arch(residuals)
        if p_value < alpha:
            print("ARCH effects detected. Proceeding with GARCH modeling.")
            from arch import arch_model
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            # plot_acf(residuals**2, lags=20)
            # plot_pacf(residuals**2, lags=20)
        
            # Automated Order Selection
            arch_type = 'GARCH'
            best_aic = float("inf")
            best_order = None
            for p in range(1, 10):  
                for q in range(1, 10):  
                    model = arch_model(residuals, vol=arch_type, p=p, q=q)
                    fit = model.fit(disp="off")
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = {'p':p, 'q':q} #(p, q)
            
            print(f"Best order: {best_order} with AIC: {best_aic}")
            
            from statsmodels.stats.diagnostic import acorr_ljungbox
            het_model = arch_model(residuals, vol=arch_type, **best_order)
            fit = het_model.fit(disp="off")
            lb_test = acorr_ljungbox(fit.resid**2, lags=range(1,37), return_df=True)
            lb_test['Independent'] = lb_test['lb_pvalue'] > alpha
            print(lb_test)
            steps = len(self.data_handler.y_test)
            forecast_var = fit.forecast(horizon=steps).variance.values.ravel()
            forecast_var *= scaling_factor**2
            self.is_heteroskedastic = True
        else:
            forecast_var = (self.fitted_values[model_name].var() + 
                self.residuals[model_name].var())
        
        forecast_std = np.sqrt(forecast_var)

        z = stats.norm.ppf(1-alpha/2)  # For 95% confidence interval
        lower_bound = es_forecast - z * forecast_std
        upper_bound = es_forecast + z * forecast_std
        
        # lower_bound = self.data_handler.data_postprocessing(lower_bound).ravel()
        # upper_bound = self.data_handler.data_postprocessing(upper_bound).ravel()
        
        self.confidence_intervals[model_name] = [lower_bound, upper_bound]
        self.variances[model_name] = forecast_var
        
        return lower_bound, upper_bound
    
    
    def train_prophet(self, y_train, monthly=False, holidays=False):
        """Fit Prophet model with different configurations"""
        model_name = 'Prophet'
        m = Prophet(growth='linear', weekly_seasonality=False)
        
        # Capacities must be supplied for logistic growth in column "cap"       
        # m = Prophet(growth='logistic', weekly_seasonality=False) 
        if monthly: m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        if holidays: m.add_country_holidays(country_name='UZ')
        m.fit(y_train)
        self.models[model_name] = m
        
        train_dates = pd.DataFrame(self.data_handler.y_raw_train.index, columns=['ds'])
        
        fitted_values_scaled = m.predict(train_dates)['yhat'].values.reshape(-1, 1)
        self.fitted_values_scaled[model_name] = fitted_values_scaled        
        self.fitted_values[model_name] = self.data_handler.data_postprocessing(
            fitted_values_scaled)
        self.residuals_scaled[model_name] = self.data_handler.y_train - fitted_values_scaled
        self.residuals[model_name] = self.data_handler.y_raw_train - self.fitted_values[model_name]
        
    
    def forecast_prophet(self):
        model_name = 'Prophet'
        m = self.models[model_name]
        # future = m.make_future_dataframe(periods=steps)
        future = pd.DataFrame(self.data_handler.y_raw_test.index, columns=['ds'])
        prediction = m.predict(future)
        forecast_scaled = prediction['yhat'].values.reshape(-1, 1)
        forecast = self.data_handler.data_postprocessing(forecast_scaled)
        self.point_forecasts[model_name] = forecast
        self.point_forecasts_scaled[model_name] = forecast_scaled
        
        lower_bound_scaled = prediction['yhat_lower'].values.reshape(-1, 1)
        upper_bound_scaled = prediction['yhat_upper'].values.reshape(-1, 1)
        
        lower_bound = self.data_handler.data_postprocessing(lower_bound_scaled).ravel()
        upper_bound = self.data_handler.data_postprocessing(upper_bound_scaled).ravel()        
        self.confidence_intervals[model_name] = [lower_bound, upper_bound]
        
        # Use Prophet's default variance estimate
        residual_variance = ((upper_bound - lower_bound) / (2 * 1.96))**2
        forecast_variance = np.full(len(self.data_handler.y_raw_test), residual_variance)
        self.variances[model_name] = forecast_variance
        
        return forecast
    
    def grid_search_RF(self, X_train, y_train):
        """Perform GridSearchCV for later usage of the best parameters"""
        print('Performing GridSearchCV...')
        
        tscv = TimeSeriesSplit(n_splits=5)

        # Random Forest
        rf = RandomForestRegressor(random_state=42)

        # Define the parameter grid to search over
        param_grid = {
            'bootstrap': [True, False],
            'n_estimators': [50, 150, 200, 250, 300, 400, 600],       # Number of trees in the forest
            'max_depth': [1, 3, 5, 10, 20, 30, 40, 60, 80, 100],        # Maximum depth of each tree
            'min_samples_split': [1, 2, 5, 10],        # Minimum samples required to split a node
            'min_samples_leaf': [1, 2, 4],          # Minimum samples required at each leaf node
            'max_features': [1.0, 'sqrt', 'log2']  # Number of features to consider at each split
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid, 
            cv=tscv,                         # TimeSeriesSplit cross-validation
            n_jobs=-1,                    # Use all available cores
            scoring='neg_mean_absolute_error',  # Choose an appropriate scoring metric
            verbose=1                    # Set verbosity for progress output
        )

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train.ravel())
        best_params = grid_search.best_params_
        print("GridSearchCV results", best_params)
        
        return best_params
        
        
    def train_random_forest(self, X_train, y_train, num_top_models=10):
        """Train an RandomForestRegressor model with Optuna for hyperparameter optimization."""
        
        # Define the objective function
        def objective(trial):
            # Define the hyperparameter search space
            params = {
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "n_estimators": trial.suggest_int("n_estimators", 50, 800, step=50),
                "max_depth": trial.suggest_int("max_depth", 10, 100),
                "min_samples_split": trial.suggest_int("min_samples_split", 1, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", [1.0, "sqrt", "log2", None]),
            }
        
            # Initialize RandomForestRegressor with hyperparameters
            model = RandomForestRegressor(**params, random_state=42)
        
            # TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            rmse_scores = []
        
            # Time-series aware cross-validation
            for train_index, val_index in tscv.split(X_train):
                X_t, X_val = X_train[train_index], X_train[val_index]
                y_t, y_val = y_train[train_index], y_train[val_index]
        
                model.fit(X_t, y_t)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmse_scores.append(rmse)
        
            # Return the negative mean RMSE for optimization
            return -np.mean(rmse_scores)
        
        # Create the study
        study = optuna.create_study(direction="maximize")  # Maximize because we're using negative RMSE
      
        best_grid_params = self.grid_search_RF(X_train, y_train)

        # Enqueue the GridSearchCV best parameters as an initial trial
        study.enqueue_trial(best_grid_params)
        
        print("Training RandomForestRegressor model with Optuna...")
        study.optimize(objective, n_trials=250, n_jobs=-1)
        
        # Save top trials
        best_trials = sorted(study.trials, key=lambda x: x.value)[:num_top_models]
        best_params = [trial.params for trial in best_trials]
        
        # Insert best results from Grid search because it improves overall results
        best_params.insert(0, best_grid_params)
    
        self.models['RandomForestRegressor'] = best_params
        self.save_model(best_params, 'RandomForestRegressor')
        
    def forecast_random_forest(self, X_train, y_train, X_test):
        """Forecast using the Random Forest model."""
        model_name = 'RandomForestRegressor'
        
        if model_name not in self.models:
            best_params = self.load_model(model_name)
        else:
            best_params = self.models[model_name]
        
        final_predictions = []
        for params in best_params:
            model = RandomForestRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            test_predictions = model.predict(X_test)
            final_predictions.append(test_predictions)
        
        # self.point_forecasts_scaled[model_name] = final_predictions
        return final_predictions
    
    def random_forest_bootstrap(self, X_train, y_train, X_test, n_models=3, n_bootstraps=100):
        """Build the confidence intervals using Random Forest models and bootstrapping"""
        model_name = 'RandomForestRegressor'
        
        if model_name not in self.models:
            best_params = self.load_model(model_name)
        else:
            best_params = self.models[model_name]
            
        best_params = best_params[:n_models]
        
        predictions = []
        bootstrap_predictions = []
        
        print("Starting bootstrapping for Random Forest")
        # Bootstrap resampling
        for i in range(n_bootstraps):
            X_bootstrap, y_bootstrap = resample(X_train, y_train, random_state=i)
            for params in best_params:
                rf_bootstrap = RandomForestRegressor(**params, random_state=42)
                rf_bootstrap.fit(X_bootstrap, y_bootstrap)
                test_prediction = rf_bootstrap.predict(X_test)
                predictions.append(test_prediction)
            # # Average predictions across the 10 models
            prediction = np.mean(predictions, axis=0)
            bootstrap_predictions.append(prediction)

        # Convert to numpy array
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate mean prediction and intervals
        mean_prediction = bootstrap_predictions.mean(axis=0)
        lower_bound = np.percentile(bootstrap_predictions, 2.5, axis=0)
        upper_bound = np.percentile(bootstrap_predictions, 97.5, axis=0)
        
        self.confidence_intervals[model_name] = [lower_bound, upper_bound]
        
        return mean_prediction, lower_bound, upper_bound
    
    def random_forest_tree_ci(self, X_train, y_train, X_test, n_models=3):
        """Build the confidence intervals using Random Forest models"""
        model_name = 'RandomForestRegressor'
        
        if model_name not in self.models:
            best_params = self.load_model(model_name)
        else:
            best_params = self.models[model_name]
            
        best_params = best_params[:n_models]
        lower_bounds = []
        upper_bounds = []
        
        for params in best_params:
            rf_model = RandomForestRegressor(**params, random_state=42)
            rf_model.fit(X_train, y_train)

            # Get predictions from individual trees
            tree_predictions = np.array([
                tree.predict(X_test) for tree in rf_model.estimators_
                ])
            
            # Calculate mean prediction and intervals
            lower_bound = np.percentile(tree_predictions, 2.5, axis=0)
            upper_bound = np.percentile(tree_predictions, 97.5, axis=0)
            
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        
        lower_bound = np.mean(lower_bounds, axis=0)
        upper_bound = np.mean(upper_bounds, axis=0)
        
        return lower_bound, upper_bound
    
    def forecast_cycle_RF(self, detrended_series, sarima_prefix, pretrain=False, plot=False, save_file=False):
        model_name = 'RandomForestRegressor'
        data_handler = self.data_handler
        trend_forecast = self.point_forecasts_scaled[f'SARIMA_{sarima_prefix}'].ravel()
        trend_fittedvalues = self.fitted_values_scaled[f'SARIMA_{sarima_prefix}'].ravel()

        # Extract features for RF forecasting
        features = data_handler.get_features()
        # enc = OneHotEncoder(handle_unknown='ignore')
        # X = enc.fit_transform(features)
        
        date_scaler = MinMaxScaler()
        X = date_scaler.fit_transform(features.to_numpy())
        
        X_train, X_test, cycle_train, cycle_test = train_test_split(X, detrended_series, 
                                                    test_size=data_handler.test_ratio, 
                                                    shuffle=False)
        
        num_top_models = 10
        
        if pretrain: self.train_random_forest(X_train, cycle_train, num_top_models)
        
        # Calculate the cycle forecast using saved best params
        predictions = self.forecast_random_forest(X_train, cycle_train, X_test)
        
        # Determine the number of member RF models and
        # save the best cycle+trend forecast in the manager
        best_num_models = self.calc_composite_score(predictions, f'SARIMA_{sarima_prefix}', num_top_models, plot=True, save_file=True)
        
        # Get model fit values
        cycles_fitted = self.forecast_random_forest(X_train, cycle_train, X_train)
        cycle_fitted = np.mean(cycles_fitted[:best_num_models], axis=0)
        self.fitted_values_scaled[model_name] = cycle_fitted
        
        rf_fittedvalues_scaled = trend_fittedvalues.ravel() + cycle_fitted
        rf_fittedvalues = data_handler.data_postprocessing(rf_fittedvalues_scaled.reshape(-1, 1))
        self.fitted_values[model_name] = rf_fittedvalues.reshape(-1, 1)
        
        lower_bound, upper_bound = self.random_forest_tree_ci(
            X_train, cycle_train, X_test, n_models=best_num_models)
        
        lower_bound += trend_forecast
        upper_bound += trend_forecast
        
        rf_forecast = self.point_forecasts[model_name]
        lower_bound = data_handler.data_postprocessing(lower_bound.reshape(-1, 1))
        upper_bound = data_handler.data_postprocessing(upper_bound.reshape(-1, 1))
        
          
        residual_variance = ((upper_bound - lower_bound) / (2 * 1.96))**2
        forecast_variance = residual_variance.ravel()
        self.variances[model_name] = forecast_variance
        
        if plot:
            ind = data_handler.y_raw_test.index
            plt.figure(figsize=(10, 6))
            plt.plot(ind, data_handler.y_raw_test, label="Actual", color="black")
            plt.plot(ind, rf_forecast, label="Mean Prediction", color="blue")
            plt.fill_between(
                ind, 
                lower_bound.ravel(), upper_bound.ravel(), 
                color="gray", alpha=0.2, label="95% Prediction Interval")
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc="lower center", ncol=3)
            if save_file: save_plot(plt, f'{model_name}_conf')
            plt.title("Confidence Intervals for Random Forest")
            plt.show()
            
    def create_sequences(self, data, look_back):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)
    
    def build_seq2seq_model(self, input_shape, latent_dim=50):
        encoder_inputs = tf.keras.layers.Input(shape=(input_shape[1], 1))
        encoder = tf.keras.layers.GRU(latent_dim, return_state=True, dropout=0.2, recurrent_dropout=0.2)
        encoder_outputs, state_h = encoder(encoder_inputs)
    
        decoder_inputs = tf.keras.layers.RepeatVector(1)(state_h)
        decoder_gru = tf.keras.layers.GRU(latent_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
        decoder_dense = tf.keras.layers.Dense(1)
        decoder_outputs = decoder_dense(decoder_outputs)
    
        model = tf.keras.models.Model(encoder_inputs, decoder_outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def train_GRU(self, train, test, look_back):
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # train_scaled = scaler.fit_transform(train.reshape(-1, 1))
        # test_scaled = scaler.transform(test.reshape(-1, 1))
    
        X_train, y_train = self.create_sequences(train, look_back)
        X_test, y_test = self.create_sequences(test, look_back)
    
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
        model = self.build_seq2seq_model(X_train.shape, latent_dim=50)
    
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    
        model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, 
                  validation_data=(X_test, y_test), 
                  # callbacks=[early_stopping, lr_schedule]
                  )
    
        predictions = model.predict(X_test).reshape(-1)
        # y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
        # predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
    
        return predictions

    # def forecast_cycle_GRU(self, detrended_series, sarima_prefix, pretrain=False, plot=False):
    def forecast_cycle_GRU(self, look_back, pretrain=False, plot=False):
        model_name = 'GRU'
        data_handler = self.data_handler
        train = self.data_handler.y_train
        test = self.data_handler.y_test
        scaled_forecast = self.train_GRU(train, test, look_back=look_back) 
        self.point_forecasts_scaled[model_name] = scaled_forecast
        GRU_forecast = self.data_handler.data_postprocessing(scaled_forecast.reshape(-1, 1))
        self.point_forecasts[model_name] = GRU_forecast
        
        if plot:
            ind = data_handler.y_raw_test.index[look_back:]
            plt.figure(figsize=(10, 6))
            plt.plot(ind, data_handler.y_raw_test[look_back:], label="True Values", color="black")
            plt.plot(ind, GRU_forecast, label="Mean Prediction", color="blue")
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc="lower center", ncol=3)
            plt.title("Confidence Intervals for GRU")
            plt.show()
        
    def comp_score_df(self, df):
        # Convert to DataFrame for easier handling
        # df = pd.DataFrame(metrics, dtype=float)

        # Normalize the metrics
        df['MAPE_norm'] = (df['MAPE'] - df['MAPE'].min()) / (df['MAPE'].max() - df['MAPE'].min())
        df['R2_norm'] = (df['R2'] - df['R2'].min()) / (df['R2'].max() - df['R2'].min())
        df['TheilsU_norm'] = (df['Theils U'] - df['Theils U'].min()) / (df['Theils U'].max() - df['Theils U'].min())

        # Define weights for the metrics
        weights = {'MAPE': 2/3, 'R2': 1/6, 'Theils U': 1/6}

        # Calculate composite score
        df['Composite_Score'] = (
            weights['MAPE'] * df['MAPE_norm'] +
            weights['R2'] * (1 - df['R2_norm']) +
            weights['Theils U'] * df['TheilsU_norm']
        )
        
        return df
        
    def calc_composite_score(self, predictions, trend_model_name, 
                             num_top_models, plot=False, save_file=False):
        # Calculate mean forecasts for different number of models
        # predictions = self.point_forecasts_scaled[cycle_model_name]
        trend_forecast = self.point_forecasts_scaled[trend_model_name].ravel()
        
        metrics = []
        rf_forecasts = []
        for k in range(1, num_top_models + 1):
            cycle_forecast = np.mean(predictions[:k], axis=0)
            scaled_forecast = trend_forecast + cycle_forecast
            rf_forecast = self.data_handler.data_postprocessing(scaled_forecast.reshape(-1, 1))
            rf_forecasts.append(rf_forecast)
            metrics.append(self.evaluate_metrics(self.data_handler.y_raw_test, rf_forecast))
        
        # Calculate composite score
        df = pd.DataFrame(metrics, dtype=float)
        df = self.comp_score_df(df)

        # Find the best ensemble
        # best_index = df['Composite_Score'].idxmax()
        best_index = df['Composite_Score'].idxmin()
        best_ensemble = df.loc[best_index]
        
        print(df)
        print("Best Ensemble:")
        print(best_ensemble)
        
        if plot:
            plot_index = range(1, len(metrics) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(plot_index, df['MAPE'], label='MAPE', marker='o')
            plt.plot(plot_index, df['R2'], label='1-R2', marker='o')
            plt.plot(plot_index, df['Theils U'], label="Theil's U", marker='o')
            plt.plot(plot_index, df['Composite_Score'], label="Composite Score", marker='o')
            plt.xlabel('Number of Models for averaging')
            plt.ylabel('Metric Value')
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc="lower center", ncol=4)
            plt.grid()
            if save_file: save_plot(plt, 'composite_score')
            plt.title('Metrics vs Ensemble Size')
            plt.show()
        
        forecast = rf_forecasts[best_index]
        self.point_forecasts['RandomForestRegressor'] = forecast
        # self.point_forecasts['RandomForestRegressor'] = self.data_handler.data_postprocessing(forecast.reshape(-1, 1))
        
        # Add 1 to indicate the number of models
        return best_index + 1
    
    def __ensemble_ci(self, model_name):
        avg_forecast = self.point_forecasts[model_name]
        residuals = self.data_handler.y_raw_test.values.ravel() - avg_forecast
        meta_model_var = residuals.var()
        model_weights = self.ensemble_model_weights[model_name]
        # member_model_vars = np.fromiter(self.variances.values(), dtype=float)
        
        ## Convert a list like or variances into an array where each scalar 
        ## element is broadcasted to match the length of the largest array
        
        # variance_vals = self.variances.values()
        variance_vals = self.variance_values
        # Find the length of the longest array in the list
        max_length = max(len(x) if isinstance(x, np.ndarray) else 1 for x in variance_vals)
        # Broadcast scalars to arrays and stack them
        member_model_vars = np.vstack([x if isinstance(x, np.ndarray) else np.full(max_length, x) for x in variance_vals])
        total_var = np.dot(model_weights**2, member_model_vars) + meta_model_var
        blended_std = np.sqrt(total_var)

        z = 1.96
        lower_bound = avg_forecast - z * blended_std
        upper_bound = avg_forecast + z * blended_std

        self.residuals[model_name] = residuals
        self.confidence_intervals[model_name] = [lower_bound, upper_bound]
        
    
    def blending(self, models_fit, forecasts, weighted=False):
        """Calculate average forecast"""
        is_weighted = "_weighted" if weighted else ""
        model_name = f'blending{is_weighted}'
        
        if weighted:
            y_raw_train = self.data_handler.y_raw_train
            MAEs = [mean_absolute_error(y_raw_train, model_fit) for model_fit in models_fit]
            MAEs = np.array(MAEs)
            MAEs = 1 / MAEs # to make more error -> less weight
            total_MAE = MAEs.sum()
            model_weights = MAEs/total_MAE
            # print('model_weights', model_weights)
            avg_forecast = np.average(forecasts, axis=0, weights=model_weights)
        else:
            avg_forecast = np.mean(forecasts, axis=0)
            num_models = len(forecasts)
            model_weights = np.array([1/num_models]*num_models)
            
        self.point_forecasts[model_name] = avg_forecast
        self.ensemble_model_weights[model_name] = model_weights
        self.__ensemble_ci(model_name)
        
        return avg_forecast
    
    def stacking(self, models_fit, forecasts):
        model_name = 'stacking'
        # stacked_model = GradientBoostingRegressor()
        stacked_model = LinearRegression()
        stacked_model.fit(models_fit.T, self.data_handler.y_raw_train)
        # print('coef', stacked_model.coef_)
        # stacked_model.fit(models_fit.T, data_handler.y_raw_train)
        stacked_forecast = stacked_model.predict(forecasts.T)
        
        self.point_forecasts[model_name] = stacked_forecast.ravel()
        self.ensemble_model_weights[model_name] = stacked_model.coef_[0]
        self.__ensemble_ci(model_name)
        return stacked_forecast
    
    def inverse_expected_variance(self, forecasts):
            variances = np.array(self.variance_values)
            inv_var = 1/variances
            total_vars = inv_var.sum(axis=0)
            weights = inv_var / total_vars
            ensemble_forecast = np.array([np.dot(forecasts[:, n], weights[:, n]) 
                                for n in range(forecasts.shape[1])
                                ])
            
            model_name = 'inverse_variance'
            
            self.point_forecasts[model_name] = ensemble_forecast
            self.ensemble_model_weights[model_name] = weights.mean(axis=1)
            self.__ensemble_ci(model_name)
            
            return ensemble_forecast
    
    def best_ensemble(self, model_names, plot=False, save_file=False):
        y_raw_test = self.data_handler.y_raw_test
        forecasts = np.array([self.point_forecasts[name] for name in model_names])
        models_fit = np.array([self.fitted_values[name] for name in model_names])
        
        # Store only the variances of needed models
        self.variance_values = list(map(self.variances.get, model_names))
        
        # Drop one level: e.g. (3, 12, 1) -> (3, 12)
        forecasts = forecasts.reshape(forecasts.shape[:2])
        models_fit = models_fit.reshape(models_fit.shape[:2])
        
        ensemble_names = ['blending', 'blending_weighted', 'stacking', 'inverse_variance']
        
        blended_forecast = self.blending(models_fit, forecasts)
        blended_forecast_w = self.blending(models_fit, forecasts, weighted=True)
        stacked_forecast = self.stacking(models_fit, forecasts)
        inv_var_forecast = self.inverse_expected_variance(forecasts)
        ensemble_forecasts = [blended_forecast, blended_forecast_w, stacked_forecast, inv_var_forecast]
        
        metrics = []
        for name, forecast in zip(ensemble_names, ensemble_forecasts):
            metrics.append(self.evaluate_metrics(y_raw_test, forecast))
            self.plot_confidence_interval(name, name.replace('_', ' ').title(), save_file=True)
        
        # Calculate composite score
        df = pd.DataFrame(metrics, dtype=float)
        df = self.comp_score_df(df)
        df.to_excel('metrics_ensemble.xlsx')
        clm = ['MAPE', 'R2', 'Theils U', 'Composite_Score']
        latex_table = df[clm].round(4).sort_values(['Composite_Score']).to_latex(index=False)
        with open('metrics_ensemble.tex', 'w') as f:
            f.write(latex_table)
        
        # Find the best ensemble
        # best_index = df['Composite_Score'].idxmax()
        best_index = df['Composite_Score'].idxmin()
        best_ensemble = df.loc[best_index]
        
        print(df)
        print("Best Ensemble:", ensemble_names[best_index])
        print(best_ensemble)
        
        if plot:
            ind = y_raw_test.index
            plt.figure(figsize=(12, 6))
            plt.plot(ind, y_raw_test, label='Actual', color='black')
            # plt.plot(ind, blended_forecast_w, label='Weighted blending forecast', color='blue')
            plt.plot(ind, blended_forecast, label='Blending forecast', color='blue')
            plt.plot(ind, stacked_forecast, label='Stacking forecast', color='darkgreen')
            plt.plot(ind, inv_var_forecast, label='Inverse Variance forecast', color='red')
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc="lower center", ncol=4)
            if save_file: save_plot(plt, 'ensemble_comparison')
            plt.title("Ensemble Forecasts vs Actual Data")
            plt.show()
            
        return ensemble_forecasts[best_index]
        
    def dm_test(self, actual, forecast1, forecast2, loss='MSE', h=1):
        """
        Conduct the Diebold-Mariano test for predictive accuracy.
        
        Parameters:
            actual: np.array
                Array of actual values.
            forecast1: np.array
                Array of forecasts from model 1.
            forecast2: np.array
                Array of forecasts from model 2.
            loss: str
                Loss function to use ('MSE' for mean squared error, 'MAE' for mean absolute error).
            h: int
                Forecast horizon.
        
        Returns:
            DM test statistic and p-value.
        """
        # Calculate forecast errors
        e1 = actual - forecast1
        e2 = actual - forecast2
        
        # Define the loss differential
        if loss == 'MSE':
            d = e1**2 - e2**2
        elif loss == 'MAE':
            d = np.abs(e1) - np.abs(e2)
        else:
            raise ValueError("Invalid loss function. Use 'MSE' or 'MAE'.")
        
        # Calculate mean and variance of loss differential
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        
        # Adjust variance for forecast horizon
        n = len(d)
        denom = (2 * (h - 1) / h) if h > 1 else 0
        dm_stat = d_mean / np.sqrt((d_var / n) + denom)
        
        # Compute p-value
        p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=n-1))
        
        if p_value < 0.05:
            conclusion = "There is difference (Fail to reject the null hypothesis)."
            # HO: the two models have equal forecasting accuracy
        else:
            conclusion = "No significant difference between the models."
    
        return conclusion, p_value
    
    def plot_forecasts(self, actual, model_names, labels, show=False, save_file=False):
        """Plot actual vs. forecasted values."""
        forecasts = np.array([self.point_forecasts[name] for name in model_names])
        # Drop one level: e.g. (3, 12, 1) -> (3, 12)
        forecasts = np.array(forecasts)
        forecasts = forecasts.reshape(forecasts.shape[:2])
        
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual, label='Actual', color='black')
        for forecast, label in zip(forecasts, labels):
            plt.plot(actual.index[-len(forecast):], forecast, label=label)
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc="lower center", ncol=len(labels)+1)
        if save_file: save_plot(plt, 'models_comparison')
        plt.title("Model Forecasts vs Actual Data")
        if show: plt.show()
        
    def plot_confidence_interval(self, model_name, legend, show=False, save_file=False):
        ind = self.data_handler.y_raw_test.index
        
        plt.figure(figsize=(10, 6))
        plt.plot(ind, self.data_handler.y_raw_test, label='Actual')
        plt.plot(ind, self.point_forecasts[model_name], label=f'{legend} forecast')

        plt.fill_between(
            ind,
            self.confidence_intervals[model_name][0],
            self.confidence_intervals[model_name][1],
            color='gray',
            alpha=0.2,
            label='95% confidence interval'
        )
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc="lower center", ncol=3)
        if save_file: save_plot(plt, f'{model_name}_conf')
        # plt.title("")
        if show: plt.show()
        
    def save_model(self, model_params, model_name):
        """Save a trained model to disk."""
        file_path = os.path.join(self.model_save_path, f"{model_name}.json")
        with open(file_path, 'w') as outfile: 
            json.dump(model_params, outfile)
        print(f"Model {model_name} saved")

    def load_model(self, model_name):
        """Load a saved model from disk."""
        file_path = os.path.join(self.model_save_path, f"{model_name}.json")
        if os.path.exists(file_path):
            print(f"Loading model {model_name}")
            with open(file_path, 'r') as inp:
                return json.load(inp)
        else:
            print(f"Model '{model_name}' not found at {file_path}.")
            return None
    
def save_plot(plt, title):
    plots_folder = "./Plots"
    file_path = os.path.join(plots_folder, f"{title}.png")
    plt.savefig(file_path, dpi=600, transparent=False, bbox_inches='tight')
        
  
# For multiprocessing to be able to pickle, define inverse power  
# transforamtion functions outside of the DataHandler class
def linear(x): return x
def square(x): return x**2
def product(x, transformation): return x * transformation
def inverse_boxcox(lam, x): 
    return inv_boxcox(x, lam)
    # return (np.exp(np.log(lam * x + 1) / lam) - 1) if lam != 0 else np.exp(x) - 1

def inverse_yeo_johnson(lam, transformed_data):
    """
    Perform the inverse Yeo-Johnson transformation on the given data.

    Parameters:
    - lam: float, the lambda parameter used in the original transformation
    - transformed_data: array-like, transformed data to invert

    Returns:
    - original_data: array-like, data in the original scale
    """
    transformed_data = np.array(transformed_data)
    original_data = np.zeros_like(transformed_data)
    
    # For y >= 0
    pos_mask = transformed_data >= 0
    if lam != 0:
        original_data[pos_mask] = (transformed_data[pos_mask] * lam + 1) ** (1 / lam) - 1
    else:
        original_data[pos_mask] = np.exp(transformed_data[pos_mask]) - 1
    
    # For y < 0
    neg_mask = transformed_data < 0
    if lam != 2:
        original_data[neg_mask] = 1 - (-transformed_data[neg_mask] * (2 - lam) + 1) ** (1 / (2 - lam))
    else:
        original_data[neg_mask] = -np.exp(-transformed_data[neg_mask]) + 1
    
    return original_data


class IdentityScaler(BaseEstimator, TransformerMixin):
    """A scaler that performs no transformation (identity transformation).
    - Pipeline compatable: It integrates smoothly into scikit-learn pipelines.
    - Useful for conditional scaling: Use it when you want to decide dynamically 
    whether to scale the data or leave it untouched.
    - Provides consistency: Makes the code cleaner and avoids handling special 
    cases where no scaling is required.
    """
    def fit(self, X, y=None):
        # Nothing to fit; return self
        return self
    
    def transform(self, X):
        # Return the data unchanged
        return np.asarray(X)
    
    def inverse_transform(self, X):
        # Return the data unchanged
        return np.asarray(X)
    
    def fit_transform(self, X, y=None):
        # Fit and transform in one step
        return self.fit(X, y).transform(X)