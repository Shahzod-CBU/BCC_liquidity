#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:21:16 2025

@author: shahzod

Train with different data transformations using multiprocessing
"""

import os
from ud_classes1 import DataHandler, ModelManager
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import numpy as np
import pandas as pd

models_folder = "path_to_folder"

factor = "Factor_name"
csv_folder = f"path_to_folder"
file_path = os.path.join(csv_folder, f"{factor}.csv")

def transformation_train(data_handler):
    pretrain = True
    print("Processing", data_handler.transformation, data_handler.scaler_type, 'scaler', flush=True)

    test_size = data_handler.test_size
    
    # Get train and test datasets
    y_train = data_handler.y_train
    y_raw_test = data_handler.y_raw_test
    
    # Initialize the ModelManager
    model_save_path = os.path.join(models_folder, factor, 
               f'{data_handler.transformation}_{data_handler.scaler_type}')
    manager = ModelManager(data_handler, model_save_path)

    # Train Prophet
    prophet_y_train = pd.DataFrame(zip(data_handler.y_raw_train.index, y_train.ravel()),
                                   columns=['ds', 'y'])
    manager.train_prophet(prophet_y_train, monthly=True, holidays=True)
    prophet_forecast = manager.forecast_prophet()
    print('Prophet with monthly and holidays', manager.evaluate_metrics(y_raw_test, prophet_forecast))

    performance = manager.evaluate_metrics(y_raw_test, prophet_forecast)
    print('Prophet', data_handler.transformation, performance, flush=True)
    manager.performance_metrics['Prophet'] = performance

    # Train SARIMA
    sarima_prefix = 'total'
    if pretrain:
        sarima_model = manager.train_sarima(y_train, seasonal=True, seasonal_period=36, name_prefix=sarima_prefix)
    # model_params = {"order": [2, 1, 3], "seasonal_order": [1, 0, 1, 36], "trend": None}
    # sarima_model = manager.fit_sarima(y_train, model_params, f'SARIMA_{sarima_prefix}')
    # sarima_model.plot_diagnostics(figsize=(15, 12))
    sarima_forecast = manager.forecast_sarima(steps=test_size, name_prefix=sarima_prefix)
    performance = manager.evaluate_metrics(y_raw_test, sarima_forecast)
    print(f'SARIMA_{sarima_prefix}', data_handler.transformation, performance, flush=True)
    manager.performance_metrics[f'SARIMA_{sarima_prefix}'] = performance

    es_model = manager.train_exponential_smoothing(y_train, model_type='triple')
    es_forecast = manager.forecast_exponential_smoothing(steps=test_size)
    performance = manager.evaluate_metrics(y_raw_test, es_forecast)
    manager.performance_metrics['ExponentialSmoothing'] = performance
    
    # One cannot get good RF without scaling the data
    if data_handler.scaler_type != 'none':
        detrended_series, trend = data_handler.detrend(data_handler.processed_y, method='HP', plot=False)
        sarima_prefix = 'trend_rf'
        trend = np.asarray(trend)
        trend_train, trend_test = trend[:-test_size], trend[-test_size:]
        trend_model = manager.train_sarima(trend_train, seasonal=False, seasonal_period=1, name_prefix=sarima_prefix)
        manager.forecast_sarima(steps=test_size, name_prefix=sarima_prefix)
        trend_forecast = manager.point_forecasts_scaled[f'SARIMA_{sarima_prefix}'].ravel()
        
        manager.forecast_cycle_RF(detrended_series, sarima_prefix, pretrain=pretrain, plot=False)
        # manager.forecast_cycle_RF(detrended_series, sarima_prefix, pretrain=True, plot=False)
        rf_forecast = manager.point_forecasts['RandomForestRegressor']
        performance = manager.evaluate_metrics(y_raw_test, rf_forecast)
        manager.performance_metrics['RandomForestRegressor'] = performance

        # Train GRU
        # look_back = 5
        # manager.forecast_cycle_GRU(look_back, pretrain=True, plot=False)
        # GRU_forecast = manager.point_forecasts['GRU']
        # performance = manager.evaluate_metrics(y_raw_test[look_back:], GRU_forecast)
        # manager.performance_metrics['GRU'] = performance
        # print(data_handler.transformation, data_handler.scaler_type, performance, flush=True)
    return manager

    
if __name__ == '__main__':
    WORKERS = max(os.cpu_count()-1, 1)
    
    transformations = ['yeo', 'none'] # 'log', 1e13, 'sqrt', 'boxcox'
    test_ratio = 0.045
    scalers = ['none', 'standard', 'robust', 'minmax'] #

    # Initialize DataHandler
    data_handler = DataHandler(title=factor)

    # Load data
    data_handler.load_data(file_path)
    
    runners = []
    for scaler in scalers:
        for transformation in transformations:
            copy_data_handler = deepcopy(data_handler)
            copy_data_handler.preprocess_data(transformation, scaler, test_ratio)
            runners.append(copy_data_handler)
            
    processes = min(WORKERS, len(runners))
    print('Number of processes', processes, flush=True)
    with ProcessPoolExecutor(max_workers=processes) as executor:
        results = executor.map(transformation_train, runners)
    
    managers = []
    dfs = []
    # Combine the training results
    for manager in results: 
        managers.append(manager)
        df = pd.DataFrame.from_dict(manager.performance_metrics, orient='index')
        df['Transformation'] = manager.data_handler.transformation
        df['Scaler'] = manager.data_handler.scaler_type
        df.reset_index(names='Model', inplace=True)
        dfs.append(df)
    
    metrics = pd.concat(dfs, ignore_index=True)
    model_names = metrics['Model'].unique()
    df_group = metrics.groupby('Model')
    
    model_save_path = os.path.join(models_folder, factor)
    manager = ModelManager(data_handler, model_save_path)
    
    best_combinations = {}
    for model_name in model_names:
        df = df_group.get_group(model_name).copy() 
        df = manager.comp_score_df(df)
        best_index = df['Composite_Score'].idxmin()
        best_comb = df.loc[best_index]
        preprocess_types = best_comb[['Transformation', 'Scaler']].values
        best_combinations[model_name] = preprocess_types
        
        df.to_excel(f'metrics_{model_name}.xlsx')
        clm = ['Transformation', 'Scaler', 'MAPE', 'R2', 'Theils U', 'Composite_Score']
        latex_table = df[clm].round(4).sort_values(['Composite_Score']).to_latex(index=False)
        with open(f'metrics_{model_name}.tex', 'w') as f:
            f.write(latex_table)
        
        print(df)
        print(best_comb)
        print("Best Combination:", model_name, preprocess_types)

        # sarima_prefix = 'total'    
        # model_name = f'SARIMA_{sarima_prefix}'
        transformation, scaler_type = best_combinations[model_name]
        manager.model_save_path = os.path.join(model_save_path, f'{transformation}_{scaler_type}')    
    
        manager.data_handler = DataHandler(title=factor)
        data_handler = manager.data_handler
        data_handler.load_data(file_path)
        data_handler.preprocess_data(transformation, scaler_type, test_ratio)
        
        test_size = data_handler.test_size
        y_raw_test = data_handler.y_raw_test
        
        if model_name == 'SARIMA_total':
            sarima_prefix = 'total'
            # Fit SARIMA
            # sarima_model = manager.train_sarima(y_train, seasonal=True, seasonal_period=36, name_prefix=sarima_prefix)
            forecast = manager.forecast_sarima(steps=test_size, name_prefix=sarima_prefix)
            manager.SARIMA_ci(alpha=0.05, name_prefix=sarima_prefix)
            manager.plot_confidence_interval(model_name, "SARIMA", show=True, save_file=True)
        elif model_name == 'ExponentialSmoothing':
            es_model = manager.train_exponential_smoothing(data_handler.y_train, model_type='triple')
            forecast = manager.forecast_exponential_smoothing(steps=test_size)
            manager.exponentail_smoothing_ci(alpha=0.05)
            manager.plot_confidence_interval(model_name, "Exponential Smoothing", show=True, save_file=True)
        elif model_name == 'Prophet':
            prophet_y_train = pd.DataFrame(zip(data_handler.y_raw_train.index, 
                                               data_handler.y_train.ravel()),
                                           columns=['ds', 'y'])
            manager.train_prophet(prophet_y_train, monthly=True, holidays=True)
            forecast = manager.forecast_prophet()
            manager.plot_confidence_interval(model_name, 'Prophet with monthly & holidays', save_file=True)

            
        performance = manager.evaluate_metrics(y_raw_test, forecast)
        # print(model_name, performance)
        manager.performance_metrics[model_name] = performance

    # Fit trend and cycle for Random Forest    
    detrended_series, trend = data_handler.detrend(data_handler.processed_y, method='HP', plot=False)
    sarima_prefix = 'trend_rf'
    trend = np.asarray(trend)
    # trend_train, trend_test = trend.iloc[:-test_size], trend.iloc[-test_size:]
    trend_train, trend_test = trend[:-test_size], trend[-test_size:]
    trend_model = manager.train_sarima(trend_train, seasonal=False, seasonal_period=1, name_prefix=sarima_prefix)
    manager.forecast_sarima(steps=test_size, name_prefix=sarima_prefix)
    # trend_forecast = manager.point_forecasts_scaled[f'SARIMA_{sarima_prefix}'].ravel()

    manager.forecast_cycle_RF(detrended_series, sarima_prefix, pretrain=False, plot=True, save_file=True)
    
    forecast = manager.point_forecasts[model_name]
    performance = manager.evaluate_metrics(y_raw_test, forecast)
    print(model_name, performance)
    manager.performance_metrics[model_name] = performance
    
    model_names = ['Prophet', 'SARIMA_total','RandomForestRegressor']
    labels = ['Prophet', 'SARIMA', 'Random Forest']
    
    manager.plot_forecasts(actual=y_raw_test, model_names=model_names, labels=labels, save_file=True)
    
    ensemble_forecast = manager.best_ensemble(model_names, plot=True, save_file=True)
    sarima_forecast = manager.point_forecasts['SARIMA_total']
    dm_conclusion, p_value = manager.dm_test(y_raw_test.values, ensemble_forecast, sarima_forecast, loss='MAE', h=test_size)
    print(f"DM Test: {dm_conclusion} p-value: {p_value:.4f}")
    
    print(best_combinations)
    print(manager.performance_metrics)
        
        
