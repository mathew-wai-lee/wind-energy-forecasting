# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "kagglehub>=0.3.13",
#     "marimo>=0.17.0",
#     "matplotlib>=3.10.8",
#     "mlflow>=3.8.0",
#     "numpy>=2.4.0",
#     "optuna>=4.6.0",
#     "pandas>=2.3.3",
#     "pyzmq>=27.1.0",
#     "scikit-learn>=1.8.0",
#     "seaborn>=0.13.2",
#     "statsmodels>=0.14.6",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Objective and Motivation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objective
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview of Wind Turbine Anatomy
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wind Turbine Aerodynamics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Code
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pip Install, Init, and Start Pyspark
    """)
    return


@app.cell
def _():
    # from pyspark.sql import SparkSession
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    # from pyspark.sql import functions as F
    # from pyspark.sql import types as T
    # from pyspark.sql.functions import broadcast
    # from pyspark.sql.functions import col
    # from pyspark.sql.functions import *
    # '%matplotlib inline' command supported automatically in marimo
    import seaborn as sns
    return Path, os, pd, plt


@app.cell
def _():
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, f1_score, mean_absolute_percentage_error
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.tools import diff
    import sklearn
    from sklearn.model_selection import train_test_split
    return (
        SARIMAX,
        adfuller,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        plot_acf,
        plot_pacf,
        root_mean_squared_error,
    )


@app.cell
def _():
    import mlflow
    import mlflow.data
    import optuna
    import kagglehub
    import marimo as mo
    return kagglehub, mlflow, mo, optuna


@app.cell
def _(Path):
    Path.cwd().joinpath("mlruns").as_uri()
    return


@app.cell
def _():
    # from ydata_profiling import ProfileReport
    return


@app.cell
def _():
    # spark = SparkSession.builder.master("local[*]").getOrCreate()
    # sc = spark.sparkContext
    return


@app.cell
def _():
    # Get the number of cores
    # num_cores = sc.defaultParallelism

    # Get the number of executors
    # num_executors = sc.getConf().get("spark.executor.instances")

    # print(num_cores)
    # print(num_executors)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Import Data
    """)
    return


@app.cell
def _(os):
    print(os.getcwd())
    return


@app.cell
def _(kagglehub):

    # Download latest version
    path = kagglehub.dataset_download("theforcecoder/wind-power-forecasting")

    print("Path to dataset files:", path)
    return (path,)


@app.cell
def _(path):
    print(path)
    return


@app.cell
def _(Path, path, pd):
    # df = pd.read_csv("../src/main/resources/data/Turbine_Data.csv")
    df = pd.read_csv(Path(path)/'Turbine_Data.csv')
    return (df,)


@app.cell
def _():
    # df = (spark.read
    #       .option("header", "true")
    #       .option("inferSchema", "true")
    #       .csv("../data/Turbine_Data.csv"))
    return


@app.cell
def _():
    # df.printSchema()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Transform Data
    """)
    return


@app.cell
def _():
    # df_renamed = df.withColumnRenamed("_c0", "datetime_stamp")
    return


@app.cell
def _(df):
    df_renamed = df.rename({"Unnamed: 0": "datetime_stamp"}, axis=1)
    return (df_renamed,)


@app.cell
def _(df_renamed):
    df_renamed.columns
    return


@app.cell
def _(df_renamed, pd):
    df_renamed['datetime_stamp'] = pd.to_datetime(df_renamed['datetime_stamp'])
    return


@app.cell
def _(df_renamed):
    df_indexed_sorted = df_renamed.set_index("datetime_stamp").sort_index()
    return (df_indexed_sorted,)


@app.cell
def _(df_indexed_sorted):
    df_indexed_sorted.dtypes
    return


@app.cell
def _(df_indexed_sorted):
    pdf = df_indexed_sorted.copy()
    return (pdf,)


@app.cell
def _():
    # df_casted = df_renamed.withColumn("datetime_stamp", col("datetime_stamp").cast("timestamp")).drop(col("_c0"))
    # df_casted.printSchema()
    return


@app.cell
def _():
    # df_filtered = df_casted.where(
    #     to_date(col("datetime_stamp")) =="2018-01-01"
    #     )
    return


@app.cell
def _():
    # pdf = df_casted.toPandas()
    return


@app.cell
def _():
    # profile = ProfileReport(pdf, title="YData Profiling Report")
    # profile.to_file('profiling_report.html');
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Drop Empty or Single Value Cols
    """)
    return


@app.cell
def _(pdf):
    pdf_drop_bad_cols = pdf.drop(columns=['ControlBoxTemperature', 'TurbineStatus', 'WTG'], axis=1)
    return (pdf_drop_bad_cols,)


@app.cell
def _(pdf_drop_bad_cols):
    type(pdf_drop_bad_cols)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Drop Highly Correlated Columns
    """)
    return


@app.cell
def _(pdf_drop_bad_cols):
    corr_matrix = pdf_drop_bad_cols.corr()
    corr_matrix
    return (corr_matrix,)


@app.cell
def _(corr_matrix):
    corr_matrix.abs().unstack()["NacellePosition"]
    return


@app.cell
def _(corr_matrix):
    corr_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates().head(20)
    return


@app.cell
def _(pdf):
    ### Drop highly correlated features
    ### pdf[["WindSpeed","ActivePower"]]
    ### pdf[["RotorRPM", "GeneratorRPM"]]
    ### pdf[["AmbientTemperatue", "MainBoxTemperature"]]
    ### pdf[["Blade1PitchAngle","Blade2PitchAngle", "Blade3PitchAngle"]]
    ### pdf[["NacellePosition","WindDirection"]]


    df_drop_corr_cols = pdf.drop(["NacellePosition","RotorRPM", "AmbientTemperatue", "Blade2PitchAngle", "Blade3PitchAngle"], axis=1)
    return (df_drop_corr_cols,)


@app.cell
def _(df_drop_corr_cols):
    df_drop_corr_cols
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Drop Null Rows
    """)
    return


@app.cell
def _(df_drop_corr_cols):
    df_drop_nulls = df_drop_corr_cols.dropna(subset=["ActivePower"]).sort_index()
    # df_drop_nulls = df_drop_corr_cols.sort_index()
    return (df_drop_nulls,)


@app.cell
def _(df_drop_nulls, pdf):
    ### Percent non-null
    df_drop_nulls.shape[0]/pdf.shape[0]
    return


@app.cell
def _(df_drop_nulls):
    df_drop_nulls.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train/Test Split
    """)
    return


@app.cell
def _(df_drop_nulls, pd):
    def train_test_split_by_date(df: pd.DataFrame, train_test_date_split_mid_pt: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train = df.iloc[df.index <= train_test_date_split_mid_pt]
        X_test = df.iloc[df.index > train_test_date_split_mid_pt]
        # print(X_train)
        # print(X_test)
        return X_train, X_test

    train_test_date_split_mid_pt = "2019-03-01"

    X_train_time_series, X_test_time_series = train_test_split_by_date(df_drop_nulls, train_test_date_split_mid_pt)
    return X_test_time_series, X_train_time_series


@app.cell
def _(os):
    try:
        os.mkdir('./figures')
    except FileExistsError:
        print("FileExistsError: [Errno 17] File exists: './figures'")
    return


@app.cell
def _(X_test_time_series, X_train_time_series, plt):
    ### Plot x_train and x_test
    y_var = "ActivePower"

    plt.figure(figsize=(10, 6))
    plt.plot(X_train_time_series[y_var])
    plt.plot(X_test_time_series[y_var])
    plt.title("Active Power Generated")
    plt.xlabel("Date")
    plt.ylabel("ActivePower")
    time_series_fn = f"./figures/time_series.png"
    plt.savefig(time_series_fn)
    plt.close()
    # plt.show()
    return (y_var,)


@app.cell
def _(X_test_time_series, X_train_time_series, pd, y_var):
    def resample_mean(X_train: pd.DataFrame, X_test: pd.DataFrame, y_variable: str, grain: str) -> tuple[pd.Series, pd.Series]:
        return X_train[y_variable].resample(grain).mean(), X_test[y_variable].resample(grain).mean()

    X_train_resampled, X_test_resampled = resample_mean(X_train_time_series, X_test_time_series, y_variable=y_var, grain="W-MON")
    return X_test_resampled, X_train_resampled


@app.cell
def _(X_test_resampled, X_train_resampled, plt):
    ### Plot x_train and x_test
    plt.figure(figsize=(10, 6))
    plt.plot(X_train_resampled)
    plt.plot(X_test_resampled)
    plt.title("Active Power Generated")
    plt.xlabel("Date")
    plt.ylabel("ActivePower")
    time_series_resampled_fn = f"./figures/time_series_resampled.png"
    plt.savefig(time_series_resampled_fn)
    plt.close()
    # plt.show()
    return


@app.cell
def _():
    # def _apply_differencing(self, time_series: pd.Series):
        # return diff(time_series, k_seasonal_diff=1, seasonal_periods=self.seasonal_periods)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit and Forecast (SARIMA)
    """)
    return


@app.cell
def _(mlflow):
    mlflow.set_experiment('wind-energy-forecasting')
    # Enable autologging for scikit-learn
    mlflow.statsmodels.autolog()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation Functions
    """)
    return


@app.cell
def _(adfuller, pd):
    def check_stationarity(self, time_series: pd.Series) -> None:
        result = adfuller(time_series, autolag='AIC')
        p_value = result[1]
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {p_value}')
        print('Stationary' if p_value < 0.05 else 'Non-Stationary')
        return None
    return


@app.cell
def _(
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    pd,
    root_mean_squared_error,
):
    def evaluate_forecast(X_test: pd.Series, forecast) -> tuple[float, float, float, float]:
        # print(X_test[1:].sort_index())
        # print(forecast.sort_index())

        mae = mean_absolute_error(X_test[1:], forecast)
        mape = mean_absolute_percentage_error(X_test[1:], forecast)
        mse = mean_squared_error(X_test[1:], forecast)
        rmse = root_mean_squared_error(X_test[1:], forecast)

        print(f'MAE: {mae}')
        print(f'MAE: {mape}')
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')

        return mae, mape, mse, rmse
    return (evaluate_forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optuna and MLFlow
    """)
    return


@app.cell
def _(
    SARIMAX,
    X_test_resampled,
    X_test_time_series,
    X_train_resampled,
    X_train_time_series,
    evaluate_forecast,
    mlflow,
    pd,
    plot_acf,
    plot_pacf,
    plt,
):
    def objective(trial):
        with mlflow.start_run(nested=True, run_name=f'trial_{trial.number}') as child_run:  # Setting nested=True will create a child run under the parent run.
            max_p = trial.suggest_int('max_p', 1, 3)
            max_d = trial.suggest_int('max_d', 1, 3)
            max_q = trial.suggest_int('max_q', 1, 3)
            max_P = trial.suggest_int('max_P', 1, 3)
            max_D = trial.suggest_int('max_D', 1, 3)
            max_Q = trial.suggest_int('max_Q', 1, 3)
            seasonal_periods = 26
            forecast_n_steps = len(X_test_resampled) - 1
            X_train_dataset = mlflow.data.from_pandas(X_train_time_series)
            X_test_dataset = mlflow.data.from_pandas(X_test_time_series)
            X_train_resampled_dataset = mlflow.data.from_pandas(pd.DataFrame(X_train_resampled))
            X_test_resampled_dataset = mlflow.data.from_pandas(pd.DataFrame(X_test_time_series))
            params = {'max_p': max_p, 'max_d': max_d, 'max_q': max_q, 'max_P': max_P, 'max_D': max_D, 'max_Q': max_Q, 'seasonal_periods': seasonal_periods, 'random_state': 42}
            mlflow.log_params(params)
            mlflow.set_tag('Training Info', 'Time Series forecasting via SARIMA')
            mlflow.log_inputs(datasets=[X_train_dataset, X_test_dataset, X_train_resampled_dataset, X_test_resampled_dataset], contexts=['Training', 'Testing', 'Training Resampled', 'Testing Resampled'], tags_list=[None, {'my_tag': 'tag_value'}, None, None])
            plot_acf(X_train_resampled)
            acf_fn = f'./figures/acf_{trial.number}.png'
            plt.savefig(acf_fn)
            mlflow.log_artifact(acf_fn)
            plt.close()
            plot_pacf(X_train_resampled)
            pacf_fn = f'./figures/pacf_{trial.number}.png'
            plt.savefig(pacf_fn)
            mlflow.log_artifact(pacf_fn)
            plt.close()
            model = SARIMAX(X_train_resampled, order=(max_p, max_d, max_q), seasonal_order=(max_P, max_D, max_Q, seasonal_periods))
            fit_model = model.fit()  # Log current trial's parameters
            forecast = fit_model.get_forecast(steps=forecast_n_steps)
            forecast_values = forecast.predicted_mean
            forecast_ci = forecast.conf_int()  # Optional: Set a tag that we can use to remind ourselves what this run was for
            plt.figure(figsize=(10, 6))
            plt.plot(X_train_resampled, label='Training')
            plt.plot(X_test_resampled, label='Actuals', color='orange')
            plt.plot(forecast_values, label='Forecast', color='red')
            plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='k', alpha=0.15)
            plt.title('Active Power Forecast')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.legend()
            train_test_fcst_ci_fn = f'./figures/train_test_fcst_ci_{trial.number}.png'  ### Log ACF
            plt.savefig(train_test_fcst_ci_fn)
            mlflow.log_artifact(train_test_fcst_ci_fn)
            plt.close
            mae, mape, mse, rmse = evaluate_forecast(X_test_resampled, forecast_values)  # logging to mlflow      
            mlflow.log_metric('mae', mae)
            mlflow.log_metric('mape', mape)  # plt.show()
            mlflow.log_metric('mse', mse)
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metrics({'mae': mae, 'mape': mape, 'mse': mse, 'rmse': rmse})  ### Log PACF
            trial.set_user_attr('run_id', child_run.info.run_id)
            return fit_model.aic  # logging to mlflow  # plt.show()  ## Fit and forecast  ### Plot Training, Testing, Forecast, and CI  # logging to mlflow  # plt.show()  # ## Log plot_diagnostics()  # fit_model.plot_diagnostics()  # diagnostics_fn = f"./figures/diagnostics_{trial.number}.png"  # plt.savefig(diagnostics_fn)  # mlflow.log_artifact(diagnostics_fn) # logging to mlflow  # plt.close()  # # plt.show()  # Predict on the test set, compute and log the loss metric  # mlflow.log_metric("residuals", residuals)  # Log the model file  # mlflow.statsmodels.log_model(fit_model, name="model", remove_data=True)  # Make it easy to retrieve the best-performing child run later
    return (objective,)


@app.cell
def _(mlflow, run):
    print("Tracking URI: ", mlflow.get_tracking_uri())
    print("Run id:", run.info.run_id)
    print("Experiment:", run.info.experiment_id)
    return


@app.cell
def _(mlflow, objective, optuna):
    # Start an MLflow run)
    with mlflow.start_run(run_name='SARIMAX') as run:
        n_trials = 30
        mlflow.log_param('n_trials', n_trials)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params(study.best_trial.params)  # study.optimize(objective, callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,))])
        mlflow.log_metrics({'best_error': study.best_value})
        if (best_run_id := study.best_trial.user_attrs.get('run_id')):
            mlflow.log_param('best_child_run_id', best_run_id)  # Log the best trial and its run ID
    return (run,)


@app.cell
def _():
    ### TODO: add pacf and acf for resampled
    ### TODO: Add next forecasating
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sources
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    https://www.geeksforgeeks.org/machine-learning/sarima-seasonal-autoregressive-integrated-moving-average/
    """)
    return


@app.cell
def _():
    ### Apply Differencecing
    # print("Apply Differencing")
    # X_train_resampled_differenced = pd.Series(self._apply_differencing(X_train_resampled))
    # X_test_resampled_differenced = pd.Series(self._apply_differencing(X_test_resampled))

    ### Plot X_train with differencing and resampling
    # print("Plotting X_train with Differencing and resampling")
    # weekly_forecast._plot_time_series(X_train_resampled, X_test_resampled)
    # weekly_forecast._check_stationarity(X_train_resampled)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
