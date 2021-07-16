import io
import json
import math
import os
import sys
from base64 import b64decode
from datetime import date, timedelta
from typing import Any, Tuple

import boto3
import numpy as np
import pandas as pd
import snowflake.connector
from slack_sdk.webhook import WebhookClient
from statsmodels.tsa.arima.model import ARIMA


def lambda_handler(event, context):
    series = fetch_messaging_count()

    best_model = evaluate_models(series['MESSAGE_COUNT'].astype(float).tolist(), [2, 4, 6, 8, 10], range(0, 4),
                                 range(0, 3))

    return output_results(best_model)


def fetch_messaging_count():
    kms_client = boto3.client('kms')
    username = kms_client.decrypt(
        CiphertextBlob=b64decode(os.environ['SNOWFLAKE_USERNAME']),
        EncryptionContext={'LambdaFunctionName':
                               os.environ['AWS_LAMBDA_FUNCTION_NAME']}
    )['Plaintext'].decode('utf-8')
    password = kms_client.decrypt(
        CiphertextBlob=b64decode(os.environ['SNOWFLAKE_PASSWORD']),
        EncryptionContext={'LambdaFunctionName':
                               os.environ['AWS_LAMBDA_FUNCTION_NAME']}
    )['Plaintext'].decode('utf-8')

    # data
    ctx = snowflake.connector.connect(
        user=username,
        password=password,
        account='',
        warehouse='',
        database='',
        schema=''
    )

    cur = ctx.cursor()

    month_ago = date.today() - timedelta(60)
    two_months_ago = month_ago.strftime("%Y-%m-%d")

    cur.execute(f"SELECT MESSAGE_DATE, SUM(MESSAGE_COUNT) as MESSAGE_COUNT "
                f"FROM CUST.VW_CFA_MDR_HISTORICAL_COUNT "
                f"WHERE MESSAGE_DATE >= '{two_months_ago}' "
                f"GROUP BY MESSAGE_DATE "
                f"ORDER BY MESSAGE_DATE")

    series = cur.fetch_pandas_all()
    series['MESSAGE_DATE'] = pd.to_datetime(series['MESSAGE_DATE'], infer_datetime_format=True)
    series.set_index('MESSAGE_DATE', inplace=True)
    series = series.asfreq('d')

    return series


def output_results(best_model):
    history, predictions, test, ci = best_model[0], best_model[1], best_model[2], best_model[3]

    # evaluate forecasts
    mse = np.square(np.subtract(test, predictions)).mean()
    rmse = math.sqrt(mse)

    create_prediction_interval_metrics(history[-1], ci)

    difference = (predictions[-1] - history[-1]) / history[-1] * 100

    slack_message = "Arima model results for Today's Messages:\n"
    slack_message += "predicted: %.0f\n" % predictions[-1]
    slack_message += "actual: %.0f\n" % history[-1]
    slack_message += "difference: %.2f%%\n" % difference
    slack_message += "prediction interval: (%.0f, %.0f)\n" % (ci[0, 0], ci[0, 1])
    slack_message += "Test RMSE: %.3f" % rmse
    print(slack_message)

    webhook = WebhookClient()
    response = webhook.send(text=slack_message)
    assert response.status_code == 200
    assert response.body == "ok"

    return {
        'statusCode': 200,
        'body': json.dumps('Finished Calculating ARIMA model for the last 60 days')
    }


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:len(X)]
    history = [x for x in train]
    # make predictions
    predictions = list()
    ci = None

    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        results = model_fit.get_forecast()
        yhat = results.predicted_mean[0]
        predictions.append(yhat)
        history.append(test[t])

        # create 95% prediction interval based on most recent prediction
        if t == len(test) - 1:
            ci = results.conf_int(alpha=0.05)

    # calculate out of sample error
    mse = np.square(np.subtract(test, predictions)).mean()
    rmse = math.sqrt(mse)

    return [rmse, history, predictions, test, ci]


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    best_run = [float("inf")]

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order: Tuple[Any, Any, Any] = (p, d, q)
                try:
                    results = evaluate_arima_model(dataset, order)
                    results.insert(1, order)
                    if results[0] < best_run[0]:
                        best_run = results
                    print('ARIMA%s RMSE=%.3f' % (results[1], results[0]))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_run[1], best_run[0]))

    return best_run[2:]


# create prediction interval and custom metrics in cloudwatch
def create_prediction_interval_metrics(actual, ci):
    prediction_interval_lower = ci[0, 0]
    prediction_interval_upper = ci[0, 1]

    cloudwatch = boto3.client('cloudwatch')
    cloudwatch.put_metric_data(
        MetricData=[
            {
                'MetricName': 'PREDICTION_INTERVAL_LOWER',
                'Unit': 'None',
                'Value': prediction_interval_lower
            },
            {
                'MetricName': 'PREDICTION_INTERVAL_UPPER',
                'Unit': 'None',
                'Value': prediction_interval_upper
            },
            {
                'MetricName': 'ACTUAL_MESSAGE_COUNT',
                'Unit': 'None',
                'Value': actual
            },
        ],
        Namespace='Insights/MessageCountArima'
    )
