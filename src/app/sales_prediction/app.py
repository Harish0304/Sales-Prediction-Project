import io
import base64
import pandas as pd
import numpy as np
import threading
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import matplotlib.dates as mdates
from flask import Response,send_file 
import xlsxwriter
from powerbiclient import models
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    
    file = request.files['file']
    prediction_type = request.form['predictionType']
    prediction_period = int(request.form['predictionPeriod'])

    # Read the uploaded data into a pandas dataframe
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file, encoding='ISO-8859-1')
        # Identify the sales column from the dataframe
        columns = ['DATE','ORDERDATE','date','orderdate']
        for col in df.columns:
            for col1 in columns:
                if col==col1:
                    df.rename(columns={col: 'Date'}, inplace=True)
                    print("col=",df.columns)
        # Convert the 'Date' column to datetime bcz column type string after converting a column name
        df['Date'] = pd.to_datetime(df['Date'])
        

        sales_columns = ['Sales_amount', 'Sales_amt', 'Sale_amt', 'Revenue', 'Turnover','SALES']
        for col in df.columns:
            for col1 in sales_columns:
                if col==col1:
                    df.rename(columns={col: 'sales'}, inplace=True)
                    print("col=",df.columns)
        
        print(df['sales'].head(10))
        
    elif file.filename.endswith('.xls') or file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df.rename(columns={col: 'Date'}, inplace=True)

        # Identify the sales column from the dataframe
        sales_columns = ['Sales_amount', 'Sales_amt', 'Sale_amt', 'Revenue', 'Turnover','SALES']
        for col in df.columns:
            for col1 in sales_columns:
                if col==col1:
                    df.rename(columns={col: 'sales'}, inplace=True)
            

    else:
        return jsonify({'error': 'Unsupported file format.'})
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df.rename(columns={col: 'Date'}, inplace=True)
    
    

    # Identify the highest year in the data
    highest_year = df['Date'].max().year
    #print('Highest year:', highest_year)

    highest_month = df['Date'].max().month
    #print('Highest month:', highest_month)
    
    # If the prediction type is "year," predict future sales by year using SARIMA
    if prediction_type == 'year':
        # Convert the date column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Set the date column as the index
        df.set_index('Date', inplace=True)

        # Resample the data by year and sum the sales
        df_yearly = df.resample('Y').sum()

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(df_yearly, shuffle=False, test_size=0.2)

        # Fit the SARIMA model to the training data
        sarima_model = SARIMAX(train_data['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_results = sarima_model.fit()

        # Predict the sales for the test period
        predictions = sarima_results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

        # Calculate the R-squared value
        r2 = r2_score(test_data['sales'], predictions)

        # Predict the sales for the next prediction_period years
        future_predictions = sarima_results.forecast(prediction_period)

        # Create a dataframe with the future predictions
        future_df = pd.DataFrame({'Date': pd.date_range(start=df_yearly.index[-1], periods=prediction_period + 1, freq='Y'),
                              'sales': [df_yearly['sales'][-1]] + list(future_predictions[:-1])})

        # Set the Date column as the index
        future_df.set_index('Date', inplace=True)

        # Concatenate the training and testing data with the future predictions
        full_data = pd.concat([train_data, test_data], axis=0)

        # Create a plot of the predicted sales
        fig, ax = plt.subplots()
        ax.plot(full_data.index, full_data['sales'], label='Actual')
        ax.plot(future_df.index, future_df['sales'], label='Predicted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sale Amount')
        ax.set_title('Sale Amount over Time')
        ax.legend()

        date_form = mdates.DateFormatter("%Y")
        ax.xaxis.set_major_formatter(date_form)
    
        # Save the plot to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        print(future_df.head())
        
        combined_df = pd.concat([df_yearly, future_df], axis=0)
        combined_df.index.name = 'Date'

        # Create a new column for the predicted sales
        combined_df['Predicted Sales'] = np.nan

        # Fill in the predicted sales for the future period
        combined_df.iloc[-prediction_period:, combined_df.columns.get_loc('Predicted Sales')] = future_predictions

        # Create a new Excel file with the combined data
        writer = pd.ExcelWriter('sale_prediction.xlsx', engine='xlsxwriter')
        combined_df.to_excel(writer, sheet_name='Sales')
        writer.save()

 
        # Send the Excel file to the frontend and download it automatically 
        send_file('sale_prediction.xlsx', as_attachment=True)

 
    # If the prediction type is "month," predict future sales by month using SARIMA
    elif prediction_type == 'month':
        # Convert the date column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Set the date column as the index
        df.set_index('Date', inplace=True)

        # Resample the data by month and sum the sales
        df_monthly = df.resample('M').sum() 

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(df_monthly, shuffle=False, test_size=0.2)

        # Fit the SARIMA model to the training data
        sarima_model = SARIMAX(train_data['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_results = sarima_model.fit()

        # Predict the sales for the test period
        predictions = sarima_results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

        # Calculate the R-squared value
        r2 = r2_score(test_data['sales'], predictions)

        # Predict the sales for the next prediction_period months
        future_predictions = sarima_results.forecast(prediction_period)

        # Create a dataframe with the future predictions
        future_df = pd.DataFrame({'Date': pd.date_range(start=df_monthly.index[-1], periods=prediction_period + 1, freq='M')[1:],
                              'sales': future_predictions})

        # Set the Date column as the index
        future_df.set_index('Date', inplace=True)

        # Concatenate the training and testing data with the future predictions
        full_data = pd.concat([train_data, test_data, future_df], axis=0)

        # Create a plot of the predicted sales
        fig, ax = plt.subplots()
        ax.plot(full_data.index, full_data['sales'], label='Actual')
        ax.plot(future_df.index, future_df['sales'], label='Predicted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sale Amount')
        ax.set_title('Sale Amount over Time')
        ax.legend()

        date_form = mdates.DateFormatter("%Y-%b")
        ax.xaxis.set_major_formatter(date_form)
        plt.xticks(rotation=45)

        # Save the plot to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

    elif prediction_type == 'week':
        # Convert the date column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Set the date column as the index
        df.set_index('Date', inplace=True)

        # Resample the data by week and sum the sales
        df_weekly = df.resample('W').sum()

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(df_weekly, shuffle=False, test_size=0.2)

        # Fit the SARIMA model to the training data
        sarima_model = SARIMAX(train_data['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
        sarima_results = sarima_model.fit()

        # Predict the sales for the test period
        predictions = sarima_results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

        # Calculate the R-squared value
        r2 = r2_score(test_data['sales'], predictions)

        # Predict the sales for the next prediction_period weeks
        future_predictions = sarima_results.forecast(prediction_period)

        # Create a dataframe with the future predictions
        future_df = pd.DataFrame({'Date': pd.date_range(start=df_weekly.index[-1], periods=prediction_period + 1, freq='W')[1:],
                              'sales': future_predictions})

        # Set the Date column as the index and set the frequency to 'W'
        future_df.set_index('Date', inplace=True)
        future_df.index.freq = 'W'

        # Concatenate the training and testing data with the future predictions
        full_data = pd.concat([train_data, test_data, future_df], axis=0)

        # Create a plot of the predicted sales
        fig, ax = plt.subplots()
        ax.plot(full_data.index, full_data['sales'], label='Actual')
        ax.plot(future_df.index, future_df['sales'], label='Predicted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sale Amount')
        ax.set_title('Sale Amount over Time')
        ax.legend()

        # Set the x-axis labels to show the year, week of year, and day of week
        date_form = mdates.DateFormatter("%Y-W%U-%a")
        ax.xaxis.set_major_formatter(date_form)
        
        # Save the plot to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)


    #Return the plot as an image response
    return Response(buf.getvalue(), mimetype='image/png')            

    

if __name__ == '__main__':
    app.run(debug=True)
