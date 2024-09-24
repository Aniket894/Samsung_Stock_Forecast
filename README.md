

# Samsung Stock Forecast

This documentation outlines the methodology, models, and forecasting process used in the four stock forecasting files related to Samsung's stock prices, focusing on forecasting the future stock prices based on historical data from 2023-06-01 to 2024-06-13. Each of the files utilizes different forecasting techniques, including statistical models like ARIMA/SARIMA, machine learning models, and deep learning models such as LSTM.

### Detailed Documentation for Samsung Stock Forecasting Files  
**Forecasting period**: 2023-06-01 to 2024-06-13  
**Files**:  
1. `Samsung_Stock_forecast_(arima__and_sarima).ipynb`  
2. `Samsung_Stock_forecastin_with_ML_Models.ipynb`  
3. `Samsung_Univariate_Stock_forecast_with_LSTM.ipynb`  
4. `Samsung_Multivariate_Stock_forecast_with_LSTM.ipynb`  

---

### 1. **Univariate Time Series Forecasting Using ARIMA & SARIMA (`Samsung_Stock_forecast_(arima__and_sarima).ipynb`)**  

This file uses **ARIMA** and **SARIMA** to forecast Samsung's closing stock prices. These models are ideal for univariate time series data and are well-suited for capturing temporal patterns in historical price trends.

#### **Objective**:
- Forecast the next 30 and 90 days of Samsung's stock prices using ARIMA and SARIMA based on historical closing prices.

#### **Dataset**:
- **'Close' column**: Represents Samsung's daily closing stock prices.

#### **Process**:

1. **Data Preprocessing**:
   - **Loading data**: Load historical data from the start date (2023-06-01) to the end date (2024-06-13).
   - **Handling missing values**: Address any missing data points (e.g., through interpolation).
   - **Stationarity check**: Use **Augmented Dickey-Fuller (ADF) test** to check if the time series is stationary. Non-stationary data is differenced until stationarity is achieved.

2. **ARIMA Model**:
   - **Model structure**: ARIMA is a combination of three components: Auto-regressive (AR), Integrated (I), and Moving Average (MA). It forecasts future points using a linear combination of past observations and error terms.
   - **Model configuration**: Optimal values for p (AR terms), d (differencing), and q (MA terms) are selected using ACF and PACF plots, followed by grid search on different ARIMA configurations.
   - **Model fitting**: Train ARIMA on historical data.
   - **Forecasting**:
     - **Next 30 days**: Forecast future prices for the next 30 days.
    
     - ![download - 2024-09-24T081220 390](https://github.com/user-attachments/assets/9fdb3ba3-910a-402f-8946-4740f973d16a)
       

     - **Next 90 days**: Forecast future prices for the next 90 days.
     
     - ![download - 2024-09-24T081820 023](https://github.com/user-attachments/assets/76c49d69-d27f-4ccf-adbb-d141ba67b629)

   - **Evaluation**: Plot forecasted values against historical data for visual inspection.

3. **SARIMA Model**:
   - **Model structure**: SARIMA adds a seasonal component (P, D, Q) to the ARIMA model to handle periodic trends in the data.
   - **Model configuration**: Use seasonal decomposition to identify seasonal patterns. Grid search is applied to optimize parameters (p, d, q) for both seasonal and non-seasonal components.
   - **Model fitting**: Train SARIMA on the historical 'close' data.
   - **Forecasting**:
     - **Next 30 days**: Forecast future prices for the next 30 days.
    
     - ![download - 2024-09-24T081414 754](https://github.com/user-attachments/assets/044250a7-32b4-4a95-b8f0-7d51e367bb5c)
       

     - **Next 90 days**: Forecast future prices for the next 90 days.
    
     - ![download - 2024-09-24T081837 922](https://github.com/user-attachments/assets/44d65e21-c976-4546-9d6d-1874281d13c4)

   - **Evaluation**: Compare SARIMA's forecast accuracy against ARIMA using evaluation metrics (e.g., AIC, RMSE).

4. **Visualization**:
   - Visualize ARIMA and SARIMA forecasts for both 30-day and 90-day windows against actual closing prices to assess the models' fit and forecast precision.

#### **Conclusion**:
- **ARIMA** is effective for data without strong seasonal components, while **SARIMA** captures seasonality, offering potentially better accuracy if there are recurring patterns. Both models are benchmarked in terms of their predictive power and error rates.

---

### 2. **Univariate Stock Forecasting with Machine Learning Models (`Samsung_Stock_forecastin_with_ML_Models.ipynb`)**  

This file applies machine learning models to forecast Samsung’s stock prices using the **'close'** column as the target variable. Unlike ARIMA/SARIMA, machine learning models do not rely on stationarity assumptions, allowing for more flexibility in capturing complex relationships.

#### **Objective**:
- Forecast the next 30 and 90 days of stock prices using multiple machine learning models.

#### **Models Used**:
- **Linear Regression**
- **Random Forest Regressor**
- **Support Vector Regression (SVR)**
- **XGBoost Regressor**

#### **Process**:

1. **Data Preprocessing**:
   - **Normalization**: Normalize the 'close' column to make the data suitable for machine learning algorithms.
   - **Sliding window technique**: Convert the time series data into a supervised learning problem by using a sliding window approach where past n days' prices are used to predict the next day’s price.
   - **Train-test split**: Split the data into training (e.g., 80%) and testing sets (20%).

2. **Model Training**:
   - **Linear Regression**:
     - Simple model to establish a baseline. It assumes a linear relationship between past stock prices and future values.
   - **Random Forest Regressor**:
     - Ensemble model based on decision trees. It captures non-linearities and interactions between data points.
   - **Support Vector Regression (SVR)**:
     - SVR seeks to find a hyperplane that best fits the data with a margin of tolerance, making it robust to noise.
   - **XGBoost Regressor**:
     - Gradient boosting model that builds trees sequentially, optimizing the learning process by correcting previous model errors.

3. **Forecasting**:

   - *Linear Regression*
     
   - **Next 30 days**: Each model predicts closing prices for the next 30 days.
     
  
   - ![download - 2024-09-24T084545 166](https://github.com/user-attachments/assets/65ce8838-9b65-40bb-ab12-ffac126ad1d3)
  
   - ![download - 2024-09-24T085054 916](https://github.com/user-attachments/assets/6e134c5a-0e88-45cc-a463-bb8aae54c14b)



   - **Next 90 days**: Each model predicts closing prices for the next 90 days.
  
     
   -  ![download - 2024-09-24T085111 871](https://github.com/user-attachments/assets/31011ea6-22f0-4984-a27e-5da8239fccc8)

   - ![download - 2024-09-24T085124 501](https://github.com/user-attachments/assets/90fd8ff5-9ddd-44a7-80d9-3a987b0f94ad)
  
     
  
   - **Random Forest Regressor**
  
   - **Next 30 days**: Each model predicts closing prices for the next 30 days.
  
   - 
   - ![download - 2024-09-24T085731 593](https://github.com/user-attachments/assets/4c4047ac-fd64-49d2-b7ed-be408c18d37c)
   - ![download - 2024-09-24T090005 167](https://github.com/user-attachments/assets/5c2b8bdb-563e-4a2d-bd5f-42e9e63d54c3)



   - **Next 90 days**: Each model predicts closing prices for the next 90 days.
   - 

   - ![download - 2024-09-24T085810 889](https://github.com/user-attachments/assets/10207820-7e9a-431b-992d-fcd43f15279f)
     
   - ![download - 2024-09-24T085836 695](https://github.com/user-attachments/assets/96db6794-a3e5-4503-961a-7672493fdb20)



  - **Support Vector Regression (SVR)**
    
   - **Next 30 days**: Each model predicts closing prices for the next 30 days.
   
   - ![download - 2024-09-24T090423 866](https://github.com/user-attachments/assets/2f47c997-c8c6-4f31-8c34-566a7a747834)

   - ![download - 2024-09-24T090503 891](https://github.com/user-attachments/assets/2addc46c-0edd-43f1-aa75-59acf40f51d3)

   - 
   - **Next 90 days**: Each model predicts closing prices for the next 90 days.
   - 

   - ![download - 2024-09-24T090546 568](https://github.com/user-attachments/assets/fb51f60b-0ea4-4e60-ae97-f8ba48058ac1)
   - 
   - ![download - 2024-09-24T090618 275](https://github.com/user-attachments/assets/32624e1e-b68c-43a4-93aa-388825f1b8c9)


  - **XGBoost Regressor**

  
   - **Next 30 days**: Each model predicts closing prices for the next 30 days.

   - ![download - 2024-09-24T091001 307](https://github.com/user-attachments/assets/f587782f-5eec-47e8-ade6-709ec8637252)

   - ![download - 2024-09-24T091014 456](https://github.com/user-attachments/assets/383cca25-f59b-4639-9c03-7551615d976f)


   - **Next 90 days**: Each model predicts closing prices for the next 90 days.

   - ![download - 2024-09-24T091031 167](https://github.com/user-attachments/assets/2a9287c0-fba0-4353-9b85-c7a49b3a2f40)

  - ![download - 2024-09-24T091047 337](https://github.com/user-attachments/assets/8b0e6d8a-b9f9-4f28-9b26-6e24d0d300d2)



5. **Model Evaluation**:
   - **Metrics**: Evaluate models using RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R-squared. These metrics provide insights into how closely the predicted values match the actual stock prices.
   - **Comparison**: Compare the performance of each machine learning model to assess which one provides the most accurate forecasts.

6. **Visualization**:
   - Visualize the predicted future prices for both 30- and 90-day forecasts from all models.
   - Compare their performance against actual data.

#### **Conclusion**:
- The performance of each machine learning model varies based on its ability to capture non-linear relationships. **XGBoost** and **Random Forest** tend to outperform **Linear Regression**, as they handle non-linear trends better. **SVR** can perform well with small datasets but may struggle with long-term forecasting.

---

### 3. **Univariate Stock Forecasting with LSTM (`Samsung_Univariate_Stock_forecast_with_LSTM.ipynb`)**  

This file uses **Long Short-Term Memory (LSTM)** networks, a type of recurrent neural network (RNN), to forecast Samsung’s stock prices using the **'close'** column.

#### **Objective**:
- Forecast the next 30 and 90 days of stock prices using an LSTM model.

#### **LSTM Model**:
- **LSTM architecture**: LSTM networks are particularly effective at learning from time series data with long-term dependencies, as they maintain memory through internal cell states.

#### **Process**:

1. **Data Preprocessing**:
   - **Scaling**: Scale the 'close' column using MinMaxScaler to ensure that the data is normalized within a specific range, which helps in improving model training.
   - **Sequence creation**: Create time steps by converting the data into sequences, where past n days' prices are used to predict the next day's price.
   - **Train-test split**: Divide the data into training and testing sets.

2. **Model Training**:
   - **Network configuration**: Define the LSTM architecture with input layers, LSTM layers, and dense layers to output predictions.
   - **Training**: The LSTM network is trained on the preprocessed data using historical prices.
   - **Hyperparameter tuning**: Adjust model parameters such as the number of LSTM units, epochs, and batch size to improve the model's accuracy.

3. **Forecasting**:
   - **Next 30 days**: Forecast the next 30 days using the trained LSTM model.
  
   - ![download - 2024-09-24T082452 547](https://github.com/user-attachments/assets/d9420381-5340-4993-9fd5-793a0ee2c91c)
  
   - ![download - 2024-09-24T082557 693](https://github.com/user-attachments/assets/fe9c63e3-656a-4537-8433-37e3d2dc3fb5)


   - **Next 90 days**: Forecast the next 90 days using the LSTM model.
  
   - ![download - 2024-09-24T082628 150](https://github.com/user-attachments/assets/0e51bb61-5bc7-47a6-b9d9-671c037e9b4a)
  
   - ![download - 2024-09-24T082705 293](https://github.com/user-attachments/assets/560c9286-4a8d-4969-8a61-380d840e0fd3)



4. **Evaluation**:
   - **Metrics**: Evaluate model performance using MAE, RMSE, and accuracy metrics.
   - **Overfitting check**: Check for overfitting by comparing training and test set performance.

5. **Visualization**:
   - Plot the predicted 30- and 90-day forecasts and compare them with actual historical prices to assess how well the LSTM model performs.

#### **Conclusion**:
- LSTM performs well for univariate time series forecasting due to its ability to capture temporal dependencies and trends. However, it requires careful tuning of hyperparameters to avoid overfitting or underfitting.

---

### 4. **Multivariate Stock Forecasting with LSTM (`Samsung_Multivariate_Stock_forecast_with_LSTM.ipynb`)**  

This file expands the forecasting process to a **multivariate LSTM** model by incorporating multiple features (e.g., open, high, low, volume) to predict Samsung’s closing prices.

#### **Objective**:
- Forecast the next 30 and 90 days of stock prices using a multivariate LSTM model, leveraging multiple input features.

#### **Dataset**:
- **Multiple columns**: In addition to the 'close' column, other features like 'open', 'high', 'low', and 'volume

' are used as input variables for the LSTM model.

#### **Process**:

1. **Data Preprocessing**:
   - **Feature scaling**: Use MinMaxScaler to scale all the input features (open, high, low, volume).
   - **Sequence creation**: Create multivariate sequences by using past values of all features to predict the next day’s 'close' value.
   - **Train-test split**: Split the data into training and test sets.

2. **Model Training**:
   - **LSTM architecture**: Set up an LSTM model that accepts multiple features as inputs and outputs a prediction for the closing price.
   - **Training**: Train the LSTM model on the multivariate dataset.
   - **Hyperparameter tuning**: Adjust the number of LSTM units, learning rate, batch size, and epochs to find the best model.

3. **Forecasting**:
   - **Next 30 days**: Use the trained multivariate LSTM model to predict the next 30 days of stock prices.
  
   - ![download - 2024-09-24T083401 981](https://github.com/user-attachments/assets/f1745dc2-b816-4035-9e8e-d6e41f2d5d99)
  
   - ![download - 2024-09-24T083536 681](https://github.com/user-attachments/assets/f36eebef-f8cc-46d0-8bd0-0ad37bdc1255)



   - **Next 90 days**: Use the model to forecast 90 days into the future.

     
   - ![download - 2024-09-24T083623 906](https://github.com/user-attachments/assets/dc17e092-f914-4170-8876-0a728740d397)
  
   - ![download - 2024-09-24T083613 282](https://github.com/user-attachments/assets/8b28fc25-e470-4910-869f-7a5c7f549a71)



4. **Evaluation**:
   - **Metrics**: Assess model performance using RMSE, MAE, and accuracy scores.
   - **Feature importance**: Evaluate which features contribute the most to the LSTM’s predictions.

5. **Visualization**:
   - Visualize the predicted stock prices for the 30- and 90-day windows using the multivariate LSTM model.

#### **Conclusion**:
- The **multivariate LSTM** model can capture interactions between multiple variables, potentially improving the forecast accuracy over univariate models. However, multivariate models can be more complex and require more training data to perform well.

---

## Conclusion

Each of the four files focuses on different forecasting methods for predicting Samsung's stock prices, leveraging statistical models (ARIMA/SARIMA), machine learning models (Linear Regression, Random Forest, SVR, XGBoost), and deep learning models (univariate and multivariate LSTM). The forecasted results are plotted and analyzed for the next 30 and 90 days to offer a comprehensive view of how different models perform under varying conditions.

This documentation provides a structured overview of how each approach is applied, allowing for easy comparison of methods and their results.
