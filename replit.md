# Air Pollutant Forecasting Dashboard

## Overview

This is a comprehensive Streamlit web application designed for short-term forecasting of gaseous air pollutants (O₃ and NO₂) using satellite and reanalysis data. The system provides an end-to-end solution for environmental monitoring and prediction, featuring data preprocessing, machine learning model training, and interactive visualization capabilities. The application supports multi-step workflows including data upload, preprocessing, model training with LSTM/GRU and Random Forest algorithms, and real-time forecasting with safety threshold monitoring.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit with wide layout configuration
- **Navigation**: Radio button-based page navigation system with four main sections
- **State Management**: Session state management for data loading, model training, and prediction status
- **Visualization**: Plotly-based interactive charts and graphs for time series analysis
- **UI Components**: Modular design with separate pages for different workflow stages

### Backend Architecture
- **Data Processing Pipeline**: Modular architecture with separate classes for each major function
  - `DataHandler`: Manages CSV data loading and validation with flexible column naming
  - `DataPreprocessor`: Handles missing values, feature engineering, and temporal resampling
  - `PollutantForecaster`: Implements LSTM/GRU and Random Forest models for time series forecasting
  - `Visualizer`: Creates interactive plots and visualizations using Plotly

### Machine Learning Components
- **Time Series Models**: LSTM and GRU neural networks using TensorFlow/Keras
- **Ensemble Models**: Random Forest regression using scikit-learn
- **Hybrid Approach**: Combines deep learning temporal patterns with ensemble meteorological feature regression
- **Feature Engineering**: Automated creation of lag variables, rolling averages, and seasonal indicators
- **Model Evaluation**: RMSE, MAE, and R² metrics with baseline persistence model comparison

### Data Processing Architecture
- **Input Flexibility**: Supports CSV format with flexible column naming conventions
- **Temporal Handling**: Automatic datetime parsing and resampling capabilities
- **Missing Data**: Multiple strategies including interpolation and forward filling
- **Scaling**: StandardScaler and RobustScaler options for feature normalization
- **Sequence Creation**: Automatic time series sequence generation for LSTM/GRU models

### Safety and Monitoring
- **Threshold Detection**: Configurable safety thresholds for pollutant concentrations
- **Alert System**: Visual alerts when forecasted pollutants exceed safe limits
- **Performance Caching**: Streamlit caching for improved data loading performance

## External Dependencies

### Core Python Libraries
- **streamlit**: Web application framework for dashboard interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing operations
- **plotly.express** and **plotly.graph_objects**: Interactive visualization and charting
- **xarray**: Multi-dimensional data handling for satellite/reanalysis data

### Machine Learning Libraries
- **tensorflow** and **tensorflow.keras**: Deep learning framework for LSTM/GRU models
- **scikit-learn**: Machine learning algorithms including Random Forest and preprocessing tools
- **sklearn.ensemble.RandomForestRegressor**: Tree-based ensemble learning
- **sklearn.preprocessing**: Data scaling and normalization utilities
- **sklearn.model_selection**: Train-test splitting functionality
- **sklearn.metrics**: Model evaluation metrics (MSE, MAE, R²)

### Data Processing Libraries
- **warnings**: Error and warning management
- **datetime** and **timedelta**: Date and time handling utilities
- **io.StringIO** and **io.BytesIO**: In-memory file handling for uploads

### Potential Data Sources
- **Satellite Data**: Sentinel-5P/TROPOMI, MODIS for O₃ and NO₂ concentrations
- **Reanalysis Data**: ERA5, MERRA-2 for meteorological variables
- **Ground Monitoring**: Optional calibration data from air quality monitoring stations
- **Meteorological Variables**: Temperature, wind speed, humidity, solar radiation, pressure

### File Format Support
- **CSV**: Primary data input format with flexible column structure
- **NetCDF**: Potential future support for gridded satellite/reanalysis data through xarray