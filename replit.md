# Air Pollutant Forecasting Dashboard

## Overview

This is a Streamlit-based dashboard application for forecasting air pollutants (O₃ and NO₂) using satellite and reanalysis data. The application provides a complete machine learning pipeline including data upload, preprocessing, model training, and forecasting capabilities. It supports multiple forecasting models including Random Forest and deep learning models (LSTM/GRU when TensorFlow is available). The dashboard is designed for environmental scientists and researchers who need to predict short-term air pollution levels.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with multi-page navigation
- **UI Components**: Interactive dashboard with sidebar navigation, file upload widgets, data visualization charts, and metric displays
- **State Management**: Streamlit session state for maintaining application state across page interactions
- **Visualization**: Plotly for interactive charts and graphs with custom color schemes and responsive layouts

### Backend Architecture
- **Modular Design**: Object-oriented architecture with separate classes for different responsibilities
- **Core Components**:
  - `DataHandler`: Manages data loading and validation from CSV files
  - `DataPreprocessor`: Handles data cleaning, feature engineering, and train/test splitting
  - `PollutantForecaster`: Implements machine learning models for forecasting
  - `Visualizer`: Creates interactive plots and visualizations
- **Caching Strategy**: Streamlit's `@st.cache_data` decorator for performance optimization of data loading and preprocessing operations

### Data Processing Pipeline
- **Input Validation**: Flexible column naming with required columns (datetime, o3, no2, temperature, wind_speed, humidity) and optional columns
- **Preprocessing Features**:
  - Multiple missing value handling methods (interpolation, forward fill, backward fill)
  - Temporal resampling capabilities
  - Lag feature creation for time series analysis
  - Rolling window statistics generation
  - Data scaling using StandardScaler or RobustScaler
- **Train/Test Splitting**: Configurable split ratios with temporal ordering preservation

### Machine Learning Architecture
- **Model Types**: 
  - Random Forest Regressor (always available)
  - LSTM/GRU neural networks (TensorFlow-dependent with graceful fallback)
- **Target Variables**: Multi-target support for O₃ and NO₂ predictions
- **Feature Engineering**: Automated creation of lag features, rolling averages, and temporal features
- **Model Evaluation**: Comprehensive metrics including RMSE, MAE, and R² score

### Error Handling and Resilience
- **Graceful Degradation**: TensorFlow dependency is optional with fallback to simpler models
- **Input Validation**: Comprehensive data validation with user-friendly error messages
- **Warning Suppression**: Strategic filtering of non-critical warnings for cleaner user experience

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the dashboard interface
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Plotly**: Interactive data visualization (plotly.express and plotly.graph_objects)
- **Scikit-learn**: Machine learning algorithms and preprocessing tools

### Optional Dependencies
- **TensorFlow**: Deep learning framework for LSTM/GRU models (with graceful fallback if unavailable)
- **XArray**: Multi-dimensional array processing for satellite data handling

### Data Format Support
- **CSV Files**: Primary data input format with flexible column naming
- **Time Series Data**: Supports various datetime formats and temporal resampling

### Export Capabilities
- **CSV Export**: Prediction results export functionality
- **JSON Export**: Alternative data export format
- **Summary Reports**: Automated report generation for model performance