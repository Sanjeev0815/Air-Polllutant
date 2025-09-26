import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def cached_preprocess_data(data_hash, missing_method, resample_freq, create_lag_features, lag_periods, create_rolling_features, rolling_window, train_split):
    """Cached preprocessing function for better performance."""
    preprocessor = DataPreprocessor()
    return preprocessor._process_data_internal(data_hash, missing_method, resample_freq, create_lag_features, lag_periods, create_rolling_features, rolling_window, train_split)

class DataPreprocessor:
    """Handles data preprocessing for air pollutant forecasting models."""
    
    def __init__(self):
        self.scaler = None
        self.feature_names = None
        self.datetime_column = None
    
    def preprocess_data(self, data, missing_method='interpolate', resample_freq='1H',
                       create_lag_features=True, lag_periods=3,
                       create_rolling_features=True, rolling_window=6,
                       train_split=0.8):
        """
        Complete preprocessing pipeline for air pollutant data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw input data
        missing_method : str
            Method for handling missing values
        resample_freq : str
            Frequency for temporal resampling
        create_lag_features : bool
            Whether to create lag features
        lag_periods : int
            Number of lag periods to create
        create_rolling_features : bool
            Whether to create rolling average features
        rolling_window : int
            Window size for rolling averages
        train_split : float
            Proportion of data for training
        
        Returns:
        --------
        dict
            Preprocessed data split into train/test sets
        """
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Step 1: Handle datetime and set as index
        df = self._process_datetime(df, resample_freq)
        
        # Step 2: Handle missing values
        df = self._handle_missing_values(df, missing_method)
        
        # Step 3: Feature engineering
        df = self._create_features(df, create_lag_features, lag_periods,
                                  create_rolling_features, rolling_window)
        
        # Step 4: Remove any remaining NaN values (from lag/rolling features)
        df = df.dropna()
        
        if df.empty:
            raise ValueError("No data remaining after preprocessing. Try reducing lag periods or rolling window.")
        
        # Step 5: Prepare features and targets
        features, targets = self._prepare_features_targets(df)
        
        # Step 6: Scale features
        features_scaled = self._scale_features(features)
        
        # Step 7: Split data
        train_test_data = self._split_data(features_scaled, targets, train_split)
        
        # Add metadata
        train_test_data['feature_names'] = self.feature_names
        train_test_data['datetime_index'] = df.index
        train_test_data['scaler'] = self.scaler
        
        return train_test_data
    
    def _process_datetime(self, df, resample_freq):
        """Process datetime column and resample if needed."""
        if 'datetime' not in df.columns:
            raise ValueError("Datetime column not found in data")
        
        # Convert to datetime and set as index
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        
        # Sort by datetime
        df = df.sort_index()
        
        # Resample if frequency is different from current
        if resample_freq != 'original':
            # Aggregate numeric columns with mean
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_columns].resample(resample_freq).mean()
        
        return df
    
    def _handle_missing_values(self, df, method):
        """Handle missing values in the dataset."""
        if method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif method == 'interpolate':
            df = df.interpolate(method='time')
        elif method == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown missing value method: {method}")
        
        return df
    
    def _create_features(self, df, create_lag, lag_periods, create_rolling, rolling_window):
        """Create engineered features."""
        
        # Ensure we have the required columns for feature engineering
        pollutant_cols = [col for col in ['o3', 'no2'] if col in df.columns]
        met_cols = [col for col in ['temperature', 'wind_speed', 'humidity', 'solar_radiation', 'pressure'] 
                   if col in df.columns]
        
        feature_cols = pollutant_cols + met_cols
        
        if not feature_cols:
            raise ValueError("No suitable columns found for feature engineering")
        
        # Create lag features
        if create_lag and lag_periods > 0:
            for col in feature_cols:
                for lag in range(1, lag_periods + 1):
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Create rolling average features
        if create_rolling and rolling_window > 1:
            for col in feature_cols:
                df[f'{col}_rolling_{rolling_window}'] = df[col].rolling(
                    window=rolling_window, min_periods=1
                ).mean()
        
        # Create time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['season'] = df.index.month.map(self._get_season)
        
        # Create interaction features for meteorological variables
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        if 'wind_speed' in df.columns and 'temperature' in df.columns:
            df['wind_temp_interaction'] = df['wind_speed'] * df['temperature']
        
        return df
    
    def _get_season(self, month):
        """Map month to season."""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    
    def _prepare_features_targets(self, df):
        """Separate features and targets."""
        # Target columns
        target_cols = [col for col in ['o3', 'no2'] if col in df.columns]
        
        # Feature columns (everything except targets)
        feature_cols = [col for col in df.columns if col not in target_cols]
        
        if not feature_cols:
            raise ValueError("No feature columns available after preprocessing")
        
        if not target_cols:
            raise ValueError("No target columns (o3, no2) found in data")
        
        # Store feature names for later use
        self.feature_names = feature_cols
        
        # Return features and targets
        features = df[feature_cols]
        targets = df[target_cols]
        
        return features, targets
    
    def _scale_features(self, features):
        """Scale features using RobustScaler."""
        # Use RobustScaler as it's less sensitive to outliers
        self.scaler = RobustScaler()
        features_scaled = pd.DataFrame(
            self.scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        return features_scaled
    
    def _split_data(self, features, targets, train_split):
        """Split data into training and testing sets."""
        # Calculate split index (temporal split to maintain time series properties)
        split_idx = int(len(features) * train_split)
        
        # Split features
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        
        # Split targets
        y_train = targets.iloc[:split_idx]
        y_test = targets.iloc[split_idx:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def create_sequences(self, X, y, sequence_length=24):
        """
        Create sequences for LSTM/GRU models.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature data
        y : pd.DataFrame or np.array
            Target data
        sequence_length : int
            Length of input sequences
            
        Returns:
        --------
        tuple
            (X_sequences, y_sequences) for LSTM/GRU training
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X.iloc[i-sequence_length:i].values if hasattr(X, 'iloc') else X[i-sequence_length:i])
            y_sequences.append(y.iloc[i].values if hasattr(y, 'iloc') else y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def inverse_scale_predictions(self, predictions, target_columns):
        """
        Inverse scale predictions back to original scale.
        Note: This is a simplified version. In practice, you'd need to handle
        the scaling of targets separately from features.
        """
        # For now, return predictions as-is
        # In a full implementation, you'd maintain separate scalers for targets
        return predictions
