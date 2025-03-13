import numpy as np


class TaxiModel:
    def __init__(self, model):
        self.model = model
        self.features = ['abnormal_period', 'hour', 'weekday', 'month']

    def __preprocess(self, X):
        X = X.copy()
        X['weekday'] = X['pickup_datetime'].dt.weekday
        X['month'] = X['pickup_datetime'].dt.month
        X['hour'] = X['pickup_datetime'].dt.hour
        
        # Calculer les dates anormales (moins de 6300 trajets)
        df_counts = X['pickup_datetime'].dt.date.value_counts()
        abnormal_dates = df_counts[df_counts < 6300]
        X['abnormal_period'] = X['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
        
        return X[self.features]

    def __postprocess(self, raw_output):
        # your postprocessing logic: inverse transformation, etc.
        return np.expm1(raw_output)

    def fit(self, X, y):
        X_processed = self.__preprocess(X)
        y_transformed = np.log1p(y)
        self.model.fit(X_processed, y_transformed)
        return self

    def predict(self, X):
        X_processed = self.__preprocess(X)
        raw_output = self.model.predict(X_processed)
        return self.__postprocess(raw_output)