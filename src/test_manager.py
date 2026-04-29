# ./src/test_manager.py

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.metrics_manager import Metrics


class TestManager:
    
    def __init__(self, id_test: list) -> None:
        
        X_test, y_surv_test, y_count_test, groups_test = self._preparar_test(
            list_ids=id_test
        )
        
        self.X_test = X_test
        self.y_surv_test = y_surv_test
        self.y_count_test = y_count_test
        self.groups_test = groups_test
        
    @staticmethod
    def _preparar_test(list_ids: list) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        
        X_list = []
        y_survival_list = []
        y_count_list = []
        groups_list = []
        
        for idx in list_ids:
            df_motor = pd.read_csv(f'data/clean/data_motor_{idx}.csv')
            
            es_evento = bool(df_motor['evento'].iloc[0])
            max_tiempo_observado = df_motor['time_in_cycles'].max()
            tiempo_muerte_real = max_tiempo_observado + df_motor['RUL'].iloc[-1]
            
            X_motor = df_motor.drop(columns=['time_in_cycles', 'RUL', 'evento'])
            X_list.append(X_motor)
            
            for _, row in df_motor.iterrows():
                y_count_list.append(row['RUL'])
                
                if es_evento:
                    y_survival_list.append((True, tiempo_muerte_real))
                else:
                    y_survival_list.append((False, max_tiempo_observado))
                    
                groups_list.append(idx)
                
        X = pd.concat(X_list, ignore_index=True)
        y_surv = np.array(y_survival_list, dtype=[('evento', bool), ('tiempo', float)])
        y_count = np.array(y_count_list)
        groups = np.array(groups_list)
        
        return X, y_surv, y_count, groups
    
    def evaluate_best_model(self, param_grid, model_class, training_manager, with_train=False, only_last=False):
        # 1. Preparación y Entrenamiento
        clean_params = {k.replace('model__', ''): v for k, v in param_grid.items()}
        model = model_class(**clean_params)
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', model)
        ])
        
        X_train, _, y_count_train, groups_train = training_manager.get_training_data()
        pipeline.fit(X_train, y_count_train)
        
        # 2. Selección de base de datos (Train o Test)
        if with_train:
            X_base = X_train
            y_base = y_count_train
            groups_base = groups_train
            label = "ENTRENAMIENTO"
        else:
            X_base = self.X_test
            y_base = self.y_count_test
            groups_base = self.groups_test
            label = "TEST"

        # 3. Aplicación de filtro de "Última Fila" sobre la base elegida
        if only_last:
            # Importante: Usar groups_base para que funcione con Train o Test
            last_indices = pd.Series(groups_base).groupby(groups_base).tail(1).index
            X_eval = X_base.iloc[last_indices].reset_index(drop=True)
            y_eval = pd.Series(y_base).iloc[last_indices].reset_index(drop=True).to_numpy()
            modo = "DESPLIEGUE (Última fila)"
        else:
            X_eval = X_base
            y_eval = y_base
            modo = "TRAYECTORIA (Historial completo)"

        print(f"Evaluando {label} en modo {modo}: {len(X_eval)} muestras")

        # 4. Cálculo de métricas
        metrics_funcs = Metrics.get_metrics()
        results = {
            name: func(pipeline, X_eval, y_eval)
            for name, func in metrics_funcs.items()
        }
        
        return results