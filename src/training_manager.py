# ./src/training_manager.py

import pandas as pd
import numpy as np
import joblib
import math
import warnings
import os
from tqdm.auto import tqdm
from contextlib import contextmanager
from statsmodels.tools.sm_exceptions import ConvergenceWarning, DomainWarning, ValueWarning

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold, GridSearchCV

from src.metrics_manager import Metrics

@contextmanager
def _tqdm_joblib(tqdm_object):
    """Context manager para parchear joblib y mostrar barra de progreso de tqdm."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
        
def _supress_ggs_warnings():

    warnings.filterwarnings("ignore")
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', DomainWarning)
    warnings.simplefilter('ignore', ValueWarning)

    os.environ['PYTHONWARNINGS'] = 'ignore'
    
class GGSTrainingManager:
    
    def __init__(self, id_traing) -> None:
        
        X_train, y_surv_train, y_count_train, groups_train = self._preparar_entrenamiento_grupal(
            list_ids = id_traing
        )
        
        self.X_train = X_train
        self.y_surv_train = y_surv_train
        self.y_count_train = y_count_train
        self.groups_train = groups_train
        self.ggs = None
        
    @staticmethod
    def _preparar_entrenamiento_grupal(
        list_ids: list
        ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        
        X_list = []
        y_survival_list = []
        y_count_list = []
        groups_list = []

        for idx in list_ids:
            # Cargar motor
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
    
    def get_training_data(self):
        return (
            self.X_train, 
            self.y_surv_train,
            self.y_count_train,
            self.groups_train
        )
            
    def group_grid_search(self, param_grid, model, n_folds=5, silence=True):
        
        if silence:
            _supress_ggs_warnings()
        
        scaler = RobustScaler()
        metrics = Metrics.get_metrics()

        base_pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model) 
        ])

        gkf = GroupKFold(n_splits=n_folds)
        
        ggs = GridSearchCV(
            base_pipeline, 
            param_grid=param_grid, 
            cv=gkf,
            refit='S_score',
            scoring=metrics,
            n_jobs=2, 
            error_score=np.nan,
            return_train_score=True
        )
        
        n_possible_tasks = math.prod(len(v) for v in param_grid.values())
        total_tasks = n_possible_tasks * n_folds
        
        print("Starting Grid Search...")
        
        with _tqdm_joblib(tqdm(desc="GGS en progreso", total=total_tasks)) as progress_bar:
            
            ggs.fit(
                X = self.X_train, 
                y = self.y_count_train, 
                groups=self.groups_train
                )
        
        self.ggs = ggs
        
        return self.ggs

    def get_ggs_metrics(self):
        
        if not self.ggs:
            raise ValueError("To see GGS resume, run the GGS first!")
        
        print("Method in progress. Return Later!")
        
        
