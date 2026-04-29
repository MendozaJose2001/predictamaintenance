# ./src/dataset_manager.py

from sklearn.model_selection import train_test_split
import pandas as pd
import os

_path = {
    'train' : 'data/raw/train_FD001.txt',
    'test' : 'data/raw/test_FD001.txt',
    'true' : 'data/raw/RUL_FD001.txt'
    }
    
_columnas = {
    'identificadores': ['unit_number', 'time_in_cycles'],
    'settings': ['op_setting_1', 'op_setting_2', 'op_setting_3'],
    'sensores': ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
                 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
                 'PCNf_dmd', 'W31', 'W32']
    }

def _get_txt(key_path: str):
    
    try:
        
        if key_path == 'true':
            df = pd.read_csv(
                filepath_or_buffer=_path[key_path],
                sep=r'\s+',
                names=['true_RUL'],
                header=None
                )
                
            df['unit_number'] = range(1, len(df) + 1)
                
        else:
            col_names = (
                    _columnas['identificadores'] 
                    + _columnas['settings']
                    + _columnas['sensores']
                    )
                
            df = pd.read_csv(
                filepath_or_buffer=_path[key_path],
                sep=r'\s+',
                names=col_names,
                header=None
                )
            
        return df
        
    except Exception:
        raise RuntimeError("Error leyendo el archivo") 
    
def store_dataframe_csv(df, csv_name, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
        save_path = os.path.join(output_dir, csv_name+'.csv')
        
        # Exportar a CSV
        df.to_csv(save_path, index=False) 
    
class DatasetManager():
    
    _col_description = {
    'T2': 'Total temperature at fan inlet (°R)',
    'T24': 'Total temperature at LPC outlet (°R)',
    'T30': 'Total temperature at HPC outlet (°R)',
    'T50': 'Total temperature at LPT outlet (°R)',
    'P2': 'Pressure at fan inlet (psia)',
    'P15': 'Total pressure in bypass-duct (psia)',
    'P30': 'Total pressure at HPC outlet (psia)',
    'Nf': 'Physical fan speed (rpm)',
    'Nc': 'Physical core speed (rpm)',
    'epr': 'Engine pressure ratio (--)',
    'Ps30': 'Static pressure at HPC outlet (psia)',
    'phi': 'Ratio of fuel flow to Ps30 (pps/psi)',
    'NRf': 'Corrected fan speed (rpm)',
    'NRc': 'Corrected core speed (rpm)',
    'BPR': 'Bypass Ratio (--)',
    'farB': 'Burner Fuel-air ratio (--)',
    'htBleed': 'Bleed Enthalpy (--)',
    'Nf_dmd': 'Demanded fan speed (rpm)',
    'PCNf_dmd': 'Demanded corrected fan speed (rpm)',
    'W31': 'HPT coolant bleed (lbm/s)',
    'W32': 'LPT coolant bleed (lbm/s)'
    }
    
    @classmethod
    def get_col_description(cls):
        return cls._col_description
    
    @staticmethod
    def get_base_dataset():
        # 1. Cargar datos
        df_train = _get_txt('train')
        df_test = _get_txt('test')
        df_true = _get_txt('true')
        
        # 2. Calcular RUL para TRAIN
        # Agrupamos por motor y restamos el ciclo actual del máximo alcanzado
        max_cycles_train = df_train.groupby('unit_number')['time_in_cycles'].transform('max')
        df_train['RUL'] = max_cycles_train - df_train['time_in_cycles']
        df_train['evento'] = 1  # Fallo observado

        # 3. Calcular RUL para TEST (usando Ground Truth)
        # Primero sumamos 100 a los IDs de test para evitar colisiones con train
        df_test['unit_number'] += 100
        df_true['unit_number'] += 100
        
        # Obtenemos el último ciclo registrado en el archivo de test
        max_cycles_test = df_test.groupby('unit_number')['time_in_cycles'].max().reset_index()
        max_cycles_test.columns = ['unit_number', 'ultimo_ciclo_archivo']
        
        # Combinamos con el ground truth para saber la vida total real
        vida_total_test = pd.merge(max_cycles_test, df_true, on='unit_number')
        vida_total_test['vida_total'] = vida_total_test['ultimo_ciclo_archivo'] + vida_total_test['true_RUL']
        
        # Pasamos la vida total al dataframe de test y calculamos RUL fila a fila
        df_test = df_test.merge(vida_total_test[['unit_number', 'vida_total']], on='unit_number')
        df_test['RUL'] = df_test['vida_total'] - df_test['time_in_cycles']
        df_test['evento'] = 0  # Censurado (sigue vivo en el último registro del archivo)
        
        uni_dataset = pd.concat(
            [df_train, df_test.drop(columns=['vida_total'])], 
            ignore_index=True
            )
        
        return uni_dataset
    
    @staticmethod
    def generate_metadata(df):
        
        # Agrupamos por motor
        metadata = df.groupby('unit_number').agg(
            max_cycles=('time_in_cycles', 'max'),
            event=('evento', 'max')
        ).reset_index()

        # Contamos la cantidad de inputs por categoría definidos en tu diccionario 'columnas'
        metadata['n_settings'] = len(_columnas['settings'])
        metadata['n_sensores'] = len(_columnas['sensores'])

        # Reordenar columnas para el CSV final
        metadata = metadata[[
            'unit_number', 
            'max_cycles', 
            'n_settings', 
            'n_sensores', 
            'event'
        ]]
        
        return metadata
    
    @staticmethod
    def clean_dataset(df):
        
        stats = df.describe().T
        # Calculamos el IQR (Q3 - Q1)
        stats['IQR'] = stats['75%'] - stats['25%']

        # Identificar columnas constantes (Filtro IQR)
        cols_consts = stats[stats['IQR'] == 0].index.tolist()

        cols_protec = _columnas['identificadores'] + ['RUL', 'evento']
        cols_consts = [c for c in cols_consts if c not in cols_protec]
        
        clean_dataset = df.drop(columns=cols_consts)

        print(f"Columnas sin variabilidad detectadas: {len(cols_consts)}")
        print(f"Lista de candidatos a eliminación: {cols_consts}")
        print(f"Columnas eliminadas: {cols_consts}")
        
        return clean_dataset, stats
    
    @staticmethod
    def split_dataset(test_size=0.3, random_state=42):
        
        df = pd.read_csv('data/metadata.csv')
        # 1. Lista de unidades únicas
        unidades = df['unit_number'].unique()

        # 2. Split aleatorio de motores (unidad atómica)
        unidades_train, unidades_test = train_test_split(
            unidades, 
            test_size = test_size, 
            random_state = random_state
            )
        
        print(f"Motores para entrenamiento: {len(unidades_train)}")
        print(f"Motores para prueba: {len(unidades_test)}")
        
        return unidades_train, unidades_test