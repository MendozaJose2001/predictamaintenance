#./src/dataset_manager.py

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

_path: dict[str, str] = {
    'train': 'data/raw/train_FD001.txt',
    'test': 'data/raw/test_FD001.txt',
    'true': 'data/raw/RUL_FD001.txt'
}

_columnas: dict[str, list[str]] = {
    'identificadores': ['unit_number', 'time_in_cycles'],
    'settings': ['op_setting_1', 'op_setting_2', 'op_setting_3'],
    'sensores': ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
                 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
                 'PCNf_dmd', 'W31', 'W32']
}


def _get_txt(key_path: str) -> pd.DataFrame:
    """Reads a C-MAPSS raw text file and returns it as a DataFrame.

    Handles three file types: 'train', 'test', and 'true' (ground truth RUL).
    Train and test files are assigned column names from the global _columnas
    definition. The ground truth file is assigned a single 'true_RUL' column
    with a matching 'unit_number' index starting from 1.

    Args:
        key_path: Key identifying the file to read. Must be one of
            'train', 'test', or 'true'.

    Returns:
        DataFrame with named columns corresponding to the file type.

    Raises:
        RuntimeError: If the file cannot be read, with the original
            exception message included for diagnostics.
    """
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

    except Exception as e:
        raise RuntimeError(
            f"Error reading file '{_path[key_path]}': {e}"
        ) from e


def store_dataframe_csv(df: pd.DataFrame, csv_name: str, output_dir: str) -> None:
    """Saves a DataFrame as a CSV file to the specified directory.

    Creates the output directory if it does not already exist. The file
    is always saved regardless of whether the directory was just created
    or already existed.

    Args:
        df: DataFrame to save.
        csv_name: Name of the output file without the .csv extension.
        output_dir: Path to the output directory. Created if absent.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, csv_name + '.csv')
    df.to_csv(save_path, index=False)


class DatasetManager:
    """Manages loading, preparation, cleaning, and splitting of C-MAPSS data.

    All methods are static and operate on the FD001 sub-dataset. The class
    unifies the original train and test partitions into a single dataset with
    explicit RUL labels and censoring indicators, following the survival
    analysis framing described in the project methodology.
    """

    _col_description: dict[str, str] = {
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
    def get_col_description(cls) -> dict[str, str]:
        """Returns the sensor column descriptions dictionary.

        Returns:
            Dictionary mapping sensor column names to their physical descriptions,
            including units of measurement.
        """
        return cls._col_description

    @staticmethod
    def get_base_dataset() -> pd.DataFrame:
        """Builds the unified C-MAPSS dataset with RUL labels and censoring indicators.

        Loads the raw train, test, and ground truth files and merges them into a
        single DataFrame. Train engines receive evento=1 (observed failure) and
        their RUL is computed from the maximum observed cycle. Test engines receive
        evento=0 (right-censored) and their true lifetime is reconstructed by adding
        the ground truth RUL to the last observed cycle. Test unit identifiers are
        offset by 100 to avoid collisions with train identifiers.

        Returns:
            Unified DataFrame with columns from both partitions, including RUL
            and evento for every row.
        """
        # Load raw files
        df_train = _get_txt('train')
        df_test = _get_txt('test')
        df_true = _get_txt('true')

        # Compute RUL for train engines (observed failures)
        # RUL decreases from max_cycles to 0 at the last observed cycle.
        # evento=1 only at the last cycle of each motor — not for all rows.
        # Assigning evento=1 to all rows would create multiple events per motor,
        # violating the single terminal event assumption of Cox and AFT models.
        max_cycles_train = df_train.groupby('unit_number')['time_in_cycles'].transform('max')
        df_train['RUL'] = max_cycles_train - df_train['time_in_cycles']
        df_train['evento'] = (df_train['time_in_cycles'] == max_cycles_train).astype(int)

        # Offset test unit IDs to avoid collisions with train IDs
        df_test['unit_number'] += 100
        df_true['unit_number'] += 100

        # Reconstruct true lifetime for test engines using ground truth file
        max_cycles_test = (
            df_test.groupby('unit_number')['time_in_cycles']
            .max()
            .reset_index()
            .rename(columns={'time_in_cycles': 'ultimo_ciclo_archivo'})
        )
        vida_total_test = pd.merge(max_cycles_test, df_true, on='unit_number')
        vida_total_test['vida_total'] = (
            vida_total_test['ultimo_ciclo_archivo'] + vida_total_test['true_RUL']
        )

        # Compute per-row RUL for test engines (right-censored)
        df_test = df_test.merge(
            vida_total_test[['unit_number', 'vida_total']], on='unit_number'
        )
        df_test['RUL'] = df_test['vida_total'] - df_test['time_in_cycles']
        df_test['evento'] = 0

        unified = pd.concat(
            [df_train, df_test.drop(columns=['vida_total'])],
            ignore_index=True
        )

        return unified

    @staticmethod
    def generate_metadata(df: pd.DataFrame) -> pd.DataFrame:
        """Generates a summary metadata table aggregated by engine unit.

        Produces one row per engine with its maximum observed cycle, event status,
        and the number of operational settings and sensor channels available.

        Args:
            df: Unified dataset as returned by get_base_dataset().

        Returns:
            DataFrame with one row per unit_number and columns:
                unit_number, max_cycles, n_settings, n_sensores, event.
        """
        metadata = df.groupby('unit_number').agg(
            max_cycles=('time_in_cycles', 'max'),
            event=('evento', 'max')
        ).reset_index()

        metadata['n_settings'] = len(_columnas['settings'])
        metadata['n_sensores'] = len(_columnas['sensores'])

        metadata = metadata[[
            'unit_number',
            'max_cycles',
            'n_settings',
            'n_sensores',
            'event'
        ]]

        return metadata

    @staticmethod
    def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Removes constant-valued sensor columns using an IQR filter.

        Columns whose interquartile range equals zero across the entire dataset
        carry no predictive information and are dropped. Protected columns
        (identifiers, RUL, and evento) are never removed regardless of their IQR.

        Args:
            df: Unified dataset as returned by get_base_dataset().

        Returns:
            Tuple of (clean_df, stats) where:
                clean_df: Dataset with constant columns removed.
                stats: Descriptive statistics DataFrame including the computed IQR
                    column, useful for inspection and reporting.
        """
        stats = df.describe().T
        stats['IQR'] = stats['75%'] - stats['25%']

        cols_consts = stats[stats['IQR'] == 0].index.tolist()

        protected = _columnas['identificadores'] + ['RUL', 'evento']
        cols_consts = [c for c in cols_consts if c not in protected]

        clean_df = df.drop(columns=cols_consts)

        print(f"Zero-variance columns detected: {len(cols_consts)}")
        print(f"Columns removed: {cols_consts}")

        return clean_df, stats

    @staticmethod
    def split_dataset(
        test_size: float = 0.3,
        random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Splits engine units into train and test partitions.

        Reads the metadata file and performs a random split at the unit level,
        ensuring that all cycles of a given engine fall entirely within one
        partition. This prevents any form of temporal leakage between partitions.

        Args:
            test_size: Fraction of units to allocate to the test partition.
                Defaults to 0.3.
            random_state: Random seed for reproducibility. Defaults to 42.

        Returns:
            Tuple of (units_train, units_test) containing the unit_number
            arrays for each partition.
        """
        df = pd.read_csv('data/metadata.csv')
        unidades = df['unit_number'].unique()

        unidades_train, unidades_test = train_test_split(
            unidades,
            test_size=test_size,
            random_state=random_state
        )

        return unidades_train, unidades_test