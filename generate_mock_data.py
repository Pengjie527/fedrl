
import pandas as pd
import numpy as np

def main():
    print("Generating mock sepsis data... (This may take a moment)")
    n_patients = 200
    avg_stays = 20
    n_samples = n_patients * avg_stays

    # Create a DataFrame
    df = pd.DataFrame()

    # Patient and stay IDs
    df['icustayid'] = np.repeat(np.arange(n_patients), avg_stays)
    df['bloc'] = np.tile(np.arange(1, avg_stays + 1), n_patients)

    # Features
    # Binary
    df['gender'] = np.random.randint(0, 2, n_samples)
    df['mechvent'] = np.random.randint(0, 2, n_samples)
    df['re_admission'] = np.random.randint(0, 2, n_samples)
    # Continuous
    for col in ['age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1',
                'Potassium', 'Sodium', 'Chloride', 'Glucose', 'Magnesium', 'Calcium', 'Hb', 'WBC_count',
                'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'HCO3',
                'Arterial_lactate', 'SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2', 'cumulated_balance',
                'SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 'INR']:
        df[col] = np.random.rand(n_samples) * 100

    # Actions
    df['input_total'] = np.random.rand(n_samples) * 5000
    df['input_4hourly'] = df['input_total'] / 6
    df['output_total'] = np.random.rand(n_samples) * 5000
    df['output_4hourly'] = df['output_total'] / 6
    df['max_dose_vaso'] = np.random.rand(n_samples) * 1.5

    # Outcome
    # Create a simple outcome model based on some features
    score = df['SOFA'] * 0.1 + df['age'] * 0.05 - df['GCS'] * 0.1
    prob_death = 1 / (1 + np.exp(-score + np.mean(score)))
    death_per_patient = {pt: np.random.binomial(1, p=prob_death[df['icustayid'] == pt].iloc[0]) for pt in range(n_patients)}
    df['died_in_hosp'] = df['icustayid'].map(death_per_patient)

    df.to_csv(DATA_FILE_PATH, index=False)
    print(f"Mock data saved to '{DATA_FILE_PATH}'")

if __name__ == '__main__':
    DATA_FILE_PATH = "mock_sepsis_data.csv"
    main()
