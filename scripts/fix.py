import pandas as pd
import numpy as np

np.random.seed(42)
N = 800

def noise(arr, s):
    return arr + np.random.normal(0, s, len(arr))

names = ['normal', 'caution', 'alert']
spo2_ranges = [(95,100,0.5), (92,94,0.3), (85,92,0.5)]
hr_ranges =   [(60,100,2.0), (101,120,2.0), (120,160,3.0)]
temp_ranges = [(36.1,37.5,0.1), (36.1,38.0,0.2), (38.5,40.5,0.2)]

for i, name in enumerate(names):
    spo2 = np.clip(noise(np.random.uniform(spo2_ranges[i][0], spo2_ranges[i][1], N), spo2_ranges[i][2]), 80, 100)
    hr   = np.clip(noise(np.random.uniform(hr_ranges[i][0],   hr_ranges[i][1],   N), hr_ranges[i][2]),   40, 200)
    temp = np.clip(noise(np.random.uniform(temp_ranges[i][0], temp_ranges[i][1], N), temp_ranges[i][2]), 35, 42)
    timestamps = np.arange(N) * 5000
    df = pd.DataFrame({
        'timestamp':   timestamps,
        'spo2':        spo2.round(2),
        'heart_rate':  hr.round(1),
        'temperature': temp.round(2)
    })
    df.to_csv(name + '.csv', index=False)
    print('Saved ' + name + '.csv')

print('Done')