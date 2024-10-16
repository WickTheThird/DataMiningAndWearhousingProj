import pandas as pd
import numpy as np

religious_large = pd.read_csv('religious_large.csv')
religious_wide = pd.read_csv('religious_wide.csv')
crude_birth_rate = pd.read_csv('crude_birth_rate.csv', nrows=18723)
political_regime = pd.read_csv('political_regime.csv')


year_1950 = crude_birth_rate[crude_birth_rate['Year'] == 1950]

for x in year_1950['Entity']:
    print(x)
