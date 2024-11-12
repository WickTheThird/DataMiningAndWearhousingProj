import pandas as pd
import numpy as np
from django.conf import settings
from django.shortcuts import render
from . import models
from django.db import transaction
import os

def read_csv_file(filename, nrows):
    csv_dir = os.path.join(settings.BASE_DIR, 'src/csv')
    file_path = os.path.join(csv_dir, filename)
    
    try:
        df = pd.read_csv(file_path, nrows=nrows, index_col=None)
        return df
    except FileNotFoundError:
        print(f"Error: File {filename} not found in the csv directory.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

@transaction.atomic
def bulk_insert_years(df):
    
    years = list(set([row.Year for row in df.itertuples()]))
    models.Years.objects.bulk_create([models.Years(year=year) for year in years])

    return [row.year for row in models.Years.objects.all()]

@transaction.atomic
def bulk_insert_crude_birth_rates(df):

    # data = [row for row in df.itertuples()]
    # print(data[0]._4)
    
    models.CrudeBirthRate.objects.bulk_create([
        models.CrudeBirthRate(
            year=models.Years.objects.get(year=row.Year),
            entity=row.Entity,
            birth_rate=row._4
        ) for row in df.itertuples()
    ])
    return models.CrudeBirthRate.objects.all()

@transaction.atomic
def bulk_insert_population_and_demography(df):
    models.PopulationAndDemography.objects.bulk_create([
        models.PopulationAndDemography(
            year=models.Years.objects.get(year=row.Year),
            entity=row.Entity,
            population=row._4
        ) for row in df.itertuples()
    ])
    return models.PopulationAndDemography.objects.all()

@transaction.atomic
def bulk_insert_political_regimes(df):

    models.PoliticalRegieme.objects.bulk_create([
        models.PoliticalRegieme(
            year=models.Years.objects.get(year=row.Year),
            entity=row.Entity,
            political_regime=row._4
        ) for row in df.itertuples()
    ])
    return models.PoliticalRegieme.objects.all()

@transaction.atomic
def bulk_insert_religious_large(df):
    # data = [row for row in df.itertuples()]
    # print(data[0])
    
    df = df.fillna(-1)

    models.ReligiousLarge.objects.bulk_create([
        models.ReligiousLarge(
            year=models.Years.objects.get(year=row.Year),
            entity=row.Country,
            group_name=row._5,
            group_proportion=row._6,
            group_estimate=row._7,
            outlier=row.Outlier,
            anchor=row.Anchor,
            independent_country=row._4
        ) for row in df.itertuples()
    ])
    return models.ReligiousLarge.objects.all()

# Read CSV files
religious_large = read_csv_file('religious_large.csv', nrows=47069)
population_and_demography = read_csv_file('population_and_demography.csv', nrows=18723)
crude_birth_rate = read_csv_file('crude_birth_rate_old.csv', nrows=18723)
political_regime = read_csv_file('political_regime.csv', nrows=31139)

# Insert data in bulk
# years = bulk_insert_years(political_regime)
# crude_birth_rates = bulk_insert_crude_birth_rates(crude_birth_rate)
# population_and_demograpies = bulk_insert_population_and_demography(population_and_demography)
# political_regimes = bulk_insert_political_regimes(political_regime)
# religious_larges = bulk_insert_religious_large(religious_large)

# print(f"Inserted {len(years)} Years")
# print(f"Inserted {len(crude_birth_rates)} Crude Birth Rates")
# print(f"Inserted {len(population_and_demograpies)} Population and Demography")
# print(f"Inserted {len(political_regimes)} Political Regimes")
# print(f"Inserted {len(religious_larges)} Religious Large entries")
