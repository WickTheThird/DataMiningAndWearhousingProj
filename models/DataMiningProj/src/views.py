import pandas as pd
import numpy as np
from django.conf import settings
from . import models
from django.db import transaction
import os
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

matplotlib.use('Agg')

# TITLE :: CSV READING AND DB BULKING
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

@transaction.atomic
def bulk_population(df):
    
    models.Population.objects.bulk_create([
        models.Population(
            year=models.Years.objects.get(year=row.Year),
            population=row._4,
            entity=row.Entity
            ) for row in df.itertuples() if row.Year >= 1950
    ])
    
    return models.Population.objects.all()

#TITLE :: PLOTS

def plot_political_regime(country_name: str):
    data = models.PoliticalRegieme.objects.filter(entity=country_name).order_by('year')

    if not data:
        print(f"No data found for {country_name}")
        return None


    filtered_data = [entry for entry in data if entry.year.year >= 1950]
    years = np.array([entry.year.year for entry in filtered_data])
    regimes = np.array([entry.political_regime for entry in filtered_data])

    
    print(years)
    
    regeime_code = {
        0: 'Closed Autocracy',
        1: 'Electoral Autocracy',
        2: 'Electoral Democracy',
        3: 'Liberal Democracy',
    }

    regime_colors = {
        0: 'red',
        1: 'orange',
        2: 'blue',
        3: 'green',
    }

    colors = np.array([regime_colors[regime] for regime in regimes])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(years, regimes, c=colors, s=100, edgecolors='k', alpha=0.7)

    ax.set_title(f"Political Regimes in {country_name} Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Political Regime")

    for regime_score, color in regime_colors.items():
        ax.scatter([], [], color=color, label=f"{regime_score}: {regeime_code[regime_score]}")
    
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    output_dir = os.path.join(settings.MEDIA_ROOT, 'political_regime_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{country_name}_regime_plot.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.close(fig)

    return image_data, filepath

def plot_country_population(country_name: str):
    data = models.Population.objects.filter(entity=country_name).order_by('year')
    
    if not data:
        print(f"No data found for {country_name}")
        return None

    filtered_data = [entry for entry in data if entry.year.year >= 1950]

    years = np.array([entry.year.year for entry in filtered_data])
    populations = np.array([entry.population for entry in filtered_data])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, populations, marker='o', linestyle='-', color='b', label='Population')

    ax.set_title(f"Population Over Time in {country_name}", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Population", fontsize=12)
    ax.grid(True)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    output_dir = os.path.join(settings.MEDIA_ROOT, 'population_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{country_name}_population_plot.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.close(fig)

    return image_data, filepath
    

def plot_religious_groups_estimate(country_name: str):
    if country_name == 'Romania':
        models.ReligiousLarge.objects.filter(entity="Rumania").update(entity="Romania")
    
    religious_data = models.ReligiousLarge.objects.filter(entity=country_name).order_by('year')
    population_data = models.Population.objects.filter(entity=country_name).order_by('year')

    if not religious_data or not population_data:
        print(f"No data found for {country_name}")
        return None

    filtered_religious_data = [entry for entry in religious_data if entry.year.year >= 1950]
    population_by_year = {entry.year.year: entry.population for entry in population_data if entry.year.year >= 1950}

    grouped_data = {}
    for entry in filtered_religious_data:
        year = entry.year.year
        group_name = entry.group_name
        group_estimate = entry.group_estimate

        if year not in grouped_data:
            grouped_data[year] = {}
        grouped_data[year][group_name] = group_estimate

    years = sorted(grouped_data.keys())
    group_names = set(name for year in grouped_data for name in grouped_data[year])

    group_populations = {group: [] for group in group_names}

    for year in years:
        total_population = population_by_year.get(year, 0)
        for group in group_names:
            estimate = grouped_data[year].get(group, 0)
            group_populations[group].append(total_population * (estimate / 100) if total_population else 0)

    fig, ax = plt.subplots(figsize=(12, 8))

    for group, populations in group_populations.items():
        ax.plot(years, populations, label=group)

    ax.set_title(f"Religious Group Populations in {country_name} Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Population")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    output_dir = os.path.join(settings.MEDIA_ROOT, 'religious_group_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{country_name}_religious_groups_plot.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.close(fig)

    return image_data, filepath

def plot_religious_groups_estimate_areas(country_name: str):
    if country_name == 'Romania':
        models.ReligiousLarge.objects.filter(entity="Rumania").update(entity="Romania")
    
    religious_data = models.ReligiousLarge.objects.filter(entity=country_name).order_by('year')
    population_data = models.Population.objects.filter(entity=country_name).order_by('year')

    if not religious_data or not population_data:
        print(f"No data found for {country_name}")
        return None

    filtered_religious_data = [entry for entry in religious_data if entry.year.year >= 1950]
    population_by_year = {entry.year.year: entry.population for entry in population_data if entry.year.year >= 1950}

    grouped_data = {}
    for entry in filtered_religious_data:
        year = entry.year.year
        group_name = entry.group_name
        group_estimate = entry.group_estimate

        if year not in grouped_data:
            grouped_data[year] = {}
        grouped_data[year][group_name] = group_estimate

    years = sorted(grouped_data.keys())
    group_names = set(name for year in grouped_data for name in grouped_data[year])

    group_populations = {group: [] for group in group_names}

    for year in years:
        total_population = population_by_year.get(year, 0)
        for group in group_names:
            estimate = grouped_data[year].get(group, 0)
            group_populations[group].append(total_population * (estimate / 100) if total_population else 0)

    population_lists = [group_populations[group] for group in group_names]
    group_names = list(group_names)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.stackplot(years, population_lists, labels=group_names, alpha=0.8)

    ax.set_title(f"Religious Group Populations in {country_name} Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Population")
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    output_dir = os.path.join(settings.MEDIA_ROOT, 'religious_group_plots_area')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{country_name}_religious_groups_plot.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.close(fig)

    return image_data, filepath

def plot_predictions_against_test(country_name, algorithm, year, X_br, y_br):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(year, X_br, c="#ff0000", s=100, edgecolors='k', alpha=0.7)
    ax.scatter(year, y_br, c="#00ff00", s=100, edgecolors='k', alpha=0.7)

    ax.set_title(f"Compared and Expected birthrates for { country_name } Over Time using { algorithm } algorithm")
    ax.set_xlabel("Year")
    ax.set_ylabel("Birth Rate")

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    output_dir = os.path.join(settings.MEDIA_ROOT, 'prediction_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{country_name.lower().replace(' ', '_')}_{algorithm.lower().replace(' ', '_')}_prediction_plot.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.close(fig)

#TITLE :: Language Training

def train_crude_birth_rate_model():
    population_data = models.Population.objects.all().values('year', 'entity', 'population')
    religious_data = models.ReligiousLarge.objects.all().values('year', 'entity', 'group_name', 'group_proportion', 'independent_country')
    political_data = models.PoliticalRegieme.objects.all().values('year', 'entity', 'political_regime')
    birth_rate_data = models.CrudeBirthRate.objects.all().values('year', 'entity', 'birth_rate')

    population_df = pd.DataFrame(population_data)
    religious_df = pd.DataFrame(religious_data)
    political_df = pd.DataFrame(political_data)
    birth_rate_df = pd.DataFrame(birth_rate_data)

    countries = ['United Kingdom', 'France', 'Spain']
    population_df = population_df[population_df['entity'].isin(countries)]
    religious_df = religious_df[religious_df['entity'].isin(countries)]
    political_df = political_df[political_df['entity'].isin(countries)]
    birth_rate_df = birth_rate_df[birth_rate_df['entity'].isin(countries)]

    merged_df = pd.merge(population_df, religious_df, on=['year', 'entity'], how='left')
    merged_df = pd.merge(merged_df, political_df, on=['year', 'entity'], how='left')
    merged_df = pd.merge(merged_df, birth_rate_df, on=['year', 'entity'], how='left')

    merged_df = merged_df.dropna()

    X = merged_df[['population', 'group_proportion', 'political_regime', 'independent_country']]
    y = merged_df['birth_rate']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=1.0)  # Try other models here like RandomForestRegressor, etc.
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Intercept (B0): {model.intercept_}")
    print(f"Coefficients (B1, B2, B3, B4): {model.coef_}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Mean Absolute Error: {mae}")

    return model, scaler

def train_crude_birth_rate_random_forest_model():
    population_data = models.Population.objects.all().values('year', 'entity', 'population')
    religious_data = models.ReligiousLarge.objects.all().values('year', 'entity', 'group_name', 'group_proportion', 'independent_country')
    political_data = models.PoliticalRegieme.objects.all().values('year', 'entity', 'political_regime')
    birth_rate_data = models.CrudeBirthRate.objects.all().values('year', 'entity', 'birth_rate')

    population_df = pd.DataFrame(population_data)
    religious_df = pd.DataFrame(religious_data)
    political_df = pd.DataFrame(political_data)
    birth_rate_df = pd.DataFrame(birth_rate_data)

    countries = ['United Kingdom', 'France', 'Spain', 'Romania', 'Italy', 'Croatia']
    
    population_df = population_df[population_df['entity'].isin(countries)]
    religious_df = religious_df[religious_df['entity'].isin(countries)]
    political_df = political_df[political_df['entity'].isin(countries)]
    birth_rate_df = birth_rate_df[birth_rate_df['entity'].isin(countries)]

    merged_df = pd.merge(population_df, religious_df, on=['year', 'entity'], how='left')
    merged_df = pd.merge(merged_df, political_df, on=['year', 'entity'], how='left')
    merged_df = pd.merge(merged_df, birth_rate_df, on=['year', 'entity'], how='left')

    merged_df = merged_df.dropna()

    X = merged_df[['population', 'group_proportion', 'political_regime', 'independent_country']]
    y = merged_df['birth_rate']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Mean Absolute Error: {mae}")

    scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"Cross-validated R^2: {scores.mean()}")

    return model, scaler

def train_crude_birth_rate_linear_regression_model():

    population_data = models.Population.objects.all().values('year', 'entity', 'population')
    religious_data = models.ReligiousLarge.objects.all().values('year', 'entity', 'group_name', 'group_proportion', 'independent_country')
    political_data = models.PoliticalRegieme.objects.all().values('year', 'entity', 'political_regime')
    birth_rate_data = models.CrudeBirthRate.objects.all().values('year', 'entity', 'birth_rate')

    population_df = pd.DataFrame(population_data)
    religious_df = pd.DataFrame(religious_data)
    political_df = pd.DataFrame(political_data)
    birth_rate_df = pd.DataFrame(birth_rate_data)
    
    merged_df = pd.merge(population_df, religious_df, on=['year', 'entity'], how='left')
    merged_df = pd.merge(merged_df, political_df, on=['year', 'entity'], how='left')
    merged_df = pd.merge(merged_df, birth_rate_df, on=['year', 'entity'], how='left')
    merged_df = merged_df.dropna()

    X = merged_df[['population', 'group_proportion', 'political_regime', 'independent_country']]
    y = merged_df['birth_rate']

    # X = pd.get_dummies(X, columns=['political_regime'], drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)


    print(f"Intercept (B0): {model.intercept_}")
    print(f"Coefficients (B1, B2, B3, B4): {model.coef_}")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return model, scaler

def tester(model, scaler):
    countries = ['Portugal']
    
    population_data = models.Population.objects.all().values('year', 'entity', 'population')
    religious_data = models.ReligiousLarge.objects.all().values('year', 'entity', 'group_name', 'group_proportion', 'independent_country')
    political_data = models.PoliticalRegieme.objects.all().values('year', 'entity', 'political_regime')
    birth_rate_data = models.CrudeBirthRate.objects.all().values('year', 'entity', 'birth_rate')

    population_df = pd.DataFrame(population_data)
    religious_df = pd.DataFrame(religious_data)
    political_df = pd.DataFrame(political_data)
    birth_rate_df = pd.DataFrame(birth_rate_data)

    population_df = population_df[population_df['entity'].isin(countries)]
    religious_df = religious_df[religious_df['entity'].isin(countries)]
    political_df = political_df[political_df['entity'].isin(countries)]
    birth_rate_df = birth_rate_df[birth_rate_df['entity'].isin(countries)]

    merged_df = pd.merge(population_df, religious_df, on=['year', 'entity'], how='left')
    merged_df = pd.merge(merged_df,     political_df, on=['year', 'entity'], how='left')
    merged_df = pd.merge(merged_df,     birth_rate_df, on=['year', 'entity'], how='left')

    merged_df = merged_df.dropna()

    X = merged_df[['population', 'group_proportion', 'political_regime', 'independent_country']]
    y = merged_df['birth_rate']

    # print("\n\n\n")
    # print(X)
    # print("\n\n\n")

    X_scaled = scaler.transform(X)

    y_pred      = model.predict(X_scaled)
    years = merged_df[['year']]
    print(f"y_pred len: {len(y_pred)} {len(years)} {len(y)}")
    print(f"Predictions for: {y_pred}")


    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return (years, y, y_pred)


#TITLE :: RUNNERS
#! Read CSV files
# religious_large = read_csv_file('religious_large.csv', nrows=47069)
# population_and_demography = read_csv_file('population_and_demography.csv', nrows=18723)
# crude_birth_rate = read_csv_file('crude_birth_rate_old.csv', nrows=18723)
# political_regime = read_csv_file('political_regime.csv', nrows=31139)
# population = read_csv_file('population.csv', nrows=59178)

#! Insert data in bulk
# years = bulk_insert_years(political_regime)
# crude_birth_rates = bulk_insert_crude_birth_rates(crude_birth_rate)
# population_and_demograpies = bulk_insert_population_and_demography(population_and_demography)
# political_regimes = bulk_insert_political_regimes(political_regime)
# religious_larges = bulk_insert_religious_large(religious_large)
# populations = bulk_population(population)

#! Checking conversion
# print(f"Inserted {len(years)} Years")
# print(f"Inserted {len(crude_birth_rates)} Crude Birth Rates")
# print(f"Inserted {len(population_and_demograpies)} Population and Demography")
# print(f"Inserted {len(political_regimes)} Political Regimes")
# print(f"Inserted {len(religious_larges)} Religious Large entries")
# print(f"Inserted {len(populations)} Populations")

#! Plotting Political regieme
# plot_political_regime("Croatia")
# plot_political_regime("Romania")
# plot_political_regime("Spain")
# plot_political_regime("France")
# plot_political_regime("United Kingdom")

#! Plotting Population
# plot_country_population("Croatia")
# plot_country_population("Romania")
# plot_country_population("Spain")
# plot_country_population("France")
# plot_country_population("United Kingdom")

#! Plotting religious groups
# plot_religious_groups_estimate("Croatia")
# plot_religious_groups_estimate("Romania")
# plot_religious_groups_estimate("Spain")
# plot_religious_groups_estimate("France")
# plot_religious_groups_estimate("United Kingdom")

#! Area graph
# plot_religious_groups_estimate_areas("France")
# plot_religious_groups_estimate_areas("Croatia")
# plot_religious_groups_estimate_areas("Romania")
# plot_religious_groups_estimate_areas("Portugal")
# plot_religious_groups_estimate_areas("United Kingdom")

#! Training Model
# model, scaler = train_crude_birth_rate_model()
# years, base, prediction = tester(model, scaler)
# plot_predictions_against_test("Portugal", "Linear Regression", years, base, prediction)
# model, scaler = train_crude_birth_rate_random_forest_model()
# years, base, prediction = tester(model, scaler)
# plot_predictions_against_test("Portugal", "Random Forest", years, base, prediction)
