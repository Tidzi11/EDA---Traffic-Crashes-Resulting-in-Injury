# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:09:21 2024

@author: tijan
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Definisanje funkcije za prilagođeno parsiranje datuma
def custom_date_parser(date):
    return datetime.strptime(date, "%m/%d/%Y %I:%M:%S %p")

# Učitavanje CSV fajla sa podacima o saobraćajnim nesrećama i prilagođeno parsiranje kolone 'collision_datetime'
df=pd.read_csv("Traffic_Crashes_Resulting_in_Injury_20240602.csv",low_memory=False,parse_dates=['collision_datetime'], date_parser=custom_date_parser, 
                    encoding="latin_1")

# Prikazivanje prvih nekoliko redova podataka
df.head()

#%% SREDJIVANJE PODATAKA
df['type_of_collision'].nunique()

# Brisanje prvih 7 kolona 
df.drop(df.columns[:4], axis=1,inplace=True)

# Brisanje određenih kolona koje nisu potrebne za analizu
df.drop(['collision_date','collision_time','time_cat','juris','officer_id','beat_number','secondary_rd','weather_2'],axis=1,inplace=True)

# Brisanje dodatnih kolona koje nisu potrebne za analizu
df.drop(['point','data_as_of','data_updated_at','data_loaded_at','vz_pcf_code','vz_pcf_group','vz_pcf_description','vz_pcf_link','street_view','dph_col_grp','dph_col_grp_description'],axis=1,inplace=True)

# Prikazivanje preostalih kolona 
df.columns

df.info()

# Prikazivanje osnovnih statističkih podataka o numeričkim kolonama
df.describe()

#%%
# Pronalazak kolone sa najviše nedostajućih podataka
missing_data = df.isnull().sum()
column_with_most_missing = missing_data.idxmax()
max_missing_count = missing_data.max()

print(f"Column with the most missing data: {column_with_most_missing} ({max_missing_count} missing values)")
#%%
# Procenat redova koji sadrže nedostajuće podatke
total_rows = len(df)
rows_with_missing_data = df.isnull().any(axis=1).sum()
percent_missing_rows = (rows_with_missing_data / total_rows) * 100

print(f"Total percentage of rows with missing data: {percent_missing_rows:.2f}%")

#%%
# Brisanje redova sa nedostajućim podacima i računanje izgubljenih redova
df_cleaned = df.dropna()
rows_lost = total_rows - len(df_cleaned)

print(f"Total rows lost after dropping rows with missing data: {rows_lost}")
#%%
# Filtriranje redova sa negativnim vrednostima u kolonama 'number_killed' i 'number_injured'
negative_values_mask = (df['number_killed'] < 0) | (df['number_injured'] < 0)
df_cleaned = df[~negative_values_mask]

#%% EDA
# Grupisanje podataka po godinama i računanje ukupnog broja povređenih i poginulih
yearly_data = df.groupby('accident_year')[['number_killed', 'number_injured']].sum()

# Dodavanje nove kolone koja predstavlja ukupne žrtve (poginuli + povređeni)
yearly_data['total_casualties'] = yearly_data['number_killed'] + yearly_data['number_injured']


colors = ['#FF69B4', '#FFA500', '#ADFF2F']  

# Kreiranje bar grafikona za prikaz broja povređenih, poginulih i ukupnih žrtava po godinama
plt.figure(figsize=(12, 6))
yearly_data[['number_killed', 'number_injured', 'total_casualties']].plot(kind='bar', stacked=False, color=colors, ax=plt.gca())
plt.title('Yearly Casualties and Injuries')
plt.xlabel('Year')
plt.ylabel('Number of People')
plt.xticks(rotation=0)
plt.legend(title='Casualty Type')
plt.grid(axis='y')

plt.show()


#%%
def format_percentage(pct, allvalues):
    absolute = int(round(pct/100.*sum(allvalues)))
    return f'{pct:.1f}%' if pct > 2 else ''

party1_counts = df['party1_type'].value_counts()
party2_counts = df['party2_type'].value_counts()

killed_by_party1 = df.groupby('party1_type')['number_killed'].sum()
killed_by_party2 = df.groupby('party2_type')['number_killed'].sum()

# Broj grupa koje će se prikazivati na grafikonima
num_groups1 = len(party1_counts)
num_groups2 = len(party2_counts)


colors1 = plt.cm.Paired(range(num_groups1))
colors2 = plt.cm.Paired(range(num_groups2))

# Kreiranje potgrafikona za prikaz podataka
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Prvi pie chart za distribuciju tipova vozila Party 1
axes[0, 0].pie(party1_counts, autopct=lambda pct: format_percentage(pct, party1_counts), startangle=130, colors=colors1, textprops={'fontsize': 12})
axes[0, 0].set_title('Distribution of Party 1 Vehicle Types', fontsize=14)
axes[0, 0].legend(party1_counts.index, loc="best", fontsize=12)

# Drugi pie chart za distribuciju tipova vozila Party 2
axes[0, 1].pie(party2_counts, autopct=lambda pct: format_percentage(pct, party2_counts), startangle=170, colors=colors2, textprops={'fontsize': 12})
axes[0, 1].set_title('Distribution of Party 2 Vehicle Types', fontsize=14)
axes[0, 1].legend(party2_counts.index, loc="best", fontsize=12)

# Treći pie chart za broj poginulih po tipu vozila Party 1
axes[1, 0].pie(killed_by_party1, autopct=lambda pct: format_percentage(pct, killed_by_party1), startangle=120, colors=colors1, textprops={'fontsize': 12})
axes[1, 0].set_title('Number Killed by Party 1 Vehicle Types', fontsize=14)
axes[1, 0].legend(killed_by_party1.index, loc="best", fontsize=12)

# Četvrti pie chart za broj poginulih po tipu vozila Party 2
axes[1, 1].pie(killed_by_party2, autopct=lambda pct: format_percentage(pct, killed_by_party2), startangle=45, colors=colors2, textprops={'fontsize': 12})
axes[1, 1].set_title('Number Killed by Party 2 Vehicle Types', fontsize=14)
axes[1, 1].legend(killed_by_party2.index, loc="best", fontsize=12)

plt.tight_layout()
plt.show()


#%%

# Grupisanje podataka po stanju puta i sumiranje broja poginulih
killed_by_road_condition = df.groupby('road_cond_1')['number_killed'].sum()

# Sortiranje podataka po broju poginulih
killed_by_road_condition = killed_by_road_condition.sort_values(ascending=False)

# Kreiranje bar grafikona za prikaz broja poginulih po stanju puta
plt.figure(figsize=(25, 6))
killed_by_road_condition.plot(kind='bar', color='skyblue')
plt.title('Number of People Killed by Road Condition')
plt.xlabel('Road Condition')
plt.ylabel('Number of People Killed')
plt.xticks(rotation=0)
plt.grid(axis='y')

plt.show()

#%%
# Filtriranje da izbacimo redove gde je 'road_cond_1' jednak 'no unusual condition'
filtered_df = df[df['road_cond_1'] != 'No Unusual Condition']

# Grupisanje podataka po stanju puta i sumiranje broja poginulih
killed_by_road_condition = filtered_df.groupby('road_cond_1')['number_killed'].sum()

# Sortiranje podataka po broju poginulih
killed_by_road_condition = killed_by_road_condition.sort_values(ascending=False)

# Kreiranje bar grafikona za prikaz broja poginulih po stanju puta
plt.figure(figsize=(25, 6))
killed_by_road_condition.plot(kind='bar', color='skyblue')
plt.title('Number of People Killed by Road Condition (Excluding "No Unusual Condition")')
plt.xlabel('Road Condition')
plt.ylabel('Number of People Killed')
plt.xticks(rotation=0)
plt.grid(axis='y')

plt.show()
#%%
# Grupisanje podataka po mesecima i sumiranje broja poginulih i povređenih
monthly_totals = df.groupby('month')[['number_killed', 'number_injured']].sum()
monthly_totals = monthly_totals.reindex(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

# Računanje prosečnog broja poginulih i povređenih po mesecu
average_deaths = monthly_totals['number_killed'].mean()
average_injuries = monthly_totals['number_injured'].mean()

print(f"Average number of deaths per month: {average_deaths}")
print(f"Average number of injuries per month: {average_injuries}")


colors = ['#FF69B4', '#FFA500']

# Kreiranje bar grafikona za prikaz ukupnog broja poginulih i povređenih po mesecima
plt.figure(figsize=(10, 6))
monthly_totals.plot(kind='bar', ax=plt.gca(), color=colors)
plt.title('Total Deaths and Injuries per Month')
plt.xlabel('Month')
plt.ylabel('Total Number')
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.tight_layout()
plt.show()
#%%
# Grupisanje podataka sedmicno i sumiranje broja poginulih i povređenih
weekly_totals = df.groupby('day_of_week')[['number_killed', 'number_injured']].sum()

# Definisanje redosleda dana u nedelji
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Sortiranje indeksa prema definisanom redosledu dana u nedelji
weekly_totals = weekly_totals.reindex(days_order)

# Računanje prosečnog broja poginulih i povređenih po danu u nedelji
average_deaths = weekly_totals['number_killed'].mean()
average_injuries = weekly_totals['number_injured'].mean()

print(f"Average number of deaths per week: {average_deaths}")
print(f"Average number of injuries per week: {average_injuries}")


colors = ['#FF69B4', '#FFA500']

# Kreiranje bar grafikona za prikaz ukupnog broja poginulih i povređenih po danima u nedelji
plt.figure(figsize=(10, 6))
weekly_totals.plot(kind='bar', ax=plt.gca(), color=colors)
plt.title('Total Deaths and Injuries per Week')
plt.xlabel('Day of Week')
plt.ylabel('Total Number')
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.tight_layout()
plt.show()

#%%
# Mapa za prevođenje meseci sa engleskog na srpski
english_to_serbian_months = {
    'January': 'januar',
    'February': 'februar',
    'March': 'mart',
    'April': 'april',
    'May': 'maj',
    'June': 'jun',
    'July': 'jul',
    'August': 'avgust',
    'September': 'septembar',
    'October': 'oktobar',
    'November': 'novembar',
    'December': 'decembar'
}

# Dodavanje nove kolone sa mesecima na srpskom jeziku
df['month_serbian'] = df['month'].map(english_to_serbian_months)

print("DataFrame with Serbian months:")
print(df[['month', 'month_serbian', 'number_killed', 'number_injured']])

#%%
# Definisanje funkcije za dobijanje statistike za određeni mesec na srpskom jeziku
def get_month_statistics(month_serbian, df):
    # Filtriranje podataka za dati mesec
    month_data = df[df['month_serbian'] == month_serbian]

    # Računanje različitih statističkih podataka za dati mesec
    total_killed = month_data['number_killed'].sum()
    total_injured = month_data['number_injured'].sum()
    average_killed = month_data['number_killed'].mean()
    average_injured = month_data['number_injured'].mean()
    min_killed = month_data['number_killed'].min()
    min_injured = month_data['number_injured'].min()
    max_killed = month_data['number_killed'].max()
    max_injured = month_data['number_injured'].max()
    most_common_weather = month_data['weather_1'].mode()[0]

    # Ispis statističkih podataka
    print(f"Statistics for {month_serbian}:")
    print(f"Total killed: {total_killed}")
    print(f"Total injured: {total_injured}")
    print(f"Average killed: {average_killed:.2f}")
    print(f"Average injured: {average_injured:.2f}")
    print(f"Min killed: {min_killed}")
    print(f"Min injured: {min_injured}")
    print(f"Max killed: {max_killed}")
    print(f"Max injured: {max_injured}")
    print(f"Weather with most casualties: {most_common_weather}")

# Prikupljanje unosa korisnika za mesec na srpskom jeziku i pozivanje funkcije za dobijanje statistike
try:
    month_serbian = input("Enter a month in Serbian (januar, februar, ..., decembar): ").strip().lower()
    get_month_statistics(month_serbian, df)
except KeyError:
    print("Invalid month. Please enter a valid month in Serbian.")

#%%
# Dodavanje kolone sa ukupnim brojem žrtava (poginuli + povređeni)
df['total_casualties'] = df['number_killed'] + df['number_injured']

# Pretpostavljam da je naziv kolone 'Current Police Districts'
districts = df['Current Police Districts'].unique()

# Ispisivanje naziva distrikta
for district in districts:
    print(district)
# Grupisanje podataka po trenutnim policijskim distriktima i sumiranje ukupnog broja žrtava
grouped = df.groupby('Current Police Districts')['total_casualties'].sum().reset_index()

# Sortiranje podataka po policijskim distriktima
grouped = grouped.sort_values(by='Current Police Districts')

# Kreiranje linijskog grafikona za prikaz ukupnog broja žrtava po policijskim distriktima
plt.figure(figsize=(10, 6))
plt.plot(grouped['Current Police Districts'], grouped['total_casualties'], marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
plt.title('Distribution of Current Police Districts vs. Total Casualties')
plt.xlabel('Current Police Districts')
plt.ylabel('Total Casualties')
plt.grid(True)
plt.show()

#%%
# Grupisanje podataka po trenutnim policijskim distriktima i sumiranje ukupnog broja žrtava
grouped = df.groupby('Current Police Districts')['total_casualties'].sum().reset_index()

# Sortiranje podataka po policijskim distriktima
grouped = grouped.sort_values(by='Current Police Districts')


colors = plt.cm.Paired(range(len(grouped)))

# Kreiranje bar grafikona za prikaz ukupnog broja žrtava po policijskim distriktima
plt.figure(figsize=(10, 6))
plt.bar(grouped['Current Police Districts'], grouped['total_casualties'], color=colors)

# Dodavanje naslova i oznaka osi
plt.title('Distribution of Current Police Districts vs. Total Casualties')
plt.xlabel('Current Police Districts')
plt.ylabel('Total Casualties')

# Dodavanje mreže za bolju preglednost
plt.grid(True)

# Rotacija oznaka na x-osi ako su previše blizu
plt.xticks(rotation=90)

# Prikaz grafikona
plt.show()

#%%
# Kreiranje kopije sa odabranim kolonama
df_copy = df[['month', 'intersection', 'number_killed', 'number_injured']].copy()

# Prikazivanje jedinstvenih vrednosti u koloni 'intersection'
df_copy['intersection'].unique()

#%%
# Funkcija za procesiranje vrednosti u koloni 'intersection'
def process_intersection(intersection):
    if intersection == 'Intersection <= 20ft':
        return 'Intersection'
    elif intersection == 'Midblock > 20ft':
        return 'Midblock'
    elif intersection == 'Intersection Rear End <= 150ft':
        return 'Intersection Rear End'
    else:
        return 'Undefined'

#%%
# Primena funkcije za procesiranje vrednosti u koloni 'intersection'
df_copy['intersection'] = df_copy['intersection'].apply(process_intersection)

# Grupisanje podataka po tipu raskrsnice i sumiranje broja poginulih i povređenih
grouped = df_copy.groupby('intersection').agg({
    'number_killed': 'sum',
    'number_injured': 'sum'
}).reset_index()

# Sortiranje grupisanih podataka po broju poginulih i povređenih
grouped = grouped.sort_values(by=['number_killed', 'number_injured'], ascending=False)

#%%
# Prikaz prvih nekoliko redova grupisanih podataka
grouped.head()

#%% 
# Grupisanje po koloni 'primary_rd' i agregiranje statistika
grouped = df.groupby('primary_rd').agg({
    'number_killed': 'sum',
    'number_injured': 'sum',
    'tb_latitude': 'first',
    'tb_longitude': 'first'
}).reset_index()

# Kreiranje grafikona raspršenosti za povrede, gde veličina oznake odgovara broju povređenih
plt.figure(figsize=(10, 6))

plt.scatter(grouped['tb_latitude'], grouped['tb_longitude'], 
          s=grouped['number_injured'] * 1,   color='blue', alpha=0.6, label='Injured')

# Kreiranje grafikona raspršenosti za smrtne slučajeve, gde veličina oznake odgovara broju poginulih
plt.scatter(grouped['tb_latitude'], grouped['tb_longitude'], 
            s=grouped['number_killed'] * 20, color='red', alpha=0.6, label='Killed')

# Crtanje osnovnih puteva kao iscrtanih sivih linija
#plt.plot(grouped['tb_latitude'], grouped['tb_longitude'], color='gray', linestyle='dashed', linewidth=0.5)


plt.title('Map of Injuries and Fatalities by Primary Roads')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.grid(True)

plt.show()

#%%
# Grupisanje po koloni 'collision_severity' i agregiranje statistika
grouped = df.groupby('collision_severity').agg({
    'number_killed': 'sum',
    'number_injured': 'sum'
}).reset_index()

colors = plt.cm.Paired(range(len(grouped)))

plt.figure(figsize=(14, 7))

# Kreiranje dijagrama za broj povređenih po težini sudara
plt.subplot(1, 2, 1)
plt.pie(grouped['number_injured'], autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Number of Injured by Collision Severity')
plt.legend(grouped['collision_severity'], loc="best")

# Kreiranje dijagrama za broj poginulih po težini sudara
plt.subplot(1, 2, 2)
plt.pie(grouped['number_killed'], startangle=140, colors=colors)
plt.title('Number of Killed by Collision Severity')
plt.legend(grouped['collision_severity'], loc="best")

plt.tight_layout()
plt.show()
#%%
grouped = df.groupby('Current Police Districts')['total_casualties'].sum().reset_index()

# Sortiranje podataka po policijskim distriktima
grouped = grouped.sort_values(by='Current Police Districts')


colors = plt.cm.Paired(range(len(grouped)))

# Kreiranje bar grafikona za prikaz ukupnog broja žrtava po policijskim distriktima
plt.figure(figsize=(10, 6))
plt.bar(grouped['Current Police Districts'], grouped['total_casualties'], color=colors)

# Dodavanje naslova i oznaka osi
plt.title('Distribution of Current Police Districts vs. Total Casualties')
plt.xlabel('Current Police Districts')
plt.ylabel('Total Casualties')

# Dodavanje mreže za bolju preglednost
plt.grid(True)

# Rotacija oznaka na x-osi ako su previše blizu
plt.xticks(rotation=90)

# Prikaz grafikona
plt.show()
#%%
# Grupisanje po koloni 'type_of_collision' i agregiranje statistika
grouped = df.groupby('type_of_collision').agg({
    'number_killed': 'sum',
    'number_injured': 'sum'
}).reset_index()


colors = plt.cm.Paired(range(len(grouped)))

plt.figure(figsize=(20, 14))

# Prvi podgrafik - dijagram za broj povređenih po vrsti sudara
plt.subplot(2, 2, 1)
plt.pie(grouped['number_injured'], autopct='%1.1f%%', startangle=200, colors=colors, pctdistance=0.85)
plt.title('Number of Injured by Type of Collision')
plt.legend(grouped['type_of_collision'], loc="best")

# Drugi podgrafik - dijagram za broj poginulih po vrsti sudara
plt.subplot(2, 2, 2)
plt.pie(grouped['number_killed'], autopct='%1.1f%%', startangle=150, colors=colors, pctdistance=0.85)
plt.title('Number of Killed by Type of Collision')
plt.legend(grouped['type_of_collision'], loc="best")

# Treći podgrafik - stubičasti grafikon za broj povređenih po vrsti sudara
#plt.subplot(2, 2, 3)
#plt.bar(grouped['type_of_collision'], grouped['number_injured'], color=colors)
#plt.title('Number of Injured by Type of Collision')
#plt.xlabel('Type of Collision')
#plt.ylabel('Number of Injured')

# Četvrti podgrafik - stubičasti grafikon za broj poginulih po vrsti sudara
#plt.subplot(2, 2, 4)
#plt.bar(grouped['type_of_collision'], grouped['number_killed'], color=colors)
#plt.title('Number of Killed by Type of Collision')
#plt.xlabel('Type of Collision')
#plt.ylabel('Number of Killed')

plt.tight_layout()
plt.show()
#%%
# Lista sa imenima kolona koje želimo da prikažemo na grafikonima
columns_to_plot = ['road_surface', 'road_cond_1', 'lighting', 'weather_1']

# Kreiranje figure i osovina za podgrafove dimenzija 25x35
fig, axes = plt.subplots(len(columns_to_plot), 2, figsize=(25, 35))
fig.subplots_adjust(hspace=0.4)

# Petlja kroz svaku kolonu koju želimo da prikažemo
for i, column in enumerate(columns_to_plot):
    grouped = df.groupby(column).agg({
        'number_killed': 'sum',
        'number_injured': 'sum'
    }).reset_index()
    
    colors = plt.cm.Paired(range(len(grouped)))
    
    # Kreiranje stubičastog grafikona za broj povređenih po trenutnoj koloni
    axes[i, 0].bar(grouped[column], grouped['number_injured'], color=colors, label=grouped[column])
    axes[i, 0].set_title(f'Number of Injured by {column.capitalize()}')
    axes[i, 0].set_xlabel(column.capitalize())
    axes[i, 0].set_ylabel('Number of Injured')
    axes[i, 0].tick_params(axis='x', labelsize=10, rotation=45)
    axes[i, 0].legend(title=column.capitalize(), fontsize='small')

    # Kreiranje stubičastog grafikona za broj poginulih po trenutnoj koloni
    axes[i, 1].bar(grouped[column], grouped['number_killed'], color=colors, label=grouped[column])
    axes[i, 1].set_title(f'Number of Killed by {column.capitalize()}')
    axes[i, 1].set_xlabel(column.capitalize())
    axes[i, 1].set_ylabel('Number of Killed')
    axes[i, 1].tick_params(axis='x', labelsize=10, rotation=45)
    axes[i, 1].legend(title=column.capitalize(), fontsize='small')

plt.show()
#%%
# Grupisanje po koloni 'intersection' i agregiranje statistika
grouped = df.groupby('intersection').agg({
    'number_killed': 'sum',
    'number_injured': 'sum'
}).reset_index()

colors = plt.cm.Paired(range(len(grouped)))

plt.figure(figsize=(14, 7))

# Prvi podgrafik - dijagram za broj povređenih po raskrsnici
plt.subplot(1, 2, 1)
plt.pie(grouped['number_injured'], autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Number of Injured by Intersection')
plt.legend(grouped['intersection'], loc="best")

# Drugi podgrafik - dijagram za broj poginulih po raskrsnici
plt.subplot(1, 2, 2)
plt.pie(grouped['number_killed'], autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Number of Killed by Intersection')
plt.legend(grouped['intersection'], loc="best")

plt.show()
#%%
# Izračunavanje statističkih parametara za kolonu 'number_killed': srednja vrednost, medijana i standardna devijacija
stats_number_killed = df['number_killed'].agg(['mean', 'median', 'std'])
print("Statistical analysis on 'number_killed':")
print(stats_number_killed)
print()

# Izračunavanje statističkih parametara za kolonu 'number_injured': srednja vrednost, medijana i standardna devijacija
stats_number_injured = df['number_injured'].agg(['mean', 'median', 'std'])
print("Statistical analysis on 'number_injured':")
print(stats_number_injured)
print()

# Brojanje jedinstvenih vrednosti za kolonu 'road_surface'
count_road_surface = df['road_surface'].value_counts()
print("Count of unique values for 'road_surface':")
print(count_road_surface)
#%%
# Funkcija koja računa statističke parametre za zadatu kolonu DataFrame-a
def calculate_statistics(df, column):
    stats = df.groupby(column).agg({
        'number_killed': ['mean', 'max', 'min', 'median', 'std', 'sum'],
        'number_injured': ['mean', 'max', 'min', 'median', 'std', 'sum']
    }).reset_index()
    
 
    stats.columns = [column, 'mean_killed', 'max_killed', 'min_killed', 'median_killed', 'std_killed', 'total_killed',
                     'mean_injured', 'max_injured', 'min_injured', 'median_injured', 'std_injured', 'total_injured']
    
    # Sortiranje statistika po prosečnom broju poginulih i prosečnom broju povređenih
    stats_sorted = stats.sort_values(by=['mean_killed', 'mean_injured'])
    
    return stats_sorted


stats_weather = calculate_statistics(df, 'weather_1') # Računanje statističkih parametara za vremenske uslove
stats_lighting = calculate_statistics(df, 'lighting') # Računanje statističkih parametara za osvetljenje
stats_road_surface = calculate_statistics(df, 'road_surface') # Računanje statističkih parametara za stanje puta


print("Statistical Analysis for Weather:")
print(stats_weather)
print("\n")

print("Statistical Analysis for Lighting:")
print(stats_lighting)
print("\n")


print("Statistical Analysis for Road Surface:")
print(stats_road_surface)

# Podesite opciju za prikazivanje svih kolona
#pd.set_option('display.max_columns', None)
