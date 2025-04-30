#%% Trial
from google.cloud import bigquery

client = bigquery.Client(project='minerva-weather-derivatives')

query = """
    SELECT *
    FROM `bigquery-public-data.noaa_gsod.gsod2023`
    WHERE stn = 'your_station_id'
"""

query_job = client.query(query)  # Make an API request.

for row in query_job:
    print(row)


#%% Trial 2

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\giamb\AppData\Roaming\gcloud\application_default_credentials.json"

from google.cloud import bigquery
client = bigquery.Client(project='minerva-weather-derivatives')

query = """
    SELECT COUNT(*)
    FROM `bigquery-public-data.noaa_gsod.gsod1929`
    WHERE stn = '037770'
"""

query_job = client.query(query)  # Make an API request.
for row in query_job:
    print(row)


#%% Joining datasets

query2 = """
    SELECT
        gsod.*
    FROM
        `bigquery-public-data.noaa_gsod.gsod1929` AS gsod
    JOIN
        `bigquery-public-data.noaa_gsod.stations` AS stations
    ON
        gsod.stn = stations.usaf
"""


query_job = client.query(query2)  # Make an API request.
for row in query_job:
    print(row)
    
#%% Count the number of observations in the joined dataset

query3 = """
    SELECT
        COUNT(*) AS num_observations
    FROM
        `bigquery-public-data.noaa_gsod.gsod1929` AS gsod
    JOIN
        `bigquery-public-data.noaa_gsod.stations` AS stations
    ON
        gsod.stn = stations.usaf
"""
# AND stations.country = 'USA';

query_job = client.query(query3)  # Make an API request.
for row in query_job:
    print(row)
    

#%% Query to find the longest time series of observations for US stations

query4 = """
    SELECT 
        stations.*, 
        DATE_DIFF(PARSE_DATE('%Y%m%d', CAST(`end` AS STRING)), 
                  PARSE_DATE('%Y%m%d', CAST(`begin` AS STRING)), DAY) AS date_diff
    FROM `bigquery-public-data.noaa_gsod.stations` as stations
    WHERE stations.country = 'US'
    ORDER BY date_diff DESC
    LIMIT 20
"""

query_job = client.query(query4)  # Make an API request.
for row in query_job:
    print(row)

#%% Query to find the latest date in the dataset

query_latest = """
    SELECT 
        MAX(PARSE_DATE('%Y%m%d', CAST(`end` AS STRING))) AS latest_end_date
    FROM `bigquery-public-data.noaa_gsod.stations`
"""

query_job = client.query(query_latest)  # Make an API request.
for row in query_job:
    print(row)


#%% Query to merge the entire datasets (Not useful)
for year in range(1929, 2026):
    query_loop = f"""
        SELECT 
            gsod.*, 
            stations.*,
            {year} AS year
        FROM `bigquery-public-data.noaa_gsod.gsod{year}` AS gsod
        JOIN `bigquery-public-data.noaa_gsod.stations` AS stations
          ON gsod.stn = stations.usaf
    """
    query_job = client.query(query_loop)  # Make an API request.
    print(f"--- Results for year {year} ---")
    # for row in query_job:
    #     print(row)

#%% Query to select the most present countries from the merged dataset
query_merged_countries = """
SELECT 
    stations.country, 
    COUNT(*) AS num_observations
FROM 
    `bigquery-public-data.noaa_gsod.gsod*` AS gsod
JOIN 
    `bigquery-public-data.noaa_gsod.stations` AS stations
    ON gsod.stn = stations.usaf
WHERE 
    _TABLE_SUFFIX BETWEEN '1929' AND '2025'
GROUP BY 
    stations.country
ORDER BY 
    num_observations DESC
LIMIT 20
"""

query_job = client.query(query_merged_countries)  # Make an API request.
for row in query_job:
    print(row)


#%% Query to check if the station is still in function in 2025

query_station = """
SELECT
    gsod.*
FROM
    `bigquery-public-data.noaa_gsod.gsod2025` AS gsod
WHERE
    gsod.stn = '722860'
LIMIT 10
    """
    
query_job = client.query(query_station)  # Make an API request.
for row in query_job:
    print(row)


#%% Query to check the number of N/A values for the station average temperature

query_na = """
    SELECT 
        gsod.stn, 
        COUNT(*) AS num_na
    FROM 
        `bigquery-public-data.noaa_gsod.gsod*` AS gsod
    WHERE 
        gsod.stn = '722860' AND
        gsod.temp IS NULL
    GROUP BY gsod.stn
    LIMIT 1
"""

query_job = client.query(query_na)  # Make an API request.
for row in query_job:
    print(row)


#%% Query to check the number of N/A values for the station max temperature

query_na = """
    SELECT 
        gsod.stn, 
        COUNT(*) AS num_na
    FROM 
        `bigquery-public-data.noaa_gsod.gsod*` AS gsod
    WHERE 
        gsod.stn = '722860' AND
        gsod.max IS NULL
    GROUP BY gsod.stn
    LIMIT 1
"""

query_job = client.query(query_na)  # Make an API request.
for row in query_job:
    print(row)

#%% Query to check the number of N/A values for the station min temperature

query_na = """
    SELECT 
        gsod.stn, 
        COUNT(*) AS num_na
    FROM 
        `bigquery-public-data.noaa_gsod.gsod*` AS gsod
    WHERE 
        gsod.stn = '722860' AND
        gsod.min IS NULL
    GROUP BY gsod.stn
    LIMIT 1
"""

query_job = client.query(query_na)  # Make an API request.
for row in query_job:
    print(row)

#%% Query to check the number of N/A values for the station year

query_na = """
    SELECT 
        gsod.stn, 
        COUNT(*) AS num_na
    FROM 
        `bigquery-public-data.noaa_gsod.gsod*` AS gsod
    WHERE 
        gsod.stn = '722860' AND
        gsod.year IS NULL
    GROUP BY gsod.stn
    LIMIT 1
"""

query_job = client.query(query_na)  # Make an API request.
for row in query_job:
    print(row)


#%% Query to check the number of N/A values for the month

query_na = """
    SELECT 
        gsod.stn, 
        COUNT(*) AS num_na
    FROM 
        `bigquery-public-data.noaa_gsod.gsod*` AS gsod
    WHERE 
        gsod.stn = '722860' AND
        gsod.mo IS NULL
    GROUP BY gsod.stn
    LIMIT 1
"""

query_job = client.query(query_na)  # Make an API request.
for row in query_job:
    print(row)

#%% Query to check the number of N/A values for the day

query_na = """
    SELECT 
        gsod.stn, 
        COUNT(*) AS num_na
    FROM 
        `bigquery-public-data.noaa_gsod.gsod*` AS gsod
    WHERE 
        gsod.stn = '722860' AND
        gsod.da IS NULL
    GROUP BY gsod.stn
    LIMIT 1
"""

query_job = client.query(query_na)  # Make an API request.
for row in query_job:
    print(row)


#%% Query creating the csv dataset
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project='minerva-weather-derivatives')

query_df = """
SELECT 
    year,
    mo,
    da,
    temp,
    `max`,
    `min`
FROM
    `bigquery-public-data.noaa_gsod.gsod*`
WHERE
    stn = '722860'
"""

query_job = client.query(query_df)  # Execute the query
df = query_job.to_dataframe()       # Convert the results to a DataFrame

print("Before conversion:")
print(df.head())

# Convert the temperature columns from Fahrenheit to Celsius and round to 2 decimals.
df['temp'] = df['temp'].apply(lambda f: round((f - 32) * 5/9, 2))
df['max'] = df['max'].apply(lambda f: round((f - 32) * 5/9, 2))
df['min'] = df['min'].apply(lambda f: round((f - 32) * 5/9, 2))

print("\nAfter conversion:")
print(df.head())

# Save the updated DataFrame to a CSV file in your specified directory.
csv_path = r"C:\Users\giamb\OneDrive\Documents\2 Uni\Associations\4 Minerva\1 Weather Derivatives\gsod_station_722860.csv"
df.to_csv(csv_path, index=False)

#Hey guys we were finally able to extract the data from the google cloud api. We ran many SQL queries, merging all the 80 datasets, pairing it with another dataset that had the station name and location, so that we could understand the station position. Then we selected a us station, located in LA, and the third oldest in the country. We check for missing values for temperature (which is recorded daily or twice a day) and is the average of many observation taken throughout the day, then min temperature and max temperature for the day. We also selected the year, month and date, reformatted into the datetime format that is readable in python, then converted the SQL query into a csv file.


#%% Query to check the number of temp abnormalities for the station

query_na = """
    SELECT 
        gsod.year,
        gsod.mo,
        gsod.da,
        gsod.max,
        gsod.min
    FROM 
        `bigquery-public-data.noaa_gsod.gsod*` AS gsod
    WHERE 
        gsod.stn = '722860' AND
        gsod.max > 50*9/5 + 32
"""

query_job = client.query(query_na)  # Make an API request.
for row in query_job:
    print(row)
    
    
#%% Query to check the number of temp abnormalities for the station

query_na = """
    SELECT 
        gsod.year,
        gsod.mo,
        gsod.da,
        gsod.max,
        gsod.min
    FROM 
        `bigquery-public-data.noaa_gsod.gsod*` AS gsod
    WHERE 
        gsod.stn = '722860' AND
        gsod.min > 40*9/5 + 32
"""

query_job = client.query(query_na)  # Make an API request.
for row in query_job:
    print(row)