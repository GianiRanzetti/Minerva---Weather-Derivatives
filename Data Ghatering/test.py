import noaa
import pandas as pd

stations = noaa.list_stations()
stations_df = pd.DataFrame(stations)
stations_df.to_csv("stations.csv")


#stations = list_stations_2()
#locations = list_stations_2(stop = 5000)
#station2 = list_stations(20)
#stations_df = pd.DataFrame(stations)
#stations_df.to_csv("stations.csv")

#url = f"{BASE_URL}/stations?limit=5"
#response = requests.get(url, headers=HEADERS)
#obs_n = response.json()['metadata']['resultset']['count']