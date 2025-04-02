import requests
import tqdm
from time import sleep
from datetime import datetime

TOKEN = "YkICZrbZAyVnVpjNLCdsorRGDYDwXUoG" # Axelio's Token
BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
HEADERS = {"token": f"{TOKEN}"}

# def list_stations_old(offset = 1):
#     url = f"{BASE_URL}/stations?limit=20&offset={offset}"
#     response = requests.get(url, headers=HEADERS)

#     if response.status_code == 200:
#         return response.json()
    
#     else:
#         print(f"❌ Error: {response.status_code}")


def get_number_of_stations():
    url = f"{BASE_URL}/stations?limit=5"
    response = requests.get(url, headers=HEADERS)
    return response.json()['metadata']['resultset']['count']

def list_stations(limit = 500, stop = -1):
    stations = []
    offset = 1

    n_obs = stop if stop > 0 else get_number_of_stations()

    # Create tqdm object to track progress while downloading data
    with tqdm.tqdm(total = n_obs) as pbar:
        while True:
            try:
                url = f"{BASE_URL}/stations?limit={limit}&offset={offset}"
                response = requests.get(url, headers=HEADERS)

                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        pbar.close()
                        break
                    if stop > 0 and offset > stop:
                        pbar.close()
                        break
                    stations.extend(data['results'])
                    offset += limit
                    pbar.update(limit)

                    # Sleep 0.5 seconds to avoid connection problems
                    sleep(0.5)
            
            except:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Response Code [{response.status_code}] - Let's wait 5 second...")
                sleep(5)
            
            #else:
            #    print(f"❌ Error: {response.json()}")
            #    pbar.close()
            #    break
    
    return stations

def list_locations(limit = 500, stop = -1):
    locations = []
    offset = 1

    while True:
        try:
            url = f"{BASE_URL}/locations?limit={limit}&offset={offset}"
            response = requests.get(url, headers=HEADERS)

            if response.status_code == 200:
                data = response.json()
                if not data:
                    break
                if stop > 0 and offset > stop:
                    break
                locations.extend(data['results'])
                offset += limit
        except:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Response Code [{response.status_code}] - Let's wait 5 second...")
                sleep(5)
            # elif response.status_code == 503:
            #     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Service Temporaly Unavailable. Let's wait 5 second...")
            #     sleep(5)
            # else:
            #     print(f"❌ Error: {response.json()}")
            #     break

    return locations
