import os
import numpy as np
import pandas as pd

# ✅ Define base location
BASE_LATITUDE = 37.7749
BASE_LONGITUDE = -122.4194

# ✅ Define feature ranges
temperature_range = (20, 45)  # °C
humidity_range = (10, 90)  # %
wind_speed_range = (0, 50)  # km/h
wind_direction_range = (0, 360)  # degrees
precipitation_range = (0, 50)  # mm
drought_index_range = (0, 5)  # Scale 0-5
ndvi_range = (-1, 1)  # Vegetation Index
erc_range = (0, 100)  # Energy Release Component
population_density_range = (0, 1000)  # People per km²
elevation_range = (0, 3000)  # meters

# ✅ Generate synthetic wildfire dataset (4096 rows)
num_rows = 4096
data = {
    "latitude": np.random.normal(BASE_LATITUDE, 0.01, num_rows),  # Nearby variations
    "longitude": np.random.normal(BASE_LONGITUDE, 0.01, num_rows),
    "temperature": np.random.uniform(*temperature_range, num_rows),
    "humidity": np.random.uniform(*humidity_range, num_rows),
    "wind_speed": np.random.uniform(*wind_speed_range, num_rows),
    "wind_direction": np.random.uniform(*wind_direction_range, num_rows),
    "precipitation": np.random.uniform(*precipitation_range, num_rows),
    "drought_index": np.random.uniform(*drought_index_range, num_rows),
    "ndvi": np.random.uniform(*ndvi_range, num_rows),
    "erc": np.random.uniform(*erc_range, num_rows),
    "population_density": np.random.uniform(*population_density_range, num_rows),
    "elevation": np.random.uniform(*elevation_range, num_rows),
}

df = pd.DataFrame(data)

# ✅ Save CSV to the correct path
csv_path = os.path.join(os.path.dirname(__file__), "synthetic_wildfire_data.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Wildfire dataset generated: {csv_path}")
