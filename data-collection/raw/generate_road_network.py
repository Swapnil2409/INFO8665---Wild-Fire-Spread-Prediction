import networkx as nx
import pandas as pd
import os
import pickle

# ✅ Define correct CSV path
csv_path = os.path.join(os.path.dirname(__file__), "synthetic_wildfire_data.csv")

# ✅ Ensure CSV file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

# ✅ Load wildfire data
df = pd.read_csv(csv_path)

# ✅ Create road network graph
road_network = nx.Graph()

# ✅ Add nodes based on latitude & longitude
for _, row in df.iterrows():
    node = (row["latitude"], row["longitude"])
    road_network.add_node(node)

# ✅ Connect nearby nodes with edges (Using sample distance-based connection)
for i in range(len(df) - 1):
    node1 = (df.iloc[i]["latitude"], df.iloc[i]["longitude"])
    node2 = (df.iloc[i + 1]["latitude"], df.iloc[i + 1]["longitude"])
    risk = abs(df.iloc[i]["temperature"] - df.iloc[i + 1]["temperature"]) / 50  # Sample risk calculation
    road_network.add_edge(node1, node2, risk=risk)

# ✅ Save the graph in `data-collection/processed/`
save_dir = os.path.join(os.path.dirname(__file__), "../processed")
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "road_network.gpickle")

with open(save_path, "wb") as f:
    pickle.dump(road_network, f)

print(f"✅ Road Network Graph Saved at: {save_path}")
