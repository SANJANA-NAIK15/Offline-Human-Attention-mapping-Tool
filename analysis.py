import pandas as pd

# Load interaction log
df = pd.read_csv("interaction_log.csv")

print(df.head())

# Convert events to attention states
state_map = {
    "Keyboard Activity": 3,
    "Mouse Movement": 3,
    "Mouse Click": 3,
    "Idle": 1
}

states = []

for event in df["Event"]:
    if "Window Switch" in event:
        states.append(2)
    else:
        states.append(state_map.get(event, 2))

df["AttentionState"] = states

fragmentation = 0

for i in range(1, len(df)):
    if df["AttentionState"][i] != df["AttentionState"][i-1]:
        fragmentation += 1

print("Attention shifts detected:", fragmentation)

import numpy as np
import matplotlib.pyplot as plt

states = df["AttentionState"].tolist()

# Repeat the row multiple times so it becomes visible
heatmap = np.tile(states, (10, 1))

plt.figure(figsize=(14,3))

plt.imshow(
    heatmap,
    cmap="RdYlGn",
    aspect="auto",
    vmin=1,
    vmax=3
)

plt.colorbar(label="Attention Level")

plt.yticks([])
plt.xlabel("Time Progression")
plt.title("User Attention Heatmap")

plt.show()