
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6, 7])
distance, path = fastdtw(x, y, dist=lambda u, v: euclidean([u], [v]))
print(f"DTW Distance: {distance}")
print(f"Optimal Path: {path}")
plt.figure(figsize=(10, 6))
plt.plot(x, label='Sequence X', marker='o')
plt.plot(y, label='Sequence Y', marker='x')
for (i, j) in path:
    plt.plot([i, j], [x[i], y[j]], 'k--', alpha=0.5)
plt.title(f"DTW Visualization (Distance: {distance:.2f})")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
historical_sequences = [
    np.array([1, 2, 3, 4, 5, 6]),
    np.array([2, 3, 4, 5, 6, 7]),
    np.array([3, 4, 5, 6, 7, 8]),
]
future_values = [7, 8, 9]
current_sequence = np.array([2, 3, 4, 5])
min_distance = float('inf')
best_match_idx = -1
for idx, hist_seq in enumerate(historical_sequences):
    seq1 = [[x] for x in current_sequence]
    seq2 = [[x] for x in hist_seq[:-1]]  # Exclude future value
    print(f"Comparing with sequence {idx}:")
    print(f"Current sequence: {seq1}")
    print(f"Historical sequence: {seq2}")
    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    print(f"Distance to sequence {idx}: {distance}")
    if distance < min_distance:
        min_distance = distance
        best_match_idx = idx
predicted_next_value = future_values[best_match_idx]
print(f"Predicted Next Value: {predicted_next_value}")
