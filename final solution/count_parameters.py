import os
import re
from collections import defaultdict

# Mappa, ahol a képek találhatók
input_folder = "../inpainted_test_results/Rat1 - 14.42.36-14.44.34[M][0@0][0]/2nd run másolata"

# Regular expression for extracting parameters
filename_pattern = re.compile(r"frame_\d+_op(\d+)_di(\d+)_it(\d+)_r(\d+)_(TELEA|NS).png")

# Számlálók létrehozása
opening_counts = defaultdict(int)
dilate_counts = defaultdict(int)
iteration_counts = defaultdict(int)
radius_counts = defaultdict(int)
method_counts = defaultdict(int)

# Mappa bejárása
for filename in os.listdir(input_folder):
    match = filename_pattern.match(filename)
    if match:
        opening, dilate, iterations, radius, method = match.groups()
        opening_counts[int(opening)] += 1
        dilate_counts[int(dilate)] += 1
        iteration_counts[int(iterations)] += 1
        radius_counts[int(radius)] += 1
        method_counts[method] += 1

# Eredmények kiírása
print("Összesített paraméter statisztika:")
print("Opening kernel eloszlás:")
for key, count in sorted(opening_counts.items()):
    print(f"Opening: {key} -> {count} db")

print("\nDilate kernel eloszlás:")
for key, count in sorted(dilate_counts.items()):
    print(f"Dilate: {key} -> {count} db")

print("\nIterations eloszlás:")
for key, count in sorted(iteration_counts.items()):
    print(f"Iterations: {key} -> {count} db")

print("\nInpaint radius eloszlás:")
for key, count in sorted(radius_counts.items()):
    print(f"Radius: {key} -> {count} db")

print("\nInpaint method eloszlás:")
for key, count in method_counts.items():
    print(f"Method: {key} -> {count} db")
