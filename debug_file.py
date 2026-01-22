filename = "data/PureHydration_9Jan.csv"  # Ensure this matches your file name

print(f"--- Inspecting top 15 lines of {filename} ---")
with open(filename, 'r', encoding='latin1') as f:
    for i in range(15):
        print(f"Line {i}: {f.readline().strip()}")