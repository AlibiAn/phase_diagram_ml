import json
import pandas as pd
import numpy as np

def extract_temps_at_compositions(liquidus_line, target_comps = [10, 20, 30, 40,
                                                                 50 ,60, 70, 80, 90]):
    # Process incomplete samples with NaN
    if not liquidus_line or len(liquidus_line) < 2:
        return {f'liquidus_temp_at_{comp}pct': np.nan for comp in target_comps}
    
    # Collect the compositions and temperatures into numpy arrays
    compositions = np.array([point[0] for point in liquidus_line])
    temperatures = np.array([point[1] for point in liquidus_line])

    # Sort by Compositions
    sorted_indices = np.argsort(compositions)
    compositions = compositions[sorted_indices]
    temperatures = temperatures[sorted_indices]

    result = {}

    for comp in target_comps:
        # if Composition isn't covered in the Phase Diagram Sample
        if comp < compositions.min() or comp > compositions.max():
            result[f'liquidus_temp_at_{comp}pct'] = np.nan
        else:
            # Since the training data isn't given in discrete values, interpolate
            temp = np.interp(comp, compositions, temperatures)
            result[f'liquidus_temp_at_{comp}pct'] = temp
    return result

with open('selected_phase_diagrams.json', 'r') as f:
    file_paths = json.load(f)

all_results = []
errors = 0

for file_path in file_paths:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        entry_id = data.get('entry')
        liquidus_line = data.get('liquidus_line', [])

        temps = extract_temps_at_compositions(liquidus_line)
        result = {'entry_id': entry_id, **temps}
        all_results.append(result)
    except Exception as e:
        errors += 1

print(f"Processes {len(all_results)} files with {errors} errors")

# Create a Pandas DataFrame
liquidus_df = pd.DataFrame(all_results)

# Merge with existing csv file with enginered features
features_df = pd.read_csv('phase_diagram_features_with_metadata.csv')
merged_df = features_df.merge(liquidus_df, on = 'entry_id', how = 'left')

merged_df.to_csv('features_with_liquidus_curves.csv', index = False)

print(f"Saved, Shape: {merged_df.shape}")
