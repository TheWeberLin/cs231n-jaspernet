import pandas as pd
labels = pd.read_csv('species_to_idx.csv', index_col=0)
label_names = list(labels.columns)
print(label_names)