import csv
import os
import pandas as pd
import ast
import random
import numpy as np
from pdb import set_trace

num_tiles_per_wsi = 1000 # number of tiles to keep per WSI (tradeoff between information content and compute requirement)

base_dir = '/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/'  # for Polaris
tiles_dir = base_dir + 'TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/256px_9.9x/'
# added tiles from 'TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/256px_9.9x_clean_from_penmarks/' to above
tiles_dir = base_dir + 'TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_20X/tiles/256px_128um/'  # 20X magnification

# manually removed the tiles with penmarks
# replaced the dirs containing the tiles with penmarks with their clean versions

tiles_dir = base_dir + 'TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_20X_1000tiles/tiles/256px_128um/'  # 20X magnification with 1000 tiles per WSI

# csv file with the mapped data
output_csv_path = base_dir + './mapped_data_31Jan_1000tiles.csv'
# json file with the mapped data
output_json_path = base_dir + './mapped_data_31Jan_1000tiles.json'

rnaseq_clinical_df = pd.read_csv('./rnaseq_clinical_28Jan.csv') # generated by tcga_luad.heidi_rnaseq_xenabrowser_clinical.ipynb
png_files_dict = {}
# set_trace()
# Get the list of the extracted tiles (in png/jpg format) for each sample
count_tcga_wsi = 0

# use only num_tiles_per_wsi (currently set to 200) randomly chosen tiles from each WSI
for tile_dir in os.listdir(tiles_dir):
    if "TCGA" in tile_dir:  # skipping the "combined_tiles" directory
        count_tcga_wsi += 1
        count_tiles = 0
        for filename in os.listdir(os.path.join(tiles_dir, tile_dir)):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                count_tiles += 1
                tcga_id = '-'.join(filename.split('-')[:3])
                if tcga_id in png_files_dict:
                    png_files_dict[tcga_id].append(filename)
                else:
                    png_files_dict[tcga_id] = [filename]
        print("ID: ", tile_dir, " count_tiles: ", count_tiles)

# limit each key to have only 'num_tiles_per_wsi' files in val
count_gt_tiles_per_wsi = 0
for tcga_id in png_files_dict:
    if len(png_files_dict[tcga_id]) > num_tiles_per_wsi:
        count_gt_tiles_per_wsi += 1
        png_files_dict[tcga_id] = random.sample(png_files_dict[tcga_id], num_tiles_per_wsi)

# print("Number of WSIs with more than 200 tiles: ", count_gt_tiles_per_wsi)

# find keys with values that are not equal to 'num_tiles_per_wsi' (=200 or 1000)
keys_with_different_length = {key: len(value) for key, value in png_files_dict.items() if len(value) != num_tiles_per_wsi}

if keys_with_different_length:
    for key, length in keys_with_different_length.items():
        print(f"Key: {key}, Length of values: {length}")
else:
    print(f"All keys have {num_tiles_per_wsi} entries")

# extract 'days_to_death' , 'days_to_last_followup', and 'vital_status' into lists for each column
data_clinical_df = pd.DataFrame({
    'sample_id': rnaseq_clinical_df['sample_id'],
    'days_to_death': rnaseq_clinical_df['days_to_death.demographic'],
    'days_to_last_followup': rnaseq_clinical_df['days_to_last_follow_up.diagnoses'],
    'event_occurred': rnaseq_clinical_df['vital_status.demographic']
}).set_index('sample_id')

combined_df = data_clinical_df.copy()

combined_df['tiles'] = [[] for _ in range(len(combined_df))]

for idx in combined_df.index:
    print(idx)
    combined_df.at[idx, 'tiles'] = png_files_dict.get(idx, [])

# add the rnaseq data
# create a mapping from sample_id to gene_exps
gene_exps_mapping = rnaseq_clinical_df.set_index('sample_id')['gene_exps'].to_dict()

# add the gene_exps data to combined_df
combined_df['rnaseq_data'] = combined_df.index.map(gene_exps_mapping)

empty_tiles = combined_df[combined_df['tiles'].apply(lambda x: len(x) == 0)]
print(f"Rows with empty tiles: {len(empty_tiles)}")
print(empty_tiles)

# compute the number of tiles per sample
combined_df['num_tiles'] = combined_df['tiles'].apply(len)
wsis_with_inconsistent_tiles = combined_df[combined_df['num_tiles'] != num_tiles_per_wsi]
print("\nRows with inconsistent number of tiles per WSI:")
print(wsis_with_inconsistent_tiles)

# set_trace()

empty_rnaseq = combined_df[combined_df['rnaseq_data'].apply(lambda x: len(x) == 0)]
print(f"\nRows with empty rnaseq_data: {len(empty_rnaseq)}")
print(empty_rnaseq)

# check for missing survival times
missing_survival_time = combined_df[
    (combined_df['days_to_death'].isna()) &
    (combined_df['days_to_last_followup'].isna())
]
print(f"Rows with missing survival time: {len(missing_survival_time)}")
if len(missing_survival_time) > 0:
    print("Problematic rows:")
    print(missing_survival_time)

# check for missing event indicators
missing_event = combined_df[combined_df['event_occurred'].isna()]
print(f"Rows with missing event indicator: {len(missing_event)}")
if len(missing_event) > 0:
    print("Problematic rows:")
    print(missing_event)

# dead patients with missing or zero days_to_death
dead_inconsistencies = combined_df[
    (combined_df['event_occurred'] == 'Dead') &
    (combined_df['days_to_death'].isna() | (combined_df['days_to_death'] == 0))
]
print(f"Rows with inconsistent data for dead patients: {len(dead_inconsistencies)}")
if len(dead_inconsistencies) > 0:
    print("Problematic rows:")
    print(dead_inconsistencies)

# alive patients with missing or zero days_to_last_followup
alive_inconsistencies = combined_df[
    (combined_df['event_occurred'] == 'Alive') &
    (combined_df['days_to_last_followup'].isna() | (combined_df['days_to_last_followup'] == 0))
]
print(f"Rows with inconsistent data for alive patients: {len(alive_inconsistencies)}")
if len(alive_inconsistencies) > 0:
    print("Problematic rows:")
    print(alive_inconsistencies)

# number of events
num_events = combined_df[combined_df['event_occurred'] == 'Dead'].shape[0]
print(f"Number of events (deaths): {num_events}")

# event rate
event_rate = num_events / combined_df.shape[0]
print(f"Event rate: {event_rate:.2%}")

# define problematic rows
problematic_conditions = (
    combined_df['tiles'].apply(lambda x: len(x) == 0) |  # empty tiles
    # (combined_df['num_tiles'] != num_tiles_per_wsi) | # remove samples where number of tiles is not equal to num_tiles_per_wsi
    combined_df['rnaseq_data'].apply(lambda x: len(x) == 0) |  # empty rnaseq_data
    (combined_df['days_to_death'].isna() & combined_df['days_to_last_followup'].isna()) |  # missing survival time
    combined_df['event_occurred'].isna() |  # missing event indicator
    ((combined_df['event_occurred'] == 'Dead') &
     (combined_df['days_to_death'].isna() | (combined_df['days_to_death'] == 0))) |  # dead with invalid days_to_death
    ((combined_df['event_occurred'] == 'Alive') &
     (combined_df['days_to_last_followup'].isna() | (combined_df['days_to_last_followup'] == 0)))  # alive with invalid days_to_last_followup
)

# add smoking information


# remove rows that meet any of the problematic conditions
cleaned_combined_df = combined_df[~problematic_conditions] #.reset_index(drop=True)

# remove the 'num_tiles' column before saving
cleaned_combined_df = cleaned_combined_df.drop(columns=['num_tiles'])

print(f"Original dataframe rows: {len(combined_df)}")
print(f"Cleaned dataframe rows: {len(cleaned_combined_df)}")

set_trace()
cleaned_combined_df.to_csv(output_csv_path)
cleaned_combined_df.to_json(output_json_path, orient='index')
print(f"filtered data has been written to {output_csv_path} and {output_json_path}.")

set_trace()


