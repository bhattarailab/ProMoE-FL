"""Map PadChest label annotations to the MIMIC label space and filter projection views.

PadChest contains more fine-grained clinical labels than MIMIC. This script harmonizes
PadChest annotations into MIMIC-equivalent labels using a mapping created by a clinical expert.
The resulting dataset can then be used for experiments that require the MIMIC label space.

The provided mappings JSON file should map PadChest labels to MIMIC labels. Labels that do not
have a valid mapping are marked as "unmatched" and can be filtered out.

This script also filters the data to retain only images with AP and PA projection views, which are the most common and clinically relevant.
"""

import argparse
import pandas as pd
import re
import json
import argparse

def return_unique_labels(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    print(f"Read {len(df)} rows from {file_path}")

    # Check if 'Labels' column exists
    if 'Labels' not in df.columns:
        raise ValueError("The specified column 'Labels' does not exist in the CSV file.")

    # Extract unique labels from the 'Labels' column
    unique_labels = set()
    count = 0
    for labels in df['Labels'].dropna():
        # labels look like list but is a string: "['No Finding', 'Support Devices']"
        items = labels.strip("[]").replace("'", "").split(", ")
        for label in items:
            unique_labels.add(label.strip())

    # # Convert the set to a list and count the number of unique labels
    unique_labels_list = list(unique_labels)


    return unique_labels_list, len(unique_labels_list)

def get_unique_projections(df):
    return df['Projection'].unique()

def retain_only_valid_projections(df, valid_projections):
    return df[df['Projection'].isin(valid_projections)]


def change_labels_to_mimic(filepath, to_mimic_labels):
    df = pd.read_csv(filepath)

    if 'Labels' not in df.columns:
        raise ValueError("The specified column 'Labels' does not exist in the CSV file.")

    def map_labels(label_string):
        if pd.isna(label_string):
            return label_string
        items = label_string.strip("[]").replace("'", "").split(", ")
        mapped_labels = set()
        no_mapping_found = False
        for item in items:
            item = item.strip()
            if item in to_mimic_labels and to_mimic_labels[item] != "unmatched":
                mapped_labels.add(to_mimic_labels[item])
            else:
                no_mapping_found = True
                mapped_labels.add("unmatched")
                break
        return list(mapped_labels)

    df = df.dropna(subset=['Labels'])
    df['Mapped_Labels'] = df['Labels'].apply(map_labels)
    return df

def main(args):
    file_path = args.csv
    unique_labels, count = return_unique_labels(file_path)
    print(f"Unique Labels: {unique_labels}")
    print(f"Count of Unique Labels: {count}")

    labels_mimic = ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'enlarged cardiomediastinum', 'fracture', 'lung lesion', 'lung opacity', 'no finding', 'pleural effusion', 'pleural other', 'pneumonia', 'pneumothorax', 'support devices']

    mappings_path = args.mappings
    # get json from the prompt response and print matched labels
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)

    to_mimic_labels = dict()
    for key, value in mappings.items():
        to_mimic_labels[key] = value[0] #ignoring the notes

    df_mapped = change_labels_to_mimic(file_path, to_mimic_labels)

    print("\nDataFrame with Mapped Labels:\n")
    print(df_mapped.shape)

    #  filter rows to remove labels mapped to 'unmatched'
    df_mapped_filtered = df_mapped[df_mapped['Mapped_Labels'].apply(lambda x: 'unmatched' not in x)]
    print("\nDataFrame with Mapped Labels (Filtered):\n")
    print(df_mapped_filtered.shape)
    

    print(f"Original data size: {len(df_mapped_filtered)}")
    unique_projections = get_unique_projections(df_mapped_filtered)
    print(f"Unique projections found: {unique_projections}")
    
    valid_projections = ['PA', 'AP_horizontal']
    df_cleaned = retain_only_valid_projections(df_mapped_filtered, valid_projections)
    print(f"Cleaned data size: {len(df_cleaned)}")

    # export new df to csv
    output_path = args.output
    df_cleaned.to_csv(output_path, index=False) 

def parse_args():
    parser = argparse.ArgumentParser(
        description= "Map Padchest labels to equivalent MIMIC labels and retain only AP and PA projection views"
    )
    parser.add_argument( 
        '--csv',
        type=str,
        required=True,
        help='Path to your PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv file'
    )
    parser.add_argument(
        '--mappings',
        type=str,
        required=True,
        help='Path to your category_mappings.json file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Name of your output csv file. Eg. PADCHEST_mapped_APPA.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)



    
    
