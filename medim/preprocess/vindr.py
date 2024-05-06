import numpy as np
import pandas as pd
from medim.constants import *
from sklearn.model_selection import train_test_split

np.random.seed(0)

def preprocess_vindr_data():
    try:
        raw_train_df = pd.read_csv(VinDr_ORIGINAL_TRAIN)
    except:
        raise Exception(
            "Please make sure the the chexpert dataset is \
            stored at {VinDr_DATA_DIR}"
        )

    # Grouping the data by 'image_id' and taking the maximum value for each label
    aggregated_df = raw_train_df.groupby('image_id').max().reset_index()
    # Dropping the 'rad_id' column as it's not relevant after aggregation
    aggregated_df.drop(columns='rad_id', inplace=True)
    aggregated_df['image_id'] = aggregated_df['image_id'].apply(lambda x: 'train/' + x + '.png')

    train_df, valid_df = train_test_split(
        aggregated_df, train_size=0.8, random_state=0)

    print(f"Number of train samples: {len(train_df)}")
    print(f"Number of valid samples: {len(valid_df)}")

    aggregated_df.to_csv(VinDr_AllTRAIN_CSV)
    train_df.to_csv(VinDr_TRAIN_CSV)
    valid_df.to_csv(VinDr_VALID_CSV)

    test_df = pd.read_csv(VinDr_ORIGINAL_TEST)
    test_df['image_id'] = test_df['image_id'].apply(lambda x: 'test/' + x + '.png')

    test_df.to_csv(VinDr_TEST_CSV)


if __name__ == "__main__":
    preprocess_vindr_data()