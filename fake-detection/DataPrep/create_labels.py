import numpy as np
import pandas as pd 
import os

fake_path = "Images/FakeNoBackground"
real_path = "Images/RealNoBackground"

def create_labels(path):
    image_paths = []
    labels = []
    # Loop through all files in the input folder
    for filename in os.listdir(path):
        image_paths.append(os.path.join(path, filename))
        if "Fake" in path:
            labels.append("Fake")
        else:
            labels.append("Real")
    data = {"Path": image_paths, "Label": labels }
    return pd.DataFrame(data)

def create_all_labels(fake_path, real_path, output_path):
    fake_df = create_labels(fake_path)
    real_df = create_labels(real_path)
    
    df_combined = pd.concat([fake_df, real_df])
    df_combined.to_csv(output_path)
    return(df_combined)



