import os
import yaml
import pandas as pd

DIR="exp7500"

subdirectories = [
    subdir
    for subdir in os.listdir(DIR)
    if subdir.startswith("ob")
]

df_list = []

for subdir in subdirectories:
    
    with open(f"{DIR}/{subdir}/processed_data.yaml") as f:
        content = yaml.safe_load(f.read())


    for x in content["list"]:
        y = x["robot_to_object_tfm_descriptor"]
        fname = x["fname"]
        item_dict = {f"y{i}": val for i, val in enumerate(y)} 
        item_dict["fname"] = fname
        item_dict["subdir"] = subdir
        df_list.append(item_dict)
        
df = pd.DataFrame(df_list)
print(df)

df.to_csv("data_list.csv", index=None)
