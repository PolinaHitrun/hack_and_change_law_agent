import pandas as pd

with open("malignant.csv", "r") as f:
    malignant_data = pd.read_csv(f)
df_1 = malignant_data[["category", "text"]]
df_1_clean = df_1.loc[
    df_1["category"].isin(["jailbreak", "act_as"])
].rename(columns={"category": "label"})

with open("prompt_injection_detection_dataset.csv", "r") as f:
    prompt_injection_data = pd.read_csv(f)
df_2 = prompt_injection_data[["category", "text"]]
df_2_clean = df_2.loc[
    df_2["category"].isin(["Instruction Override", "Role-Playing"])
].rename(columns={"category": "label"})

combined_df = pd.concat([df_1_clean, df_2_clean], ignore_index=True)
combined_df.to_csv("combined_dataset.csv", index=False)