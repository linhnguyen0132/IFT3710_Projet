import pandas as pd

df_sub = pd.read_csv("submission_fr.csv")
df_menacees = pd.read_csv("menacee.csv")

dict_menaces = pd.Series(df_menacees['GBIF_species_name'].values, index=df_menacees['species_id']).to_dict()
set_menaces = set(dict_menaces.keys())

alertes = []
for _, row in df_sub.iterrows():
    predictions = set(map(int, str(row['Predicted_class']).split()))
    
    for m in predictions.intersection(set_menaces):
        alertes.append({"ObservationId": row['ObservationId'], "species_id": m, "Nom_Scientifique": dict_menaces[m]})

pd.DataFrame(alertes).to_csv("especes_menacee_fr.csv", index=False)