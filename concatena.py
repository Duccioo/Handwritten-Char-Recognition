import pandas as pd

# leggi i file CSV in due DataFrame separati
df1 = pd.read_csv('file1.csv', index_col=0)
df2 = pd.read_csv('result_b8df1bee.csv', index_col=0)

# concatena i due DataFrame lungo l'asse delle righe (default)
concatenated = pd.concat([df1, df2])

# aggiorna l'indice totale in base al numero concatenato dei file
concatenated.index = range(1, len(concatenated)+1)

# scrivi il DataFrame risultante in un nuovo file CSV
concatenated.to_csv('concatenated_file.csv', index_label='Index')
