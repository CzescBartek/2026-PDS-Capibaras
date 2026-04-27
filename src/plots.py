import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("/Users/laurabetev/Desktop/ITU/Projects in Data science /data_science_project_capibaras/result/predictions/predictions.csv")

# Check the structure
print(df.head())

# Plot histogram of probabilities
plt.figure()
plt.hist(df["probability"], 
         bins=20,
         edgecolor='black',
         linewidth=1.0)

# Labels and title
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.title("Histogram of Image Probabilities")

# Show plot
plt.savefig('../result/figures/histogram_probabilities.png')
plt.show()
plt.close()