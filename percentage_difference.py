import pandas as pd

def calculate_percentage_difference(value1, value2):
    # Calculate the absolute difference
    absolute_difference = abs(value1 - value2)

    # Calculate the average of the two values
    average_value = (value1 + value2) / 2.0

    # Calculate the percentage difference
    percentage_difference = (absolute_difference / average_value) * 100

    return abs(percentage_difference)

# Read the CSV data into a DataFrame
df = pd.read_csv("results/optimization_results.csv")

# Group the data by Benchmark
grouped_data = df.groupby(["Benchmark"])

# Create an empty DataFrame to store the results
df_results = pd.DataFrame(columns=["Benchmark", "Value_Difference"])

# Iterate through each group
for (benchmark,), group in grouped_data:
    # Extract the BestValue for each algorithm
    base_pso_value = group[group["Algorithm"] == "Base PSO"]["BestValue"].values[0]
    dynamic_weight_value = group[group["Algorithm"] == "PSO with Dynamic Weight"]["BestValue"].values[0]

    # Calculate the difference in BestValue
    value_difference = calculate_percentage_difference(dynamic_weight_value, base_pso_value)
    value_difference = f"{value_difference:.2f}"

    # Append the results to the new DataFrame
    df_results = pd.concat([df_results, pd.DataFrame({"Benchmark": [benchmark], "Value_Difference": [value_difference]})], ignore_index=True)

# Display the results
print(df_results)
