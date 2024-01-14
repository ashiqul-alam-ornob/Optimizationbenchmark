from typing import List
import csv
import os


def write_to_csv(file_path: str, header: List[str], data: List):
    """
    Write data to a CSV file.
    It will create the folder structure if it does not exist previously.

    Parameters:
    - file_path (str): The path to the CSV file.
    - header (list): List of strings representing the header row in the CSV file.
    - data (list): List of lists representing the data rows in the CSV file.
    """
    # Create the folder structure if it doesn't exist
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        writer.writerow(header)

        # Write the data rows
        writer.writerows(data)
