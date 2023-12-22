import requests
import pandas as pd
import sqlite3  # Import the sqlite3 library
import os


def fetch_tosdr_data():
    # Initialize an empty list to store cases
    cases_list = []

    # Loop through the available pages
    for page_number in range(1, 4):  # Adjust the range as needed
        # Make the API request
        response = requests.get(
            f"https://api.tosdr.org/case/v1/?page={page_number}").json()

        # Loop through each case and append to list
        for case in response["parameters"]["cases"]:
            cases_list.append({
                'id': case.get('id', None),
                'title': case.get('title', None),
                'description': case.get('description', None),
                'classification': case.get('classification', {}).get('human', None)
            })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(cases_list)
    
    # Save to CSV (optional)
    df.to_csv("../data/tosdr_data.csv", index=False)

    # Save to SQLite database
    save_to_db(df)


# Function to save DataFrame to SQLite database
def save_to_db(df):
    # Create or open SQLite database
    conn = sqlite3.connect('../data/tosdr_data.db')

    # Save DataFrame to SQLite database
    df.to_sql('tosdr_cases', conn, if_exists='replace', index=False)

    # Commit and close connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    fetch_tosdr_data()
