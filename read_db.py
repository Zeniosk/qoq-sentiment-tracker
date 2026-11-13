import pandas as pd
import sqlite3

# Connect to the database
conn = sqlite3.connect("data/processed/sentiment_history.db")

# Use Pandas to run a SQL query and load the results into a DataFrame
# This is standard SQL syntax!
query = "SELECT * FROM qoq_sentiments"
df = pd.read_sql_query(query, conn)

conn.close()

print("\n📊 Current Database Contents:")
print(df.to_markdown(index=False)) # Formats it as a nice markdown table