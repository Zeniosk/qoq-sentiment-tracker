import sqlite3
import pandas as pd

DB_PATH = "data/processed/sentiment_history.db"

def compare_qoq():
    conn = sqlite3.connect(DB_PATH)
    
    # Query all data, ordered by quarter and topic
    query = """
        SELECT quarter, name, sentiment, quote_verified 
        FROM qoq_sentiments 
        WHERE company = 'Infosys'
        ORDER BY quarter, name
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No data found in the database.")
        return

    print("\n" + "="*70)
    print(" 📊 INFOSYS QoQ SENTIMENT & TOPIC TRACKER")
    print("="*70)
    
    # Pivot the data to compare Q1 vs Q2 side-by-side
    pivot_df = df.pivot_table(
        index='name', 
        columns='quarter', 
        values='sentiment', 
        aggfunc='first'
    ).reset_index()
    
    # Rename columns for clarity
    pivot_df.columns.name = None
    
    print("\n📈 Topic Sentiment Shift (Q1 vs Q2):")
    print(pivot_df.to_markdown(index=False))
    
    print("\n🔍 Quote Verification Rates:")
    verification_df = df.groupby(['quarter', 'name'])['quote_verified'].mean().reset_index()
    verification_df['quote_verified'] = verification_df['quote_verified'].apply(lambda x: f"{int(x*100)}%")
    verification_df.columns = ['Quarter', 'Topic', 'Verified']
    print(verification_df.to_markdown(index=False))
    
    print("\n💡 Key Takeaway:")
    print("This structured data can now be fed directly into a frontend dashboard")
    print("or used to trigger alerts when sentiment drops on critical topics.")

if __name__ == "__main__":
    compare_qoq()