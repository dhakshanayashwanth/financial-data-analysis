import requests
import json
from datetime import datetime

# Function to fetch the latest links (example implementation)
def fetch_latest_ai_in_finance_links():
    # Example URLs to fetch data from
    urls = [
        "https://example.com/ai-in-finance-1",
        "https://example.com/ai-in-finance-2",
        "https://example.com/ai-in-finance-3",
    ]
    # Mock implementation: replace with actual fetching logic
    links = [{"title": f"AI in Finance Article {i+1}", "url": url} for i, url in enumerate(urls)]
    return links

# Fetch the latest links
latest_links = fetch_latest_ai_in_finance_links()

# Save to a JSON file
with open('latest_ai_in_finance_links.json', 'w') as file:
    json.dump({"last_updated": datetime.now().isoformat(), "links": latest_links}, file)
