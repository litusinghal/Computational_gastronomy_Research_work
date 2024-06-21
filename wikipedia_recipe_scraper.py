import pandas as pd
import requests
import string
from bs4 import BeautifulSoup

# Define the path to your Excel file
file_path = "recipe.xlsx"

# Read the Excel file
df = pd.read_excel(file_path)  # Update sheet name if needed
dish_names = df['Recipe_title'].tolist()

def wikibot(keyword):
    try:
        u_i = string.capwords(keyword)
        lists = u_i.split()
        word = "_".join(lists)
        url = "https://en.wikipedia.org/wiki/" + word
        url_open = requests.get(url)
        
        if url_open.status_code != 200:
            print("Error: Could not fetch the Wikipedia page for", keyword)
            return
        
        soup = BeautifulSoup(url_open.content, 'html.parser')
        details = soup.find_all('table', {'class': 'infobox'})
        
        for i in details:
            h = i.find_all('tr')
            for j in h:
                heading = j.find_all('th')
                detail = j.find_all('td')
                if heading is not None and detail is not None:
                    for x, y in zip(heading, detail):
                        print("{}  ::  {}".format(x.text, y.text))
                        print("------------------")
        
        # Print at max 3 paragraphs
        paragraphs = soup.find_all('p')
        for i in range(0, 3):
            if i < len(paragraphs):
                print(paragraphs[i].text)
    except Exception as e:
        print(f"An error occurred: {e}")

# Run for the entire list and write the data to a file
for dish_name in dish_names:
    print(f"Fetching information for: {dish_name}")
    wikibot(dish_name)
    print("\n=====================\n")

# Example of running for a specific dish
# wikibot(dish_names[7])
