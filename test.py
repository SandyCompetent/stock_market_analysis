import requests
from bs4 import BeautifulSoup
import re
import pandas as pd


def decode_secret_message(url: str):
    """
    Retrieves a published Google Doc, parses character coordinate data from it,
    and prints a 2D grid that reveals a secret message.

    Args:
        url: The string URL of the public Google Doc.
    """
    try:
        # 1. Retrieve the data from the URL
        response = requests.get(url, timeout=10)

        # Use BeautifulSoup to extract the clean text content
        soup = BeautifulSoup(response.content, 'html.parser')


        table = soup.find('table')
        table_rows = table.find_all('tr')

        # print(table)
        # print(table_rows)


        fortune = []
        for tr in table_rows:
            td = tr.find_all('td')
            row = [i.text for i in td]
            fortune.append(row)

        fortune = pd.DataFrame(fortune)
        fortune.drop(0, axis=0, inplace=True)
        print(fortune.head())

        try:
            result = fortune[(fortune[0] == 0) & (fortune[2] == 2)][1].item()

            # Print the final result
            print(f"DataFrame:\n{fortune}\n")
            print(f"Result: {result}")

        except ValueError:
            print("No single row matched the criteria.")

        # Convert extracted strings to integers and store all points
        # points = []
        # max_x = 0
        # max_y = 0
        # for x_str, char, y_str in matches:
        #     x, y = int(x_str), int(y_str)
        #     points.append({'x': x, 'y': y, 'char': char})
        #     # Keep track of the grid dimensions
        #     if x > max_x:
        #         max_x = x
        #     if y > max_y:
        #         max_y = y
        #
        # # 3. Build the 2D grid
        # # The grid dimensions are determined by the max coordinates found
        # grid_width = max_x + 1
        # grid_height = max_y + 1
        #
        # # Initialize the grid with space characters
        # grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
        #
        # # Populate the grid with the characters at their specified coordinates
        # for point in points:
        #     grid[point['y']][point['x']] = point['char']
        #
        # # 4. Print the final grid
        # for row in grid:
        #     print(''.join(row))

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the URL: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Let's run it with the test URL provided ---
test_url = "https://docs.google.com/document/d/e/2PACX-1vRMx5YQlZNa3ra8dYYxmv-QIQ3YJe8tbI3kqcuC7lQiZm-CSEznKfN_HYNSpoXcZIV3Y_O3YoUB1ecq/pub"

print("Decoding the message from the test URL...")
decode_secret_message(test_url)