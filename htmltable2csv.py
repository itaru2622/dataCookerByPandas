#!/usr/bin/env python3

import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import sys


def scrape_table(url: str) -> (list[list[str]], list[str]):
    """
    Scrape the table from a web page and extract the list of rows and the header.

    Args:
        url (str): The URL of the web page to scrape.

    Returns:
        Tuple[list[list[str]], list[str]]: The list of scraped table rows and the header list.
    """

    # placeholder of return data
    header = []
    rows = []

    # get content of web page
    response = requests.get(url)
    response.encoding = 'utf-8'      # specify encoding
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table') # find table in response.

    rowspan_values = [] # holder for rowspan info, to fill exact value later...

    # Extract the rows from the table
    for r, row in enumerate(table.find_all('tr')): # for each row
        cells = row.find_all(['th', 'td'])         #     get both of <th> and <td>.
        if not cells:
            continue
        row_data = []
        for i, cell in enumerate(cells):           # for each cell
            if r == 0:
                header.append(cell.text.strip())   # store data as header, for first line.
            else:
                if 'rowspan' in cell.attrs:        # when rowspan,  remember it for later use.
                    rowspan = int(cell['rowspan'])
                    value = cell.text.strip()
                    rowspan_values.append({'col': i, 'row': r, 'value': value, 'count': rowspan})
                row_data.append(cell.text.strip())

        if row_data is not []:
            rows.append(row_data)

    # fill exact value when ommited by rowspan.
    for d in rowspan_values:                      # for each rowspan info.
        starts = d.get('row')+1                   #  +1: row number needs to 'fill' omitted value. in the first row, value exiists in the cell
        col = d.get('col')                        #  the col number from begining.
        count = d.get('count')                    #  number of repeats which web page author specified by rowspan=N
        v = d.get('value')
        for r in range(starts, starts + count - 1):  # repeat for num of 'rowspan'
            cur = rows[r]
            cur.insert(col, v)
            rows[r] = cur

    rows = [row for row in rows if row] # remove blank line, when exists.
    return rows, header


def write_csv(output_file: str, rows: list[list[str]], header: list[str]) -> None:
    """
    Write the list of rows and the header to a CSV file.

    Args:
        output_file (str): The path to the output CSV file.
        rows (list[list[str]]): The list of table rows.
        header (list[str]): The header list.

    Returns:
        None
    """

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if header is not []:
            writer.writerow(header)
        writer.writerows(rows)


if __name__ == '__main__':

    import                    argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url',    type=str,                    default='https://uub.jp/opm/ml_homepage.html',  help='web page which has table.')               # input URL
    parser.add_argument('-o', '--output', type=str,                    default='/dev/stdout',                          help='path of output in csv. default:stdout')   # output file path
    args = parser.parse_args()
    print(args, file=sys.stderr)

    # Scrape the table from specified page.
    rows, header = scrape_table(args.url)

    # output data, in csv.
    write_csv(args.output, rows, header)
