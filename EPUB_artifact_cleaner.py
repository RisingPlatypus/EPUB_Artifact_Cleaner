"""
Script for deleting E-book (EPUB) artifacts like hard coded page numbers (and in future iterations watermarks etc.).
Those artifacts can happen when pages are scanned and the E-book is created by using of a text identifier, they can
also happen when a file is converted, or when the creator of the file is not well versed.


Features (v.0.9):
- Creates new Book without page number artifacts.
    - Extraction of numbers in text.
    - Identification of sequences.
    - Analysis of sequences for age number behavior.
    - Sanity check (leaves numbers alone if it's not entirely certain).
    - Deletion of page numbers from the text.
    - File overwrite protection


Usage:
Run this script, choose the EPUB file, wait.
"""


import zipfile
import xml.etree.ElementTree as ET
import os
import sys
import re
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog

def select_file():
    """
    Open a file dialog to select a file using PyQt5.
    Returns:
        - str: The selected file path or None if no file is selected.
    """
    app = QApplication([])  # Create the application context
    file_path, _ = QFileDialog.getOpenFileName(
        None, "Select a File", "", "EPUB Files (*.epub);;All Files (*)"
    )
    return file_path

def find_content_opf(epub_zip):
    """
    Find and open the content.opf file inside the epub file, which usually defines the order of the files.

    Args:
        - epub_zip: The EPUB file loaded as a zip file.

    Returns:
        - file: the content.opf file.
    """
    # Search for content.opf in the EPUB archive
    for file in epub_zip.namelist():
        if file.endswith('content.opf'):
            return file
    return None


def get_html_order_from_opf(epub_zip):
    """
    Get the correct order of the html files in the EPUB file.

    Args:
        - epub_zip: The EPUB file loaded as a zip file.

    Returns:
        - html_order
    """
    # Find content.opf file
    opf_file = find_content_opf(epub_zip)
    if not opf_file:
        raise ValueError("content.opf file not found in the EPUB archive")

    # Read the content.opf file
    opf_data = epub_zip.read(opf_file)

    # Parse the content.opf file to extract the HTML files in the correct order
    tree = ET.ElementTree(ET.fromstring(opf_data))
    root = tree.getroot()

    # Find all itemref elements in the spine section
    spine = root.find('{http://www.idpf.org/2007/opf}spine')
    if spine is not None:
        html_order = []
        for itemref in spine.findall('{http://www.idpf.org/2007/opf}itemref'):
            # Get the idref, which refers to an item in the manifest
            item_id = itemref.attrib['idref']
            # Find the corresponding item in the manifest to get the file name
            manifest = root.find('{http://www.idpf.org/2007/opf}manifest')
            for item in manifest.findall('{http://www.idpf.org/2007/opf}item'):
                if item.attrib['id'] == item_id and item.attrib['media-type'] == 'application/xhtml+xml':
                    html_order.append(item.attrib['href'])
        return html_order
    return []


def sort_files(epub_path):
    """
    Create a reproducible order of html files of the EPUB files, that can be referenced throughout the script.

    Args:
        - epub_path: file path of the file on the system.

    Returns:
        - sorted_file_list: sorted list of other files (alphabetically sorted) + html files, sorted by Ebook structure.
    """
    with zipfile.ZipFile(epub_path, 'r') as epub_zip:
        # List all files in the EPUB
        file_list = epub_zip.namelist()

        # Extract the order of HTML files from content.opf
        html_order = get_html_order_from_opf(epub_zip)

        # Split the files into different types
        html_files = [file for file in file_list if file.endswith('.html') or file.endswith('.xhtml')]
        other_files = [file for file in file_list if not (file.endswith('.html') or file.endswith('.xhtml'))]

        # Sort other file types alphabetically
        other_files.sort()

        # Sort HTML files based on the order specified in the spine
        html_files_sorted = sorted(html_files, key=lambda x: html_order.index(x) if x in html_order else float('inf'))

        # Combine the sorted lists: other files first, then sorted HTML files
        sorted_file_list = other_files + html_files_sorted

        return sorted_file_list


# Open the EPUB file, load all HTML files, and extract numbers with their positions and file index
def load_all_html_with_overall_position(epub_path):
    """
    Open the EPUB file, load all HTML files in order, and extract numbers featured in the text. Apply basic filtering.

    Args:
        - epub_path: file path of the file on the system.

    Returns:
        - filtered_matrix: Overview over numbers in text
            columns: 1. extracted number, 2. position in file, 3. overall position, 4. file index.
        - cumulative_offset: Overall length of total html code in EPUB file.
    """
    # Open file
    with zipfile.ZipFile(epub_path, 'r') as epub_zip:
        # List and sort all files in the EPUB
        file_list = sort_files(epub_path)
        # Filter for HTML files (you can adjust this filter as needed)
        html_files = [file for file in file_list if file.endswith('.html')]
        all_number_data = []
        cumulative_offset = 0  # Keeps track of the overall position offset for each file

        # Loop through each HTML file and extract numbers
        for file_index, html_filename in enumerate(html_files):

            with epub_zip.open(html_filename) as html_file:
                html_content = html_file.read().decode('utf-8')

                # Find the <body> tag's start and end positions (numbers outside the body are irrelevant)
                body_start_match = re.search(r'<body[^>]*>', html_content, re.IGNORECASE)
                body_end_match = re.search(r'</body>', html_content, re.IGNORECASE)

                # If <body> exists, get its positions; otherwise, skip this file
                if body_start_match and body_end_match:
                    body_start = body_start_match.end()  # Position after the opening <body> tag
                    body_end = body_end_match.start()   # Position before the closing </body> tag

                    # Find all numbers with their start positions
                    matches = [
                        (float(match.group()), match.start(), match.start() + cumulative_offset, file_index)
                        for match in re.finditer(r'\b\d+(\.\d+)?\b', html_content)
                    ]

                    # Filter matches to only include those within the <body> range
                    filtered_matches = [
                        match for match in matches if body_start <= match[1] < body_end
                    ]

                    # Add the filtered matches to the list for all files
                    all_number_data.extend(filtered_matches)

                # Update the cumulative offset with the length of the current HTML content
                cumulative_offset += len(html_content)

    # Convert to a NumPy array with columns: [number, position in file, overall position, file index]
    matrix = np.array(all_number_data, dtype=object)  # Use 'object' dtype to allow mixed types initially

    # Filter out rows where the number is not an integer
    filtered_matrix = np.array(
        [row for row in matrix if float(row[0]).is_integer()],  # Check if the number is a whole number
        dtype=object
    )

    if filtered_matrix.size > 0:  # Check if the filtered matrix is not empty
        filtered_matrix[:, 0] = filtered_matrix[:, 0].astype(int)  # Convert the number column to integers
    else:
        print("Filtered matrix is empty. No integers were found.")
        return np.empty((0, 4), dtype=object)  # Return an empty array with the expected shape

    return filtered_matrix, cumulative_offset

# Function to check for sequences (page numbers, chapter numbers, unrelated)
def identify_sequences(matrix):
    """
    Scans all the extracted numbers for possible sequences.

    Args:
        - matrix: Overview over numbers in text
            Columns: 1. extracted number, 2. position in file, 3. overall position, 4. file index.

    Returns:
        - matrix: Overview over numbers in text
            Columns: 1. extracted number, 2. position in file, 3. overall position, 4. file index, 5. seq. ID's
    """

    print("CHECKING FOR PAGE NUMBER ARTIFACTS")
    print("Step 1: finding hardcoded number sequences in the text.")

    sequence_id = 0  # Starting sequence identifier
    matrix_length = len(matrix)

    # Loop for start of sequence
    for i in range(matrix_length):
        if not float(matrix[i, 0]).is_integer() or not np.isnan(matrix[i, 4]):  # Already part of a sequence
            continue

        sequence = [matrix[i, 0]]  # Start a new sequence
        s_idx = [i]

        # Check for consecutive numbers, allowing for interruptions
        v_active = matrix[i, 0]
        for j in range(i + 1, matrix_length):
            if not float(matrix[j, 0]).is_integer() or not np.isnan(matrix[j, 4]):
                continue  # Skip if already part of a sequence
            elif matrix[j, 0] == v_active + 1:
                sequence.append(matrix[j, 0])
                s_idx.append(j)
                v_active += 1  # Update active value for next expected

        # If we found a sequence of at least 5 numbers
        if len(sequence) >= 5:
            sequence_id += 1  # Increment the sequence identifier
            matrix[s_idx, 4] = sequence_id  # Mark with sequence identifier


    print(f'  {sequence_id} possible sequences with more than 5 numbers found.\n')


# Function to check which sequence is most likely the page numbers
def identify_page_numbers(matrix, book_char_len):
    """
    Checks found sequences for evenness of distribution over the e-book.
    Looks for longest sequence with even distribution.

    Args:
        - matrix: Overview over numbers in text
            Columns: 1. extracted number, 2. position in file, 3. overall position, 4. file index.
        - book_char_len: Overall length of total html code in EPUB file.

    Returns:
        - seq_id: The ID of the sequence that most likely corresponds to page numbers.
            Referenced to the 5th column of the numbers matrix
        - largest_number: The largest number of the seq_id sequence (important for the "reverse" sanity check)
    """

    # Convert the 5th column to numeric, forcing invalid strings to NaN
    matrix[:, 4] = np.array([float(x) if x not in ['NaN', None, ''] else np.nan for x in matrix[:, 4]])

    # Convert the whole matrix to numeric (if necessary)
    matrix = matrix.astype(float)

    # Filter out rows where the sequence ID (third column) is None (no page numbers)
    filtered_matrix_2 = matrix[~np.isnan(matrix[:, 4])]

    ## check for somewhat even distributions
    # Length of the full text to split it into 10 sections

    # Find the top 3 longest sequences by occurrence
    sequence_ids, counts = np.unique(filtered_matrix_2[:, 4], return_counts=True)
    top_3_sequences = sequence_ids[np.argsort(-counts)[:3]]  # Sort and get top 3 IDs

    # Initialize a 2D array to store counts for each section and each sequence
    dist_count = np.zeros((3, 10), dtype=int)

    # Calculate distribution for each of the top 3 sequences
    for seq_index, seq_id in enumerate(top_3_sequences):
        seq_positions = filtered_matrix_2[filtered_matrix_2[:, 4] == seq_id][:, 2]  # Get positions for this sequence
        i_in = 0

        for i in range(10):
            i_out = int(book_char_len / 10 * (i + 1))
            dist_count[seq_index, i] = np.sum((seq_positions >= i_in) & (seq_positions < i_out))
            i_in = i_out


    print("Step 2: Check identified sequences for page number likeness.")
    print(f"  Distribution counts for each of the 3 longest sequences across sections:\n")
    print(dist_count)

    # Arrays to store results
    valid_sequences = []  # To store indices of sequences that meet the conditions
    sequence_totals = []  # To store total counts of sequences that meet the conditions

    # Iterate over each sequence's distribution
    for seq_index in range(3):
        middle_section = dist_count[seq_index, 1:9]  # Ignore the first and last entries

        # Check if all middle entries are non-zero
        if np.all(middle_section > 0):
            # Get the minimum and maximum of the middle section
            min_count = np.min(middle_section)
            max_count = np.max(middle_section)

            # Check if max_count is no more than double min_count
            if max_count <= 2 * min_count:
                valid_sequences.append(seq_index)  # Store the sequence index
                sequence_totals.append(np.sum(middle_section))  # Store the total count for this sequence

    # Print results
    if valid_sequences:

        print(f"  Found {len(valid_sequences)} sequence(s) that look like page numbers.\n")

        # Find the sequence with the maximum total count
        max_total_index = valid_sequences[np.argmax(sequence_totals)]
        seq_id = top_3_sequences[max_total_index]

        # Filter the rows belonging to the longest valid sequence
        valid_rows = filtered_matrix_2[filtered_matrix_2[:, 4] == seq_id]

        # Extract the first column (numbers)
        sequence_numbers = valid_rows[:, 0]

        # Find the smallest and largest numbers in the sequence
        smallest_number = np.min(sequence_numbers)
        largest_number = np.max(sequence_numbers)

        # Get the total count of numbers in the sequence
        total_count = len(sequence_numbers)

        # Print detailed information for the longest valid sequence
        print(f"  The longest plausible sequence revealed {total_count} page numbers")
        print(f"  Smallest page number: {smallest_number}")
        print(f"  Largest page number: {largest_number}\n")
    else:
        print("No sequences satisfy the page number conditions.")
        sys.exit("Exiting the script as no valid sequences were found.")

    return seq_id, largest_number

def sanity_check_with_backwards_sequence(matrix, page_number_seq_id, largest_number):
    """
    Checking if the identified page sequence also makes sense in the other direction.

    Args:
        - matrix (numpy.ndarray): The input matrix with 5 columns, sorted by column 3 (position in e-book).
        - page_number_seq_id (float): The sequence ID of the page numbers.
        - largest_number (float): The starting point for the new sequence.

    Returns:
        - matrix (numpy.ndarray): The updated matrix with a new column indicating the backwards sequence based on
            the highest number of the page sequence.
    """

    print("Step 3: Sanity check for the most likely page number sequence.")

    # Ensure the matrix is float type to allow NaN assignments
    matrix = matrix.astype(float)

    # Add a new column to log the backward sequence (initialize with NaN)
    if matrix.shape[1] == 5:
        # matrix = np.column_stack((matrix, np.nan))
        # Add a new empty column for sequence IDs
        matrix = np.hstack((matrix, np.full((matrix.shape[0], 1), np.nan, dtype=float)))

    # Find the starting row for the new sequence
    start_index = np.where((matrix[:, 0] == largest_number) & (matrix[:, 4] == page_number_seq_id))[0]
    # start_index = np.where(matrix[:, 0] == largest_number)[0]
    if len(start_index) == 0:
        print("Starting point not found in the matrix.")
        return matrix
    start_index = start_index[0]

    # Iterate upwards in the matrix and build the new sequence
    current_value = largest_number - 1  # Start counting backwards
    for row_index in range(start_index - 1, -1, -1):  # Iterate upwards without changing the matrix order
        row_value = matrix[row_index, 0]

        # Check if the current row value matches the expected value
        if row_value == current_value:
            matrix[row_index, 5] = page_number_seq_id # Log the new sequence ID in column 6
            current_value -= 1  # Decrement to the next expected value

    return matrix


def create_tilt_matrix_with_logging(old_matrix, page_number_seq_id):
    """
    Create a matrix that indicates which numbers are safe to remove. If the forwards and backwards positions
    of a sequence match, it is confirmed that it is indeed a page number.

    Parameters:
    - old_matrix (numpy.ndarray): The page number matrix with forward and backward sequence indication.
        Columns: 1. numbers, 5. forward sequence, 6. backward sequence
    - page_number_seq_id (float): The sequence ID of the page numbers.

    Returns:
    - tilt_matrix (numpy.ndarray): The resulting 2-column matrix.
        1st column is the position of the number to be deleted within a html file
        2nd column is the id of the html file, relative to the idx of html files found with the sorting mechanism
    """
    # Identify rows where the first condition (5th column) matches
    condition_1_rows = old_matrix[old_matrix[:, 4] == page_number_seq_id]

    # Filter rows where both category columns (5th and 6th columns) match the page_number_seq_id
    filtered_rows = condition_1_rows[condition_1_rows[:, 5] == page_number_seq_id]

    # Calculate conflicts
    total_possible = len(condition_1_rows)  # Rows satisfying the first condition
    sequence_conflicts = total_possible - len(filtered_rows)  # Rows not satisfying the second condition

    # Log results
    print(f"  Of the {total_possible} possible page numbers, {sequence_conflicts} "
          f"  had a sequence conflict and will not be deleted.\n")

    # Initialize a list to collect rows for the tilt_matrix
    tilt_rows = []

    for row in filtered_rows:
        number = int(row[0])  # Get the number (1st column)
        digits = len(str(abs(number)))  # Number of digits in the number
        base_index = int(row[1])  # Original index (2nd column)
        file_index = int(row[3])  # File index (4th column)

        # Add one row for each digit, incrementing the position index
        for i in range(digits):
            tilt_rows.append([base_index + i, file_index])

    # Convert the list of rows to a numpy array
    tilt_matrix = np.array(tilt_rows, dtype=int)

    return tilt_matrix

def create_cleaned_epub_file(epub_path, tilt_matrix):
    """
    Create a new EPUB file excluding all the parts of the HTMLs that have been identified as artifacts.
    (Hardcoded page numbers in v.0.9).

    Parameters:
    - epub_path (str): Path to the original EPUB file.
    - tilt_matrix (numpy.ndarray): The resulting 2-column matrix.
        1st column is the position of the number to be deleted within an HTML file
        2nd column is the id of the HTML file, relative to the idx of HTML files found with the sorting mechanism
    """

    print("CREATING CLEANED FILE  (v0.9 only removes page number artifacts)")

    # Get the directory, filename, and extension
    directory, filename = os.path.split(epub_path)
    file_name, file_extension = os.path.splitext(filename)

    # Add 'cleaned' to the filename before the extension
    cleaned_filename = file_name + '_cleaned' + file_extension

    # Create the full output path
    output_path = os.path.join(directory, cleaned_filename)

    # Check if the output file already exists
    if os.path.exists(output_path):
        red_warning = f"\033[91mThe file '{cleaned_filename}' already exists. Overwrite? (yes/no):\033[0m "
        user_input = input(red_warning).strip().lower()
        if user_input not in ['yes', 'y']:
            print(f"Please move or rename the existing file '{cleaned_filename}' and try again.")
            return

    # Open the original EPUB file
    with zipfile.ZipFile(epub_path, 'r') as epub:
        # List and sort all files in the EPUB
        file_list = sort_files(epub_path)

        # Filter for HTML files (you can adjust this filter as needed)
        html_files = [file for file in file_list if file.endswith('.html')]

        # Create a new EPUB file
        with zipfile.ZipFile(output_path, 'w') as new_epub:
            for file in file_list:
                if file.endswith('.html'):
                    html_idx = html_files.index(file)
                    html_tilt = tilt_matrix[tilt_matrix[:, 1] == html_idx, 0]

                    with epub.open(file) as html_file:
                        html_content = html_file.read().decode('utf-8')

                        # Create a new string excluding the characters at the indices in html_tilt
                        html_content_filtered = ''.join(
                            [char for idx, char in enumerate(html_content) if idx not in html_tilt])

                    # Write the filtered HTML content to the new EPUB
                    new_epub.writestr(file, html_content_filtered)

                else:
                    # Copy all other files as-is
                    with epub.open(file) as other_file:
                        new_epub.writestr(file, other_file.read())

    print(f"New EPUB created without (most) page number artifacts: {output_path}")


# Ask for file location
if __name__ == "__main__":
    epub_path = select_file()
    if epub_path:
        print(f"Selected file: {epub_path}")
    else:
        print("No file selected.")



# Identify numbers in teh ebook text. Create a matrix.
# Column1 = number; C.2 = idx within html, C.3 = grand idx, C.4 = file idx
filtered_matrix, cumulative_html_length = load_all_html_with_overall_position(epub_path)

# Add a new empty column for sequence IDs
filtered_matrix = np.hstack((filtered_matrix, np.full((filtered_matrix.shape[0], 1), np.nan, dtype=float)))

# Try to identify sequences of numbers and add a sequence ID in C.5
identify_sequences(filtered_matrix)

# Find the number sequences that behave like page numbers
page_number_seq_id, largest_number = identify_page_numbers(filtered_matrix, cumulative_html_length)

# Perform a sanity check on the found page number sequence by backwards sequencing
updated_matrix = sanity_check_with_backwards_sequence(filtered_matrix, page_number_seq_id, largest_number)

# Generate the tilt_matrix based on page number artifacts
tilt_matrix = create_tilt_matrix_with_logging(updated_matrix, page_number_seq_id)

# Create new cleaned EPUB file by excluding artifacts
create_cleaned_epub_file(epub_path, tilt_matrix)

