# EPUB Artifact Cleaner (v0.9)

EPUB Artifact Cleaner is a Python script designed to remove hard-coded artifacts from EPUB files, such as page numbers, to enhance readability and formatting. These artifacts often result from poor EPUB conversions or improperly scanned pages.
The Cleaner will create a new EPUB file in the same directory called '[original_name]_cleaned.epub'.

## Features
### Current (v0.9)
- **Compatibility with EPUB Files**:
  - Seamless handling of standard EPUB file structures.
- **Page Number Cleanup**:
  - Detects and removes hard-coded page numbers.
  - Identifies numerical sequences indicative of page numbers.
  - Includes sanity checks to ensure only unwanted numbers are removed.
- **File Safety**:
  - Prevents accidental overwriting of original or other files.

### Planned
- **v1.0**:
  - Improvement of page number detection acuracy.
  - Removal of watermarks.
- **v1.1**:
  - Removal of random line breaks.
- **v2.0**:
  - Support for additional e-book formats.

## Usage
1. **Install dependencies**
  - Run the following command to install the required libraries:

  pip install numpy pyqt5

2. **Run the script**
3. Select the target EPUB file via the file dialog.
4. A new EPUB file, free of page number artifacts, will be generated in the specified output directory.

## Requirements
- Python 3.x
- Libraries: `numpy`, `PyQt5`
- Other standard libraries: `zipfile`, `os`, `sys`, `re`, `xml.etree.ElementTree`

## Installation
Clone the repository and navigate to its directory:

  git clone https://github.com/RisingPlatypus/EPUB_Artifact_Cleaner.git
  cd EPUB_artifact_cleaner

Run the script:

  python EPUB_artifact_cleaner.py


## Contributing
Contributions are welcome! Feel free to submit issues or pull requests for bug fixes or new features.

## License
This project is licensed under the MIT License.
