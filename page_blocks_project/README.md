# Page Blocks Classification Analysis Project

This project analyzes the Page Blocks Classification dataset from the UCI Machine Learning Repository and generates a comprehensive report with visualizations.

## Project Structure

```
page_blocks_project/
├── code/
│   ├── analysis.py
│   └── generate_report.py
├── data/
├── images/
├── output/
└── requirements.txt
```

## Setup and Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Analysis

1. First, run the analysis script to generate the plots:
   ```bash
   cd code
   python analysis.py
   ```

2. Then, generate the report:
   ```bash
   python generate_report.py
   ```

3. The final report will be saved as `2022CSxxx.docx` in the `output` directory.

4. Convert the Word document to PDF using your preferred method (Microsoft Word, LibreOffice, etc.).

## Customization

- Update your name and registration number in `generate_report.py`
- The watermark text can be modified in `analysis.py`

## Output Files

- All plots are saved in the `images/` directory
- The dataset is downloaded and saved in the `data/` directory
- The final report is generated in the `output/` directory

## Requirements

- Python 3.7+
- See requirements.txt for Python package dependencies 