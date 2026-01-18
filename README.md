# Formulae-Documentation

This repository contains the documentation, source code, and analysis tools for the HardHaq submission. It is designed to handle parameter optimization and data visualization for Comsol simulations.

## Repository Structure

The repository is organized into three main sections:

### 1. Website (Analysis Tool)
Located in the `Website/` directory.
This is a Streamlit-based web application designed to compare and analyze parameter files.
- **Features**:
    - Scans folders for parameterized text files (e.g., `V_rf_300.txt`).
    - Extracts numeric tables and generates interactive plots.
    - Compares multiple files against a baseline.
    - Provides statistical summaries and correlation analysis.
- **Usage**: Detailed instructions for running the app are available in [Website/README.md](Website/README.md).

### 2. Comsol Optimization
Located in the `Comsol_Optimization/` directory.
Contains scripts related to the optimization processes.
- `Comsol_Optimize.py`: Main script for optimization logic.
- `Comsol_SchizoTest`: Test or auxiliary file.

### 3. Documentation (PDF)
Located in the `PDF/` directory.
Contains the formal submission documents and visuals.
- `HardHaq_Submission.pdf`: The final submission paper.
- `HardHaq_Submission.tex`: LaTeX source for the submission.
- `PDF_Visuals/`: Directory containing visual assets used in the documentation.

## Getting Started

To run the analysis dashboard:

1. Navigate to the `Website` directory.
2. Install the required dependencies:
   ```bash
   pip install -r Website/requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run Website/streamlit_app.py
   ```
