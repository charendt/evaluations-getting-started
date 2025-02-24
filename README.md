# AI Evaluation Workspace

This workspace contains Jupyter notebooks and utility scripts for running AI model evaluations.  
It includes notebooks for basic usage, AI Foundry evaluation, and simulation of AI search that uses composite evaluation such as QAEvaluator.

## Workspace Structure

- **Notebooks**:  
  - `01 Basics.ipynb`: Walks through basic concepts.  
  - `02 AI Foundry Evaluation.ipynb`: Contains evaluation examples using AI Foundry tests.  
  - `03 AI Search Simulator.ipynb`: Simulates AI search and executes evaluations (QAEvaluator, GroundednessEvaluator, etc.).
- **utils/**: Contains Python modules for model endpoints and model functions used across evaluations.
- **assets/**: Stores image assets referenced in the notebooks.

## Requirements

Ensure you have Python 3.12+ installed. Install dependencies using:

```sh
pip install -r requirements.txt
