# Project Repository

## Overview

This repository contains key files related to **optical bandgap energy calculation** and **Gaussian Process Regression (GPR) modeling** for **DFT-based material simulations**.

## Files and Structure

- **`main.ipynb`** – The main page of the Jupyter Notebook interface for this project.
- **`data/`** – Contains datasets generated from Density Functional Theory (DFT) simulations and used for Gaussian Process Regression (GPR) modeling.
- **`models/`** – A directory for storing machine learning models.
- **`plots/`** – Stores visualizations, including spectral absorbance plots, Tauc plots, and GPR prediction comparisons.
- **`src/`** – A folder containing source code and functions used in this project.
  

## Project Description

This project focuses on **optical bandgap prediction** using **Gaussian Process Regression (GPR)** integrated with **Density Functional Theory (DFT) simulations**. Key features:
- **Automated bandgap extraction** using the **Tauc plot method**.
- **GPR-based interpolation and extrapolation** of spectral data.
- **Uncertainty quantification** for better model reliability.

## Dependencies

To run this project, you will need:

- Python 3.x
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy

## Usage

1. **Run `main.ipynb` in Jupyter Notebook.**  
   - The notebook automatically loads functions from `src/`, processes datasets from `data/`, and performs calculations.
2. **Check `plots/` for generated visualizations.**
3. **Review the results** and analysis at the end of notebook.


## Citation

If you use this work, please cite the conference paper:
...
