# AI-Powered Disaster Risk Predictor

## Important
This project uses a synthetic prototype dataset for demonstration.
It is NOT an official disaster-warning system.

## Setup on Windows

Open PowerShell inside this folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py train_model.py
python -m streamlit run app.py
```

If PowerShell blocks activation, you can skip activation and use:

```powershell
py -m pip install -r requirements.txt
py train_model.py
python -m streamlit run app.py
```

## Files

- data/disaster_data.csv - prototype training data
- train_model.py - trains Random Forest model
- model/flood_model.pkl - generated model
- app.py - Streamlit dashboard
