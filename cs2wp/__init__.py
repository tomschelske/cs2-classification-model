"""CS2 Round Win-Probability Model.

Pipeline:
    parse.py     .dem files            -> parsed demo data (Polars)
    features.py  parsed demo data      -> labeled snapshot rows
    baseline.py  man-advantage rule    -> the bar every model must beat
    train.py     snapshots.parquet     -> model.pkl + metrics.json
    api.py       model.pkl             -> FastAPI POST /predict

See cs2-win-probability-project.md for the full plan.
"""

__version__ = "0.1.0"
