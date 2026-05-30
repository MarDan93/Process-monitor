"""Helpers for serializing/deserializing session state in dcc.Store."""
import base64
import pickle
import io
import numpy as np
import pandas as pd


def model_to_b64(model: dict) -> str:
    """Pickle the model dict (contains sklearn objects) and base64-encode it."""
    return base64.b64encode(pickle.dumps(model)).decode()


def b64_to_model(s: str) -> dict:
    return pickle.loads(base64.b64decode(s.encode()))


def df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient='split', default_handler=str)


def json_to_df(s: str) -> pd.DataFrame:
    return pd.read_json(io.StringIO(s), orient='split')


def parse_upload(contents: str, filename: str) -> pd.DataFrame:
    """Parse a dcc.Upload content string into a DataFrame."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if filename.endswith('.csv'):
        return pd.read_csv(io.StringIO(decoded.decode('utf-8', errors='replace')))
    elif filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(decoded))
    raise ValueError(f"Unsupported file type: {filename}")


def mon_file_to_dict(name: str, n_rows: int, mon: dict) -> dict:
    return {
        'name': name,
        'n_rows': n_rows,
        'mon_pkl': base64.b64encode(pickle.dumps(mon)).decode(),
    }


def mon_dict_to_mon(d: dict) -> dict:
    return pickle.loads(base64.b64decode(d['mon_pkl'].encode()))
