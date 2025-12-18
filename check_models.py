from pathlib import Path
import pickle
import joblib

for p in ['diabetic_prediction/diabetes_model.pkl','diabetic_prediction/scaler.pkl']:
    b = Path(p).read_bytes()
    print('\nFile:', p)
    print('First bytes:', b[:20])
    try:
        obj = pickle.loads(b)
        print('Loaded with pickle:', type(obj))
    except Exception as e:
        print('Pickle load error:', repr(e))
    try:
        obj = joblib.load(p)
        print('Loaded with joblib:', type(obj))
    except Exception as e:
        print('Joblib load error:', repr(e))
    # Try to detect if it's a numpy file
    try:
        import numpy as np
        with open(p,'rb') as f:
            head = f.read(8)
        print('Head bytes for detection:', head)
    except Exception as e:
        print('Numpy detect error:', repr(e))
