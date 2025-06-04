import os
import pandas as pd
from run_pipeline import run_pipeline


def test_run_pipeline_real(tmp_path):
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame({'time': [0.0, 0.5, 1.0], 'signal': [1.0, 2.0, 3.0]})
    df.to_csv(csv_path, index=False)

    run_pipeline(model='real', input_file=str(csv_path), run_wwz=False, seed=123)

    output_file = f"outputs/{csv_path.stem}_signal.csv"
    try:
        assert os.path.exists(output_file)
    finally:
        if os.path.exists('outputs'):
            import shutil
            shutil.rmtree('outputs')
