import os
import shutil

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
            shutil.rmtree('outputs')


def test_run_pipeline_seed_propagation(tmp_path):
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_pipeline(model='qpix', run_wwz=False, seed=111)
        first = pd.read_csv('outputs/qpix_signal.csv')['signal'].to_numpy()

        run_pipeline(model='qpix', run_wwz=False, seed=222)
        second = pd.read_csv('outputs/qpix_signal.csv')['signal'].to_numpy()

        run_pipeline(model='qpix', run_wwz=False, seed=111)
        third = pd.read_csv('outputs/qpix_signal.csv')['signal'].to_numpy()
    finally:
        if os.path.exists('outputs'):
            shutil.rmtree('outputs')
        os.chdir(original_cwd)

    assert not (first == second).all()
    assert (first == third).all()
