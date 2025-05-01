
import os
from tte_fits_to_csv import bin_tte_fits

def batch_convert_tte(data_dir="data", output_dir="data", bin_width=0.01):
    fits_files = [f for f in os.listdir(data_dir) if f.endswith(".fit") or f.endswith(".fits")]
    for f in fits_files:
        fits_path = os.path.join(data_dir, f)
        base = os.path.splitext(f)[0]
        csv_output = os.path.join(output_dir, f"{base}.csv")
        try:
            bin_tte_fits(fits_path, csv_output, bin_width=bin_width)
        except Exception as e:
            print(f"Failed to process {f}: {e}")

if __name__ == "__main__":
    batch_convert_tte()
