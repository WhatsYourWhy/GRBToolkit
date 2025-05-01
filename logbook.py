
import csv
import os
from datetime import datetime

LOGBOOK_PATH = "outputs/logbook.csv"

def log_run(model, seed, params, output_csv, segments=None, notes=""):
    log_exists = os.path.exists(LOGBOOK_PATH)
    with open(LOGBOOK_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not log_exists:
            writer.writerow(["Date", "Model", "Seed", "Params", "Output CSV", "BB Segments", "Notes"])
        writer.writerow([
            datetime.now().isoformat(timespec='seconds'),
            model,
            seed,
            params,
            output_csv,
            segments if segments is not None else "",
            notes
        ])
    print(f"Logged run to {LOGBOOK_PATH}")
