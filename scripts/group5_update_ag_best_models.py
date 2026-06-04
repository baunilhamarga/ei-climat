from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from group5_update_single_model import (
    AUTOGLUON_BEST_TABULAR_MODEL,
    AUTOGLUON_BEST_TIMESERIES_MODEL,
    main,
)
from group5_energy.pipeline import AUTOGLUON_TABULAR_MODEL, AUTOGLUON_TIMESERIES_MODEL


if __name__ == "__main__":
    main(AUTOGLUON_TABULAR_MODEL, output_model_name=AUTOGLUON_BEST_TABULAR_MODEL)
    main(AUTOGLUON_TIMESERIES_MODEL, output_model_name=AUTOGLUON_BEST_TIMESERIES_MODEL)
