from group5_update_single_model import main
from group5_energy.pipeline import AUTOGLUON_TABULAR_MODEL, AUTOGLUON_TIMESERIES_MODEL


if __name__ == "__main__":
    main(AUTOGLUON_TABULAR_MODEL)
    main(AUTOGLUON_TIMESERIES_MODEL)
