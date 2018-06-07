import meas_handler
import helpers


if __name__ == '__main__':
    results = meas_handler.Results(helpers.datadir+'sap.json', description='skyline_all_probes')
    results.get_results()

