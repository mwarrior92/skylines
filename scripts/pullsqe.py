import meas_handler
import helpers


if __name__ == '__main__':
    results = meas_handler.Results(helpers.datadir+'sqe2.json', description='short_query_experiment')
    results.get_results()
