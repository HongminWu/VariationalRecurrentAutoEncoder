from smach_based_introspection_framework._constant import anomaly_classification_feature_selection_folder
import coloredlogs, logging
import os, ipdb
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

coloredlogs.install()
def run():
    logger = logging.getLogger()
    logger.info("load_csv_data_from_filtered_scheme")
    
    folders = glob.glob(os.path.join(
        anomaly_classification_feature_selection_folder,
        'No.* filtering scheme',
        'anomalies_grouped_by_type',
        'anomaly_type_(*)',
    )) 

    target = []
    df_frames = []
    iid = 0
    for folder in folders:
        logger.info(folder)
        path_postfix = os.path.relpath(folder, anomaly_classification_feature_selection_folder).replace("anomalies_grouped_by_type"+os.sep, "")

        prog = re.compile(r'anomaly_type_\(?([^\(\)]+)\)?')
        anomaly_type = prog.search(path_postfix).group(1)
        csvs = glob.glob(os.path.join(folder, '*', '*.csv'))
        for j in csvs:
            df =  pd.read_csv(j, sep = ',')
            # delete the 1st column with is time index
            df = df.drop(['Unnamed: 0'], axis = 1)
            iid = iid + 1
            df['id']   = iid
            df['time'] = df.index
            df_frames.append(df)            
            target.append(anomaly_type)
    timeseries = pd.concat(df_frames)
    timeseries.index = range(timeseries.shape[0])
    y = pd.Series(np.array(target), range(1,len(target) + 1))
    logger.info('Successfully! Convert the data format for tsfresh package')

    return timeseries, y
    
if __name__ == '__main__':
    timeseries, y = run()

    timeseries.to_csv('timeseries.csv')
    y.to_csv('y.csv')

    plot_iid = 1
    cols = timeseries.columns.tolist()
    cols.remove('id')
    timeseries[timeseries.id == plot_iid][cols].plot(x='time', title='Example: timeseries for id=%s (%s)' % (plot_iid, y[plot_iid]), figsize=(24,6))
    plot_iid = 3
    timeseries[timeseries.id == plot_iid][cols].plot(x='time', title='Example: timeseries for id=%s (%s)' % (plot_iid, y[plot_iid]), figsize=(24,6))
    plt.show()
