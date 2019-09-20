import utils
import os

data_dir = "logs/DataAggregation/"
dirs = os.listdir(data_dir)
for dir in dirs:
    utils.log_data(data_dir + dir + "/", data_dir + dir + "/")