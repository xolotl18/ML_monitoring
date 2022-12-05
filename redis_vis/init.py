import os

import ml2rt
import numpy as np
import pandas as pd
import redis
import redisai as rai
from tqdm import tqdm

con = rai.Client(host='localhost', port=6379)
r = redis.Redis(host='localhost', port=6379)

# storing the model in Redis

fft_model = ml2rt.load_model('fft.pt')
out = con.modelset('model:fft', 'torch', 'cpu', fft_model)
print(out)

# storing the signals in Redis

input_url = "bosch_cnc"
metadata_url = os.path.join(
    input_url,
    "metadata.csv"
)
metadata_df = pd.read_csv(metadata_url)

DATA_COLUMNS = {
    "x": np.float32,
    "y": np.float32,
    "z": np.float32,
}

filenames = os.listdir(input_url)
filenames.remove('metadata.csv')
filenames.remove('.DS_Store')
# print(filenames[:4])

for filename in tqdm(filenames):
    file_path = os.path.join(input_url, filename)
    data_df = pd.read_csv(
        file_path,
        names=DATA_COLUMNS.keys(),
        dtype=DATA_COLUMNS,
    )
    metadata_row = metadata_df.loc[metadata_df['filename'] == filename]

    x = np.array(data_df['x']).tobytes()
    y = np.array(data_df['y']).tobytes()
    z = np.array(data_df['z']).tobytes()
    mapping = {
        'x': x,
        'y': y,
        'z': z,
        'machine_id': str(metadata_row.machine_id.values[0]),
        'process_id': str(metadata_row.op_id.values[0]),
        'sampling_f': str(metadata_row.sampling_f.values[0]),
        'status': str(metadata_row.status.values[0]),
    }
    signal_name = filename.split('.')[0]

    r.hset(name='signal:'+signal_name, mapping=mapping)
