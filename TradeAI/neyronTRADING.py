import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import datetime
from psutil import Process
from IPython.display import display_html

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator

from PIL import Image, ImageDraw

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 20)
# you can change config here)))
tickers = ["AAPL.US", "AMZN.US", "MSFT.US", "NVDA.US", "TSLA.US", "GOOGL.US", ]
names = ["Apple", "Amazon", "Microsoft", "Nvidia", "Tesla", "Google"]

timeframe_in = "D1"  # timeframe for neural network training - input and we will trade on the same timeframe
timeframe_out = "W1"  # timeframe for neural network training - output - predicted value

# how many weeks use for image generation
weeks_per_image = 2

# parameters for drawing images
period_sma_slow = 6  # period of slow SMA - period in days
period_sma_fast = 3  # period of fast SMA - period in days

draw_window = 128  # draw window # draw_window => is from num days in weeks_per_image
steps_skip = 16  # step shift of the data window
draw_size = 128  # side size of the square picture

folder_NN = "NN"
folder_NN_valid = "NN_valid"
# Do it for all shares
df_all_in = {}
df_all_out = {}
for ticker in tickers:
    df_all_in[ticker] = pd.read_csv(f"archive/{timeframe_in}/{ticker}_{timeframe_in}.csv", parse_dates=['datetime'])
    df_all_out[ticker] = pd.read_csv(f"archive/{timeframe_out}/{ticker}_{timeframe_out}.csv", parse_dates=['datetime'])


def show_dfs_in_side_by_side(dfs, captions):
    _disp_dfs = []
    for i in range(len(dfs)):
        _df = dfs[i]
        _caption = captions[i]
        _df_styler = _df.style.set_table_attributes("style='display:inline'").set_caption(_caption)
        _disp_dfs.append(_df_styler._repr_html_())
    display_html(_disp_dfs, raw=True)


show_N_values = 25  # None - for all
show_dfs_in_side_by_side(dfs=[df_all_in["AAPL.US"][:show_N_values], df_all_out["AAPL.US"][:show_N_values]],
                         captions=["Apple daily D1", "Apple weekly W1"])
# P.S. If you copy|edit this notebook, then you see these three dataframes in side by side
show_N_values = 40
show_it_only_once = True
df_in = pd.DataFrame()
df_out = pd.DataFrame()

# Do it for all shares
for ticker in tickers:
    df_in = df_all_in[ticker]  # get df for ticker - input
    df_out = df_all_out[ticker][["datetime", "close"]]  # get df for ticker - output

    # join features and target by datetime and two timeframes D1, W1
    packed_w1 = list(zip(df_out.datetime, df_out.close))
    df_in['target'] = [[_close_w1 for _date_w1, _close_w1 in packed_w1 if _date_d1 < _date_w1] for _date_d1 in
                       df_in["datetime"]]  # < - as W1 is formed on next working day - so not <=
    df_in['target'] = [el[0] if len(el) else np.nan for el in df_in["target"]]
    df_in['target_datetime'] = [[_date_w1 for _date_w1, _close_w1 in packed_w1 if _date_d1 < _date_w1] for _date_d1 in
                                df_in["datetime"]]  # < - as W1 is formed on next working day - so not <=
    df_in['target_datetime'] = [el[0] if len(el) else np.nan for el in df_in["target_datetime"]]
    break
show_N_values = 40
show_it_only_once = True
df_in = pd.DataFrame()
df_out = pd.DataFrame()

# Do it for all shares
for ticker in tickers:
    df_in = df_all_in[ticker]  # get df for ticker - input
    df_out = df_all_out[ticker][["datetime", "close"]]  # get df for ticker - output

    # join features and target by datetime and two timeframes D1, W1
    packed_w1 = list(zip(df_out.datetime, df_out.close))
    df_in['target'] = [[_close_w1 for _date_w1, _close_w1 in packed_w1 if _date_d1 < _date_w1] for _date_d1 in
                       df_in["datetime"]]  # < - as W1 is formed on next working day - so not <=
    df_in['target'] = [el[0] if len(el) else np.nan for el in df_in["target"]]
    df_in['target_datetime'] = [[_date_w1 for _date_w1, _close_w1 in packed_w1 if _date_d1 < _date_w1] for _date_d1 in
                                df_in["datetime"]]  # < - as W1 is formed on next working day - so not <=
    df_in['target_datetime'] = [el[0] if len(el) else np.nan for el in df_in["target_datetime"]]

    # create sequence
    df_seq = pd.DataFrame(columns=["sequence", "target"])
    # create sequence for dates - for analytics
    df_seq_dates = pd.DataFrame(columns=["sequence_dates", "target_date"])

    # prepare values for sequence
    packed_d1 = list(zip(df_in.datetime, df_in.close))
    packed_w1 = list(zip(df_out.datetime, df_out.close))
    j = 0
    for i in range(len(df_out)):
        _date_w1, _close_w1 = packed_w1[i][0], packed_w1[i][1]
        arr = []
        arr_dates = []
        while True:
            _date_d1, _close_d1 = packed_d1[j][0], packed_d1[j][1]
            #         print(i, j, _date_d1, _date_w1, _date_d1 < _date_w1, arr, len(arr))
            if _date_d1 < _date_w1:
                arr.append(_close_d1)
                arr_dates.append(_date_d1.date())  # .date() as we work with D1, for others timeframes - remove .date()
            else:
                break
            j += 1
            if j >= len(df_in): break
        df_seq.loc[len(df_seq.index)] = [arr, _close_w1]
        df_seq_dates.loc[len(df_seq_dates.index)] = [arr_dates, _date_w1]

    # calc len of sequencies
    df_seq['len_seq'] = df_seq.apply(lambda row: len(row["sequence"]), axis=1)
    df_seq_dates['len_seq_dates'] = df_seq_dates.apply(lambda row: len(row["sequence_dates"]), axis=1)

    break
show_N_values = 40
show_it_only_once = True
df_in = pd.DataFrame()
df_out = pd.DataFrame()

# Do it for all shares
for ticker in tickers:
    df_in = df_all_in[ticker]  # get df for ticker - input
    df_out = df_all_out[ticker][["datetime", "close"]]  # get df for ticker - output

    # join features and target by datetime and two timeframes D1, W1
    packed_w1 = list(zip(df_out.datetime, df_out.close))
    df_in['target'] = [[_close_w1 for _date_w1, _close_w1 in packed_w1 if _date_d1 < _date_w1] for _date_d1 in
                       df_in["datetime"]]  # < - as W1 is formed on next working day - so not <=
    df_in['target'] = [el[0] if len(el) else np.nan for el in df_in["target"]]
    df_in['target_datetime'] = [[_date_w1 for _date_w1, _close_w1 in packed_w1 if _date_d1 < _date_w1] for _date_d1 in
                                df_in["datetime"]]  # < - as W1 is formed on next working day - so not <=
    df_in['target_datetime'] = [el[0] if len(el) else np.nan for el in df_in["target_datetime"]]

    # create sequence
    df_seq = pd.DataFrame(columns=["sequence", "target"])
    # create sequence for dates - for analytics
    df_seq_dates = pd.DataFrame(columns=["sequence_dates", "target_date"])

    # prepare values for sequence
    packed_d1 = list(zip(df_in.datetime, df_in.close))
    packed_w1 = list(zip(df_out.datetime, df_out.close))
    j = 0
    for i in range(len(df_out)):
        _date_w1, _close_w1 = packed_w1[i][0], packed_w1[i][1]
        arr = []
        arr_dates = []
        while True:
            _date_d1, _close_d1 = packed_d1[j][0], packed_d1[j][1]
            #         print(i, j, _date_d1, _date_w1, _date_d1 < _date_w1, arr, len(arr))
            if _date_d1 < _date_w1:
                arr.append(_close_d1)
                arr_dates.append(_date_d1.date())  # .date() as we work with D1, for others timeframes - remove .date()
            else:
                break
            j += 1
            if j >= len(df_in): break
        df_seq.loc[len(df_seq.index)] = [arr, _close_w1]
        df_seq_dates.loc[len(df_seq_dates.index)] = [arr_dates, _date_w1]

    # calc len of sequencies
    df_seq['len_seq'] = df_seq.apply(lambda row: len(row["sequence"]), axis=1)
    df_seq_dates['len_seq_dates'] = df_seq_dates.apply(lambda row: len(row["sequence_dates"]), axis=1)

    df = df_out.copy()
    df["sequence"] = df_seq["sequence"]
    df["len_seq"] = df_seq["len_seq"]
    df["sequence_dates"] = df_seq_dates["sequence_dates"]
    df["len_seq_dates"] = df_seq_dates["len_seq_dates"]

    break
np.unique(df_seq['len_seq'].values, return_counts=True)
pd.options.mode.chained_assignment = None  # default='warn'
# create folders for images
folder_NN_timeframe = None
folder = folder_NN
if not os.path.exists(folder): os.makedirs(folder)
for timeframe in [timeframe_in, ]:  # timeframe_out
    _folder = os.path.join(folder, f"training_dataset_{timeframe}")
    folder_NN_timeframe = _folder
    if not os.path.exists(_folder): os.makedirs(_folder)
    for _class in [0, 1]:
        _folder_class = os.path.join(_folder, f"{_class}")
        if not os.path.exists(_folder_class): os.makedirs(_folder_class)
show_N_values = 40
show_it_only_once = True
df_in = pd.DataFrame()
df_out = pd.DataFrame()

# Do it for all shares
for ticker in tickers:
    df_in = df_all_in[ticker]  # get df for ticker - input
    df_out = df_all_out[ticker][["datetime", "close"]]  # get df for ticker - output

    # join features and target by datetime and two timeframes D1, W1
    packed_w1 = list(zip(df_out.datetime, df_out.close))
    df_in['target'] = [[_close_w1 for _date_w1, _close_w1 in packed_w1 if _date_d1 < _date_w1] for _date_d1 in
                       df_in["datetime"]]  # < - as W1 is formed on next working day - so not <=
    df_in['target'] = [el[0] if len(el) else np.nan for el in df_in["target"]]
    df_in['target_datetime'] = [[_date_w1 for _date_w1, _close_w1 in packed_w1 if _date_d1 < _date_w1] for _date_d1 in
                                df_in["datetime"]]  # < - as W1 is formed on next working day - so not <=
    df_in['target_datetime'] = [el[0] if len(el) else np.nan for el in df_in["target_datetime"]]

    # create sequence
    df_seq = pd.DataFrame(columns=["sequence", "target"])
    # create sequence for dates - for analytics
    df_seq_dates = pd.DataFrame(columns=["sequence_dates", "target_date"])

    # prepare values for sequence
    packed_d1 = list(zip(df_in.datetime, df_in.close))
    packed_w1 = list(zip(df_out.datetime, df_out.close))
    j = 0
    for i in range(len(df_out)):
        _date_w1, _close_w1 = packed_w1[i][0], packed_w1[i][1]
        arr = []
        arr_dates = []
        while True:
            _date_d1, _close_d1 = packed_d1[j][0], packed_d1[j][1]
            #         print(i, j, _date_d1, _date_w1, _date_d1 < _date_w1, arr, len(arr))
            if _date_d1 < _date_w1:
                arr.append(_close_d1)
                arr_dates.append(_date_d1.date())  # .date() as we work with D1, for others timeframes - remove .date()
            else:
                break
            j += 1
            if j >= len(df_in): break
        df_seq.loc[len(df_seq.index)] = [arr, _close_w1]
        df_seq_dates.loc[len(df_seq_dates.index)] = [arr_dates, _date_w1]

    # calc len of sequencies
    df_seq['len_seq'] = df_seq.apply(lambda row: len(row["sequence"]), axis=1)
    df_seq_dates['len_seq_dates'] = df_seq_dates.apply(lambda row: len(row["sequence_dates"]), axis=1)

    df = df_out.copy()
    df["target"] = df["close"]  # it was already shifted!!!
    df["sequence"] = df_seq["sequence"]
    df["len_seq"] = df_seq["len_seq"]
    df["sequence_dates"] = df_seq_dates["sequence_dates"]
    df["len_seq_dates"] = df_seq_dates["len_seq_dates"]

    df['pc'] = df['close'].pct_change()  # percent change
    df['target_class'] = [1 if el >= 0 else 0 for el in df["pc"]]  # class: 1 - up, 0 - down

    for i in range(len(df) - 1):
        if i < weeks_per_image: continue  # get N weeks
        target = df["target"].loc[i]
        seq = []
        for j in range(weeks_per_image):  # get N previous sequences
            print(555, i, j, i - weeks_per_image + j + 1, seq, df["sequence"].loc[i - weeks_per_image + j + 1])
            seq = seq + df["sequence"].loc[i - weeks_per_image + j + 1]  # extend
            print(f"\tNow: {seq} => Target: {target}")

        if i == 7: break

    break


# seq_all: [0.14, 0.14, 0.16, 0.15, 0.16, 0.16]
# sma_fast: [nan, nan, nan, nan, 0.15, 0.154]
# sma_slow: [nan, nan, nan, nan, nan, nan]
# seq: [0.14, 0.14, 0.16, 0.15, 0.16, 0.16] => target: 0.16

def generate_img_with_Nan(sma_fast, sma_slow, seq, log=None):
    """Image generation for neural network training/test"""

    # we need to use len of seq for fast and slow SMA
    sma_fast = sma_fast[len(sma_fast) - len(seq):]
    sma_slow = sma_slow[len(sma_slow) - len(seq):]

    _max = np.nanmax(sma_fast + sma_slow + seq)
    _min = np.nanmin(sma_fast + sma_slow + seq)
    if log: print(44444444, _max, _min)
    _delta_h = _max - _min
    _k_h = (draw_window - 1) / _delta_h  # scaling factor by _h for squaring
    w, h = draw_window, draw_window

    # creating new Image object - https://www.geeksforgeeks.org/python-pil-imagedraw-draw-line/
    img = Image.new("RGB", (w, h))
    img1 = ImageDraw.Draw(img)

    arr = sma_fast
    arr = [x for x in arr if not np.isnan(x)]
    for i in range(1, len(arr)):
        _k_w = (w - 1) / (len(arr) - 1)

        _h_1 = int((arr[i - 1] - _min) * _k_h)
        _h = int((arr[i] - _min) * _k_h)
        _w_1 = int((i - 1) * _k_w)
        _w = int(i * _k_w)

        if log: print(7777, i, arr[i], arr[i - 1], len(arr), " | ", _k_w, _k_h, " | ", _min, _max, " | []", _w_1, _h_1,
                      ", ", _w, _h, "]")

        shape = [(_w_1, _h_1), (_w, _h)]
        img1.line(shape, fill="blue", width=0)

    arr = sma_slow
    arr = [x for x in arr if not np.isnan(x)]
    for i in range(1, len(arr)):
        _k_w = (w - 1) / (len(arr) - 1)

        _h_1 = int((arr[i - 1] - _min) * _k_h)
        _h = int((arr[i] - _min) * _k_h)

        _w_1 = int((i - 1) * _k_w)
        _w = int(i * _k_w)

        if log: print(7777, i, arr[i], arr[i - 1], len(arr), " | ", _k_w, _k_h, " | ", _min, _max, " | []", _w_1, _h_1,
                      ", ", _w, _h, "]")

        shape = [(_w_1, _h_1), (_w, _h)]
        img1.line(shape, fill="green", width=0)

    arr = seq
    arr = [x for x in arr if not np.isnan(x)]
    for i in range(1, len(arr)):
        _k_w = (w - 1) / (len(arr) - 1)

        _h_1 = int((arr[i - 1] - _min) * _k_h)
        _h = int((arr[i] - _min) * _k_h)

        _w_1 = int((i - 1) * _k_w)
        _w = int(i * _k_w)

        if log: print(7777, i, arr[i], arr[i - 1], len(arr), " | ", _k_w, _k_h, " | ", _min, _max, " | []", _w_1, _h_1,
                      ", ", _w, _h, "]")

        shape = [(_w_1, _h_1), (_w, _h)]
        img1.line(shape, fill="red", width=0)

    return img


show_N_values = 40
show_it_only_once = True
show_it_only_per_ticker = {ticker: True for ticker in tickers}
print(show_it_only_per_ticker)
df_in = pd.DataFrame()
df_out = pd.DataFrame()

# Do it for all shares
for ticker in tickers:
    df_in = df_all_in[ticker]  # get df for ticker - input
    df_out = df_all_out[ticker][["datetime", "close"]]  # get df for ticker - output

    # join features and target by datetime and two timeframes D1, W1
    packed_w1 = list(zip(df_out.datetime, df_out.close))
    df_in['target'] = [[_close_w1 for _date_w1, _close_w1 in packed_w1 if _date_d1 < _date_w1] for _date_d1 in
                       df_in["datetime"]]  # < - as W1 is formed on next working day - so not <=
    df_in['target'] = [el[0] if len(el) else np.nan for el in df_in["target"]]
    df_in['target_datetime'] = [[_date_w1 for _date_w1, _close_w1 in packed_w1 if _date_d1 < _date_w1] for _date_d1 in
                                df_in["datetime"]]  # < - as W1 is formed on next working day - so not <=
    df_in['target_datetime'] = [el[0] if len(el) else np.nan for el in df_in["target_datetime"]]

    # create sequence
    df_seq = pd.DataFrame(columns=["sequence", "target"])
    # create sequence for dates - for analytics
    df_seq_dates = pd.DataFrame(columns=["sequence_dates", "target_date"])

    # prepare values for sequence
    packed_d1 = list(zip(df_in.datetime, df_in.close))
    packed_w1 = list(zip(df_out.datetime, df_out.close))
    j = 0
    for i in range(len(df_out)):
        _date_w1, _close_w1 = packed_w1[i][0], packed_w1[i][1]
        arr = []
        arr_dates = []
        while True:
            _date_d1, _close_d1 = packed_d1[j][0], packed_d1[j][1]
            #         print(i, j, _date_d1, _date_w1, _date_d1 < _date_w1, arr, len(arr))
            if _date_d1 < _date_w1:
                arr.append(_close_d1)
                arr_dates.append(_date_d1.date())  # .date() as we work with D1, for others timeframes - remove .date()
            else:
                break
            j += 1
            if j >= len(df_in): break
        df_seq.loc[len(df_seq.index)] = [arr, _close_w1]
        df_seq_dates.loc[len(df_seq_dates.index)] = [arr_dates, _date_w1]

    # calc len of sequencies
    df_seq['len_seq'] = df_seq.apply(lambda row: len(row["sequence"]), axis=1)
    df_seq_dates['len_seq_dates'] = df_seq_dates.apply(lambda row: len(row["sequence_dates"]), axis=1)

    df = df_out.copy()
    df["target"] = df["close"]  # it was already shifted!!!
    df["sequence"] = df_seq["sequence"]
    df["len_seq"] = df_seq["len_seq"]
    df["sequence_dates"] = df_seq_dates["sequence_dates"]
    df["len_seq_dates"] = df_seq_dates["len_seq_dates"]

    df['pc'] = df['close'].pct_change()  # percent change
    df['target_class'] = [1 if el >= 0 else 0 for el in df["pc"]]  # class: 1 - up, 0 - down

    for i in range(len(df) - 1):
        if i < weeks_per_image: continue  # get N weeks

        # get all sequences from begin
        seq_all = df["sequence"].loc[:i].sum()  # for fast and slow SMA

        # if we haven't enough data
        if weeks_per_image * 5 > len(seq_all): continue

        # now we have sequence and target, so we can add fast and slow SMA
        sma_fast = pd.DataFrame(seq_all, columns=["sma_fast"]).rolling(period_sma_fast).mean()  # add SMA fast
        sma_slow = pd.DataFrame(seq_all, columns=["sma_slow"]).rolling(period_sma_slow).mean()  # add SMA slow
        sma_fast = sma_fast["sma_fast"].tolist()
        sma_slow = sma_slow["sma_slow"].tolist()

        _date = df["datetime"].loc[i]
        _target_class = df["target_class"].loc[i]
        target = df["target"].loc[i]
        seq = df["sequence"].loc[i - weeks_per_image + 1:i].sum()  # get N previous sequences

        # this code is equal one row upper))
        #         seq = []
        #         for j in range(weeks_per_image): # get N previous sequences
        #             print(555, i, j, i-weeks_per_image+j+1, seq, df["sequence"].loc[i-weeks_per_image+j+1])
        #             seq = seq + df["sequence"].loc[i-weeks_per_image+j+1]  # extend
        #             print(f"\tNow: {seq} => Target: {target}")

        # now we can draw image
        # image generation for neural network training/test

        # generate image
        img = generate_img_with_Nan(sma_fast, sma_slow, seq)

        # show 3 last images
        if i in list(range(len(df) - 4,
                           len(df))) or i <= 4:  # show once image for every ticker  show_it_only_per_ticker[ticker] and
            print(ticker, _date)
            print("seq:", seq[len(seq) - 10:], "=>", "target:", target, "=>", "target_class:", _target_class)
            print(sma_fast[len(sma_fast) - 10:])
            print(sma_slow[len(sma_slow) - 10:])
            plt.imshow(img)
            plt.show()
            show_it_only_per_ticker[ticker] = False

        # let's put it in folder by class
        _date_str = _date.strftime("%Y_%m_%d_%H_%M_%S")
        _filename = f"{ticker}-{timeframe_in}-{_date_str}.png"
        _path = os.path.join("NN", f"training_dataset_{timeframe_in}")

        # images classification
        if _target_class:
            _path = os.path.join(_path, "1")
        else:
            _path = os.path.join(_path, "0")

        img.save(os.path.join(_path, _filename))

        if i == 7: break
    break
import tensorflow as tf
from tensorflow import keras
from tensorflow import config

print("Num GPUs Available: ", len(config.list_physical_devices('GPU')))
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Rescaling
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, LambdaCallback

seed = 777  # to produce similar result in each run - sets value for random seed  # 42++ 77+
data_dir = os.path.join("NN", f"training_dataset_{timeframe_in}")  # folder with data
num_classes = 2  # total classes
epochs = 20  # number of epochs
batch_size = 10  # batch size
img_height, img_width = draw_size, draw_size  # images size
input_shape = (img_height, img_width, 3)  # dimention of image 3 - RGB
keras.utils.set_random_seed(seed)
model = Sequential([
    Rescaling(1. / 255),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes)
])
# version with Adam optimization is a stochastic gradient descent method
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
# Train data
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    #     shuffle=True,
    #     shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size)
file_paths_train_ds = train_ds.file_paths
print(file_paths_train_ds[:10], "len:", len(file_paths_train_ds))
class_0, class_1 = 0, 0
for _filename in file_paths_train_ds:
    if _filename.find("/0/") >= 0: class_0 += 1
    if _filename.find("/1/") >= 0: class_1 += 1
print(f"Train Data - Found class==0 images {class_0}")
print(f"Train Data - Found class==1 images {class_1}")
print(f"Train Data - Total class==1 + Class==0 images {class_1 + class_0}")
print(train_ds.class_names)
# Validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    #     shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size)
val_ds_future = val_ds
file_paths_val_ds = val_ds_future.file_paths
print(file_paths_val_ds[:10], "len:", len(file_paths_val_ds))
class_0, class_1 = 0, 0
for _filename in file_paths_val_ds:
    if _filename.find("/0/") >= 0: class_0 += 1
    if _filename.find("/1/") >= 0: class_1 += 1
print(f"Validation Data - Found class==0 images {class_0}")
print(f"Validation Data - Found class==1 images {class_1}")
# some code - in process of writing
elems_in_both_lists = list(set(file_paths_train_ds) & set(file_paths_val_ds))
print(elems_in_both_lists[:10], "len:", len(elems_in_both_lists))
print(val_ds.class_names)
# Train data
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    labels='inferred',
    label_mode='int',
    #     seed=123,
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# check train data
file_paths_train_ds = train_ds.file_paths
# print(file_paths_train_ds)
class_0, class_1 = 0, 0
for _filename in file_paths_train_ds:
    if _filename.find("/0/") >= 0: class_0 += 1
    if _filename.find("/1/") >= 0: class_1 += 1
print(f"Train Data - Found class==0 images {class_0}")
print(f"Train Data - Found class==1 images {class_1}")
print(f"Train Data - Total class==1 + Class==0 images {class_1 + class_0}")
# Validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    labels='inferred',
    label_mode='int',
    #     seed=123,
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# check validation data
file_paths_val_ds = val_ds_future.file_paths
# print(file_paths_val_ds)
class_0, class_1 = 0, 0
for _filename in file_paths_val_ds:
    if _filename.find("/0/") >= 0: class_0 += 1
    if _filename.find("/1/") >= 0: class_1 += 1
print(f"Validation Data - Found class==0 images {class_0}")
print(f"Validation Data - Found class==1 images {class_1}")
print(data_dir)
# Train, Validation data
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + "/",
    validation_split=0.2,
    subset="both",
    # seed=123,
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# check train data
file_paths_train_ds = train_ds.file_paths
# print(file_paths_train_ds)
class_0, class_1 = 0, 0
for _filename in file_paths_train_ds:
    if _filename.find("/0/") >= 0: class_0 += 1
    if _filename.find("/1/") >= 0: class_1 += 1
print(f"Train Data - Found class==0 images {class_0}")
print(f"Train Data - Found class==1 images {class_1}")
print(f"Train Data - Total class==1 + Class==0 images {class_1 + class_0}")
# check validation data
file_paths_val_ds = val_ds_future.file_paths
# print(file_paths_val_ds)
class_0, class_1 = 0, 0
for _filename in file_paths_val_ds:
    if _filename.find("/0/") >= 0: class_0 += 1
    if _filename.find("/1/") >= 0: class_1 += 1
print(f"Validation Data - Found class==0 images {class_0}")
print(f"Validation Data - Found class==1 images {class_1}")
# Train, Validation data
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + "/",
    validation_split=0.2,
    subset="both",
    seed=seed,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# check train data
file_paths_train_ds = train_ds.file_paths
# print(file_paths_train_ds)
class_0, class_1 = 0, 0
for _filename in file_paths_train_ds:
    if _filename.find("/0/") >= 0: class_0 += 1
    if _filename.find("/1/") >= 0: class_1 += 1
print(f"Train Data - Found class==0 images {class_0}")
print(f"Train Data - Found class==1 images {class_1}")
print(f"Train Data - Total class==1 + Class==0 images {class_1 + class_0}")
# check validation data
file_paths_val_ds = val_ds_future.file_paths
# print(file_paths_val_ds)
class_0, class_1 = 0, 0
for _filename in file_paths_val_ds:
    if _filename.find("/0/") >= 0: class_0 += 1
    if _filename.find("/1/") >= 0: class_1 += 1
print(f"Validation Data - Found class==0 images {class_0}")
print(f"Validation Data - Found class==1 images {class_1}")
folder_NN_timeframe = os.path.join(folder_NN, f"training_dataset_{timeframe_in}")
folder_NN_timeframe_class_1 = os.path.join(folder_NN_timeframe, "1")
folder_NN_timeframe_class_0 = os.path.join(folder_NN_timeframe, "0")
onlyfiles_1 = [f for f in os.listdir(folder_NN_timeframe_class_1) if
               os.path.isfile(os.path.join(folder_NN_timeframe_class_1, f))]  # filenames for class == 1
onlyfiles_0 = [f for f in os.listdir(folder_NN_timeframe_class_0) if
               os.path.isfile(os.path.join(folder_NN_timeframe_class_0, f))]  # filenames for class == 0
print("Total images for class 1: ", len(onlyfiles_1), "for class 0: ", len(onlyfiles_0))
train_validation_split = 0.2
train_size_class_1 = int(len(onlyfiles_1) * (1 - train_validation_split))
val_size_1 = len(onlyfiles_1) - train_size_class_1
train_size_class_0 = int(len(onlyfiles_0) * (1 - train_validation_split))
val_size_0 = len(onlyfiles_0) - train_size_class_0
from random import shuffle

shuffle(onlyfiles_1)
shuffle(onlyfiles_0)

# create folders for validation images
folder_NN_timeframe = None
folder = folder_NN_valid
if not os.path.exists(folder): os.makedirs(folder)
for timeframe in [timeframe_in, ]:  # timeframe_out
    _folder = os.path.join(folder, f"training_dataset_{timeframe}")
    folder_NN_timeframe = _folder
    if not os.path.exists(_folder): os.makedirs(_folder)
    for _class in [0, 1]:
        _folder_class = os.path.join(_folder, f"{_class}")
        if not os.path.exists(_folder_class): os.makedirs(_folder_class)
import shutil

# move validation data for class 1
for i in range(val_size_1):
    filename = onlyfiles_1[i]
    _from = os.path.join(folder_NN, f"training_dataset_{timeframe_in}")
    _from = os.path.join(os.path.join(_from, '1'), filename)
    _to = os.path.join(folder_NN_valid, f"training_dataset_{timeframe_in}")
    _to = os.path.join(os.path.join(_to, '1'), filename)
    shutil.move(_from, _to)

# move validation data for class 0
for i in range(val_size_0):
    filename = onlyfiles_0[i]
    _from = os.path.join(folder_NN, f"training_dataset_{timeframe_in}")
    _from = os.path.join(os.path.join(_from, '0'), filename)
    _to = os.path.join(folder_NN_valid, f"training_dataset_{timeframe_in}")
    _to = os.path.join(os.path.join(_to, '0'), filename)
    shutil.move(_from, _to)
folder_NN_timeframe = os.path.join(folder_NN, f"training_dataset_{timeframe_in}")
folder_NN_timeframe_class_1 = os.path.join(folder_NN_timeframe, "1")
folder_NN_timeframe_class_0 = os.path.join(folder_NN_timeframe, "0")
onlyfiles_1 = [f for f in os.listdir(folder_NN_timeframe_class_1) if
               os.path.isfile(os.path.join(folder_NN_timeframe_class_1, f))]  # filenames for class == 1
onlyfiles_0 = [f for f in os.listdir(folder_NN_timeframe_class_0) if
               os.path.isfile(os.path.join(folder_NN_timeframe_class_0, f))]  # filenames for class == 0
print("Total images for class 1: ", len(onlyfiles_1), "for class 0: ", len(onlyfiles_0))
folder_NN_timeframe = os.path.join(folder_NN_valid, f"training_dataset_{timeframe_in}")
folder_NN_timeframe_class_1 = os.path.join(folder_NN_timeframe, "1")
folder_NN_timeframe_class_0 = os.path.join(folder_NN_timeframe, "0")
onlyfiles_1 = [f for f in os.listdir(folder_NN_timeframe_class_1) if
               os.path.isfile(os.path.join(folder_NN_timeframe_class_1, f))]  # filenames for class == 1
onlyfiles_0 = [f for f in os.listdir(folder_NN_timeframe_class_0) if
               os.path.isfile(os.path.join(folder_NN_timeframe_class_0, f))]  # filenames for class == 0
print("Total images for class 1: ", len(onlyfiles_1), "for class 0: ", len(onlyfiles_0))
data_dir = os.path.join(folder_NN, f"training_dataset_{timeframe_in}")
# Train data
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# check train data
file_paths_train_ds = train_ds.file_paths
# print(file_paths_train_ds)
class_0, class_1 = 0, 0
for _filename in file_paths_train_ds:
    if _filename.find("/0/") >= 0: class_0 += 1
    if _filename.find("/1/") >= 0: class_1 += 1
print(f"Train Data - Found class==0 images {class_0}")
print(f"Train Data - Found class==1 images {class_1}")
print(f"Train Data - Total class==1 + Class==0 images {class_1 + class_0}")
data_dir = os.path.join(folder_NN_valid, f"training_dataset_{timeframe_in}")
# Validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# check validation data
file_paths_val_ds = val_ds_future.file_paths
# print(file_paths_val_ds)
class_0, class_1 = 0, 0
for _filename in file_paths_val_ds:
    if _filename.find("/0/") >= 0: class_0 += 1
    if _filename.find("/1/") >= 0: class_1 += 1
print(f"Validation Data - Found class==0 images {class_0}")
print(f"Validation Data - Found class==1 images {class_1}")
# here we will save all our trained models
list_for_models = []


class SaveAllModels(tf.keras.callbacks.Callback):

    def __init__(self, external_list):
        self.list_obj = external_list

    def on_epoch_end(self, epoch, logs=None):
        self.list_obj.append({
            "model": self.model,
            "loss": logs['loss'],
            "accuracy": logs['accuracy'],
            "val_loss": logs['val_loss'],
            "val_accuracy": logs['val_accuracy'],
        })


save_all_models_callback = SaveAllModels(list_for_models)
callbacks = [
    save_all_models_callback,  # ModelCheckpoint(" 'model_{epoch:1d}.hdf5'"),
]
keras.utils.set_random_seed(seed)
model = Sequential([
    Rescaling(1. / 255),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes)
])

# version with Adam optimization is a stochastic gradient descent method
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(),
    #     loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'])
# starting the learning process
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
# add grid
ax = fig.gca()
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(color='gray', linestyle='--', which="both", alpha=0.5)
# plt.grid(color='gray', linestyle='--', which="both")  # Specify grid with line attributes
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
# add grid
ax = fig.gca()
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(color='gray', linestyle='--', which="both", alpha=0.5)
# plt.grid(color='gray', linestyle='--')  # Specify grid with line attributes
plt.savefig("Training and Validation Accuracy and Loss.png", dpi=150)

plt.show()