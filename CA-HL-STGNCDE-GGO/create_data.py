import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


excel_file = r" "
sheets = pd.read_excel(excel_file, sheet_name=None)


sheet_names = list(sheets.keys())

sheet_data = [sheets[sheet_name].iloc[:, 1:].values for sheet_name in sheet_names]
original_sheet_data = [data.copy() for data in sheet_data]

scaler = StandardScaler()
for i in range(len(sheet_data)):
    sheet_data[i] = scaler.fit_transform(sheet_data[i])


window_size =
seq_lenth_y =
step_size =


X = []
y = []


for i in range(0, sheet_data[0].shape[0] - window_size - seq_lenth_y + 1, step_size):
    all_features_window = []

    for j in range():

        feature_window = sheet_data[j][i:i + window_size]

        all_features_window.append(np.expand_dims(feature_window, axis=-1))

    window = np.concatenate(all_features_window, axis=-1)
    X.append(window)
    y.append(original_sheet_data[][i + window_size:i + window_size + seq_lenth_y])

X = np.array(X)
y = np.array(y)
y = np.expand_dims(y, axis=-1)

train_size = int( * len(X))
val_size = int( * len(X))
test_size = len(X) - train_size - val_size

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print("X_train.shape:", X_train.shape,"y_train.shape:",y_train.shape)
print("X_val.shape:", X_val.shape,"y_val.shape:",y_val.shape)
print("X_test.shape:", X_test.shape,"y_test.shape:",y_test.shape)

np.savez('', x=X_train, y=y_train)
np.savez('', x=X_val, y=y_val)
np.savez('', x=X_test, y=y_test)