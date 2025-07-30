import torch
import pickle

model_type='CQL'
LSTM=True

if ('C51' in model_type or 'QR' in model_type) and LSTM:
    Q_Net = 1
elif 'C51' not in model_type and LSTM:
    Q_Net = 2
elif 'C51' in model_type and not LSTM:
    Q_Net = 3
else:
    Q_Net = 4

print('Q_Net:', Q_Net)