# partly copied from https://habr.com/ru/post/581932/

# !pip install transformers
# !pip install datasets==1.15.1
# !pip install sentencepiece

import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import Adam
import pandas as pd
from tqdm.auto import trange
import random
import numpy as np


# MODEL
model = 'sberbank-ai/ruT5-large' # ('google/mt5-small')('cointegrated/rut5-small')('sberbank-ai/ruT5-base')
tokenizer = T5Tokenizer.from_pretrained(model)
model = T5ForConditionalGeneration.from_pretrained(model).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# DATA
df_train = pd.read_csv('drive/MyDrive/komi/data/train_example_2/df_train.csv')
df_val = pd.read_csv('drive/MyDrive/komi/data/train_example_2/df_val.csv')
pairs = list(zip(df_train['komi'].tolist(), df_train['rus'].tolist()))

# TRAIN
batch_size = 2
report_steps = 200
epochs = 3

model.train()
losses = []
for epoch in range(epochs):
    print('EPOCH', epoch)
    for i in trange(0, int(len(pairs) / batch_size)):
        batch = pairs[i * batch_size: (i + 1) * batch_size]
        x = tokenizer([p[0] for p in batch], return_tensors='pt', padding=True).to(model.device)
        y = tokenizer([p[1] for p in batch], return_tensors='pt', padding=True).to(model.device)
        y.input_ids[y.input_ids == 0] = -100
        loss = model(
            input_ids=x.input_ids,
            attention_mask=x.attention_mask,
            labels=y.input_ids,
            decoder_attention_mask=y.attention_mask,
            return_dict=True
        ).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if i % report_steps == 0:
            print('step', i, 'loss', np.mean(losses[-report_steps:]))

# SAVE
model.eval()
torch.save(model.state_dict(), 'drive/MyDrive/komi/data/sber_t5_large_adm_media_3epocs.pth')

# EVAL
from tqdm import tqdm
from datasets import load_metric


results = []
for n, row in tqdm(df_val.iterrows()):
    inputs = tokenizer(
      row['komi'],
      return_tensors='pt').to(model.device)
    with torch.no_grad():
        hypotheses = model.generate(**inputs)
    result = tokenizer.decode(hypotheses[0], skip_special_tokens=True)
    results.append(result)


metric = load_metric("bleu")
score = metric.compute(predictions=[i.split() if type(i) != float else ''.split() for i in results], 
                       references=[[i.split()] for i in df_val['rus'].tolist()])
