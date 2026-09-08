import torch
import faiss
import numpy as np
from recbole.quick_start import load_data_and_model

# 1. 加载模型
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='saved/StuRec.pth'
)

model.eval()

# 2. 提取 POI (item) embedding
item_num = dataset.item_num
embeddings = []

with torch.no_grad():
    for item_id in range(1, item_num):
        item_tensor = torch.tensor([item_id])
        emb = model.item_embedding(item_tensor)
        emb = emb.cpu().numpy()
        embeddings.append(emb[0])

embeddings = np.array(embeddings).astype('float32')

print("embedding shape:", embeddings.shape)

# 3. 构建 FAISS index
d = embeddings.shape[1]
quantizer = faiss.IndexFlatL2(d)

index = faiss.IndexIVFPQ(
    quantizer,
    d,
    1,      # IVF1
    128,    # PQ128
    4       # x4
)

index.train(embeddings)
index.add(embeddings)

# 4. 保存 index（⚠️路径必须对）
faiss.write_index(
    index,
    "data/NYC/OPQ128,IVF1,PQ128x4.strict.index"
)

print("✅ Index build finished!")
