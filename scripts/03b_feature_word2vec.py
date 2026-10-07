"""
第三步 ③ 特征提取 · 路线二：词嵌入（Word Embedding Approach）
==========================================================
原理：在本地语料上训练 Word2Vec 静态向量（Skip-gram），把"语义空间"中
      与种子词相近的词都算作短视表达，突破词典覆盖不全的限制。
设计：
  - 种子集合 S：短视锚词（同典一）；参照锚词 L：长期锚词
  - 文档向量采用 TF-IDF 加权平均（第三代 RoBERTa 算力要求高，本科条件不具）
  - 两条候选测度：
      EmbShort  = Σ tfidf(w)·maxcos(w, S) / Σ tfidf(w)         <｜hy_place▁holder▁no▁813｜>锚相似度
      EmbNet    = EmbShort − EmbLong                           净短视相似度（推荐）
输出：results/routeB_word2vec.csv
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "results")
DIAG = pd.read_csv(os.path.join(RES, "extraction_diagnostics.csv"), dtype={"code": str})
DIAG = DIAG[DIAG["tokens_clean"] > 0].reset_index(drop=True)

SEED_SHORT = ["短期", "眼前", "尽快", "尽早", "立刻", "马上", "权宜", "应急",
              "临时", "暂时", "立竿见影", "速成"]
SEED_LONG = ["长期", "长远", "持续", "战略", "可持续", "布局", "核心竞争力",
             "转型升级", "基础研究", "沉淀"]

docs, keys = [], []
for _, d in DIAG.iterrows():
    p = os.path.join(PROC, f"{d['code']}_{int(d['year'])}_clean.txt")
    if os.path.exists(p):
        docs.append(open(p, encoding="utf-8").read())
        keys.append((d["code"], int(d["year"])))
print(f"[INFO] 参与训练语料 {len(docs)} 篇")

sentences = [t.split() for t in docs]
w2v = Word2Vec(sentences=sentences, vector_size=120, window=6, min_count=4,
               sg=1, negative=8, workers=8, seed=2026, epochs=12)
w2v.save(os.path.join(RES, "word2vec_md&a.model"))
print(f"[OK] Word2Vec 词表 {len(w2v.wv)} 词，维度 {w2v.vector_size}")


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def seed_vec(seeds):
    vs = [unit(w2v.wv[s]) for s in seeds if s in w2v.wv]
    return unit(np.mean(vs, axis=0)) if vs else None


# 以 TF-IDF 加权的"最大相似度"文档得分为主，避免单纯平均被高频虚词稀释
vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2, max_df=0.85,
                      max_features=60000)
tfidf = vec.fit_transform([" ".join(s) for s in sentences])
vocab = np.array(vec.get_feature_names_out())

n_missing = sum(1 for w in vocab if w not in w2v.wv)
print(f"[INFO] TF-IDF 词表 {len(vocab)} 词，其中 {n_missing} 词无向量（将跳过）")

# 相似度矩阵准备：只算有向量的词
have = np.array([w in w2v.wv for w in vocab])
V = np.vstack([unit(w2v.wv[w]) if have[i] else np.zeros(w2v.vector_size)
               for i, w in enumerate(vocab)])

v_short, v_long = seed_vec(SEED_SHORT), seed_vec(SEED_LONG)
sim_short = V @ v_short
sim_long = V @ v_long

rows = []
for idx, (code, year) in enumerate(keys):
    row = tfidf[idx]
    _, j = row.nonzero()
    w = row.data
    mask = have[j]
    if not mask.any():
        continue
    jj, ww = j[mask], w[mask]
    denom = ww.sum()
    emb_s = float((ww * sim_short[jj]).sum() / denom)
    emb_l = float((ww * sim_long[jj]).sum() / denom)
    rows.append({"code": code, "year": year,
                 "EmbShort": round(emb_s, 6), "EmbLong": round(emb_l, 6),
                 "EmbNet": round(emb_s - emb_l, 6)})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "routeB_word2vec.csv"), index=False, encoding="utf-8-sig")
print(f"[OK] 词嵌入路线完成：{len(df)} 篇")
print(df[["EmbShort", "EmbLong", "EmbNet"]].describe().to_string())

present_s = [w for w in SEED_SHORT if w in w2v.wv]
present_l = [w for w in SEED_LONG if w in w2v.wv]
missing = [w for w in SEED_SHORT + SEED_LONG if w not in w2v.wv]
print(f"\n种子词覆盖：短视 {len(present_s)}/{len(SEED_SHORT)}，长期 {len(present_l)}/{len(SEED_LONG)}"
      f"；低于 min_count 而未入词表：{missing}")
print("\n与短视种子最相似的 25 个词（核对语义空间是否合理）：")
try:
    for w, sc in w2v.wv.most_similar(present_s, topn=25):
        print(f"   {w}\t{sc:.3f}")
except Exception as e:
    print("   ", e)
print("\n与长期锚词最相似的 15 个词：")
try:
    for w, sc in w2v.wv.most_similar(present_l, topn=15):
        print(f"   {w}\t{sc:.3f}")
except Exception as e:
    print("   ", e)
