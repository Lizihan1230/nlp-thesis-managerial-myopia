"""
第三步 ③ 特征提取 · 路线三：主题模型 LDA（Topic Model Approach）
==============================================================
原理：不预设词典，让模型从语料中归纳主题；再用主题与词典的重合度定位
      "短视主题"，取其高概率词构成** LDA 扩展词表 **，回到词典法计数。
      —— 采用课件推荐的"路径 B"：主题用来定词，最终测度仍保持可复现。
K 的确定：困惑度(Perplexity) + 主题连贯性(c_v) + 人工可解释性三重判据。
输出：results/routeC_lda.csv               文档-主题分布
      results/lda_model_selection.csv      Model selection 表
      results/lda_topics.csv              主题词表（供人工命名）
      data/lexicon/lexicon_lda_expanded.txt  LDA 扩展词典
"""
import os, numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.matutils import cossim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "results")
DIAG = pd.read_csv(os.path.join(RES, "extraction_diagnostics.csv"), dtype={"code": str})
DIAG = DIAG[DIAG["tokens_clean"] > 0].reset_index(drop=True)

docs, keys = [], []
for _, d in DIAG.iterrows():
    p = os.path.join(PROC, f"{d['code']}_{int(d['year'])}_clean.txt")
    if os.path.exists(p):
        docs.append(open(p, encoding="utf-8").read())
        keys.append((d["code"], int(d["year"])))
print(f"[INFO] 参与 LDA 的文档 {len(docs)} 篇")

# ---- 词向量准备：先做小 Corpus-specific 的 gensim Dictionary（算 c_v 用）----
texts = [t.split() for t in docs]
gs_dict = Dictionary(texts)
gs_dict.filter_extremes(no_below=3, no_above=0.6, keep_n=30000)
corpus = [gs_dict.doc2bow(t) for t in texts]
print(f"[INFO] gensim 词典 {len(gs_dict)} 词")

seed = ["短期", "眼前", "尽快", "临时", "应急", "权宜", "暂时", "立刻", "短期业绩"]

# ---- K 的选择 ----
sel = []
for K in [5, 8, 10, 12, 15, 18, 20]:
    lda = LdaModel(corpus=corpus, id2word=gs_dict, num_topics=K, random_state=42,
                   passes=8, alpha="asymmetric", eta="auto", chunksize=64, iterations=200)
    per = lda.log_perplexity(corpus)
    coh = CoherenceModel(model=lda, texts=texts, dictionary=gs_dict,
                         coherence="c_v", processes=1).get_coherence()
    top = [[w for w, _ in lda.get_topic_terms(k, topn=15)] for k in range(K)]
    sel.append({"K": K, "perplexity": round(float(per), 4), "coherence_c_v": round(float(coh), 4)})
    print(f"   K={K:>2}  perplexity={per:.3f}  c_v={coh:.4f}", flush=True)

seldf = pd.DataFrame(sel)
seldf.to_csv(os.path.join(RES, "lda_model_selection.csv"), index=False, encoding="utf-8-sig")

best = int(seldf.sort_values("coherence_c_v", ascending=False)["K"].iloc[0])
print(f"[OK] 按 c_v 最优选择 K = {best}")

lda = LdaModel(corpus=corpus, id2word=gs_dict, num_topics=best, random_state=42,
               passes=12, alpha="asymmetric", eta="auto", chunksize=64, iterations=300)

# ---- 输出主题词表，并按与种子词的重合度给每个主题打分 ----
id2w = {v: k for k, v in gs_dict.token2id.items()}
topics, scores = [], []
for k in range(best):
    ws = [(id2w[i], p) for i, p in lda.get_topic_terms(k, topn=25)]
    topics.append(ws)
    ov = sum(p for w, p in ws if w in seed)
    cov = len([w for w, _ in ws if w in seed])
    scores.append({"topic": k, "seed_prob_mass": round(float(ov), 5), "seed_word_cnt": cov,
                   "top_words": " ".join(f"{w}({p:.3f})" for w, p in ws[:15])})
scores = pd.DataFrame(scores).sort_values(["seed_word_cnt", "seed_prob_mass"], ascending=False)
scores.to_csv(os.path.join(RES, "lda_topics.csv"), index=False, encoding="utf-8-sig")
print("\n主题-种子重合度排序：")
print(scores[["topic", "seed_word_cnt", "seed_prob_mass"]].to_string(index=False))

# ---- 文档-主题分布 ----
rowsel = []
for (code, year), bow in zip(keys, corpus):
    dist = dict(lda.get_document_topics(bow, minimum_probability=0))
    rowsel.append({"code": code, "year": year,
                   **{f"topic_{k}": round(float(dist.get(k, 0.0)), 6) for k in range(best)}})
dd = pd.DataFrame(rowsel)
dd.to_csv(os.path.join(RES, "routeC_lda.csv"), index=False, encoding="utf-8-sig")

# ---- 路径 B：以"短视主题"高概率词构造 LDA 扩展词典 ----
TARGET = int(scores["topic"].iloc[0])
SECOND = int(scores["topic"].iloc[1]) if len(scores) > 1 else TARGET
exp_words = []
for k in dict.fromkeys([TARGET, SECOND]):
    exp_words += [id2w[i] for i, p in lda.get_topic_terms(k, topn=40)]
exp_words = list(dict.fromkeys(exp_words))
with open(os.path.join(BASE, "data", "lexicon", "lexicon_lda_expanded.txt"), "w",
          encoding="utf-8") as f:
    f.write("# LDA 扩展词典 v1.0 —— 由 %d 主题 LDA（K=%d）中主题 %d / %d 的高概率词构成\n"
            % (len(docs), best, TARGET, SECOND))
    f.write("# 用途：路线三测度 LDA_Lex 的词表（路径 B：主题只用来定词）\n")
    f.write("\n".join(exp_words) + "\n")
print(f"[OK] LDA 扩展词典 {len(exp_words)} 词 → lexicon_lda_expanded.txt")
print("示例词：", " ".join(exp_words[:30]))
json.dump({"best_k": best, "target_topic": TARGET, "second_topic": SECOND},
          open(os.path.join(RES, "lda_best.json"), "w"))
