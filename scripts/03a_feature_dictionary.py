"""
第三步 ③ 特征提取 · 路线一：词典法（Dictionary Approach）
==========================================================
同时产出 **v1.0（无语境约束）** 与 **v2.0（邻近剔除）** 两套词频，
用于展示"内容效度检验 → 回到第三步修订词典"的迭代过程。

v2.0 匹配规则：
  命中词 w 在位置 i 计入，当且仅当
    toks[i-1] ∉ 禁前邻词(w)  且  toks[i+1] ∉ 禁后邻词(w)
四套输出口径：
  Myopia_v1       命中/有效词数×100           （v1.0 词典，对照用）
  Myopia_dict     命中/有效词数×100           （v2.0 词典，主测度）
  Myopia_allw     命中/全部词数×1000          （分母稳健性）
  Myopia_ln       ln(1+命中)                 （篇幅稳健性）
输出：results/routeA_dict.csv        文档级词频与测度
      results/concordance_all.csv    全部索引行
      results/dict_v1_vs_v2.csv      v1/v2 逐词命中对比（论文表格素材）
"""
import os, numpy as np, pandas as pd, collections, jieba

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "results")
DIAG = os.path.join(RES, "extraction_diagnostics.csv")


def load_lexicon(path):
    """返回 {word: (facet, set(禁后邻), set(禁前邻))}"""
    lex = {}
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith("---"):
            continue
        parts = ln.split()
        if len(parts) >= 2:
            nxt = set([x for x in parts[2].split("|") if x]) if len(parts) > 2 else set()
            prv = set([x for x in parts[3].split("|") if x]) if len(parts) > 3 else set()
            lex[parts[0]] = (parts[1], nxt, prv)
    return lex


LEX1 = load_lexicon(os.path.join(BASE, "data", "lexicon", "shortterm_lexicon.txt"))
LEX2 = load_lexicon(os.path.join(BASE, "data", "lexicon", "shortterm_lexicon_v2.txt"))
LEXL = load_lexicon(os.path.join(BASE, "data", "lexicon", "longterm_lexicon.txt"))
for _w in LEXL:
    jieba.add_word(_w, freq=10000)
FACETS = sorted(set(v[0] for v in LEX2.values()))
print(f"[INFO] v1.0 词典 {len(LEX1)} 词；v2.0 词典 {len(LEX2)} 词；侧面 {FACETS}")


def count_matches(toks, lex, strict):
    """strict=True 时启用邻近剔除。返回 (总命中, 分侧面命中, 索引行)"""
    total = 0
    per = collections.Counter()
    idx = []
    for i, w in enumerate(toks):
        if w not in lex:
            continue
        facet, bnxt, bprv = lex[w]
        if strict:
            # 后邻检查窗口放宽至 2 个 token，降低分词不确定性的影响
            if bnxt and set(toks[i + 1:i + 3]) & bnxt:
                continue
            if bprv and i > 0 and toks[i - 1] in bprv:
                continue
        total += 1
        per[facet] += 1
        idx.append((i, w, facet))
    return total, per, idx


rows, conc = [], []
diag = pd.read_csv(DIAG, dtype={"code": str})
word_hits1, word_hits2 = collections.Counter(), collections.Counter()

for _, d in diag.iterrows():
    code, year = d["code"], int(d["year"])
    base = f"{code}_{year}"
    try:
        all_tok = open(os.path.join(PROC, base + "_tokens.txt"), encoding="utf-8").read().split()
        cln_tok = open(os.path.join(PROC, base + "_clean.txt"), encoding="utf-8").read().split()
    except FileNotFoundError:
        continue
    if len(cln_tok) < 500:
        continue

    h1, _, _ = count_matches(cln_tok, LEX1, strict=False)
    h2, p2, idx2 = count_matches(cln_tok, LEX2, strict=True)
    hl, _, idxl = count_matches(cln_tok, LEXL, strict=True)
    for i, w, f in idx2:
        if i in range(len(cln_tok)):
            word_hits2[w] += 1
    for w in LEX1:                      # v1 逐词命中（用于对比表）
        c = cln_tok.count(w)
        if c:
            word_hits1[w] += c

    row = {"code": code, "year": year,
           "tokens_all": len(all_tok), "tokens_clean": len(cln_tok),
           "hits_v1": h1, "hits_v2": h2, "hits_long": hl,
           "Myopia_v1": round(h1 / len(cln_tok) * 100, 6),
           "Myopia_dict": round(h2 / len(cln_tok) * 100, 6),
           "Myopia_dict_allwords": round(h2 / len(all_tok) * 1000, 6),
           "Myopia_ln": round(np.log1p(h2), 6),
           "n_distinct_words": len(set(cln_tok))}
    row["Rate_short"] = h2 / len(cln_tok) * 100
    row["Rate_long"] = hl / len(cln_tok) * 100
    row["NTH_ratio"] = round((h2 - hl) / max(h2 + hl, 1) * 100, 6)   # 净时间视野(比率式)
    row["NTH_diff"] = round(row["Rate_short"] - row["Rate_long"], 6)  # 净时间视野(差值式)
    row["NTH_log"] = round(np.log1p(h2) - np.log1p(hl), 6)            # 对数式
    for f in FACETS:
        row[f"hits_{f}"] = int(p2.get(f, 0))
        row[f"Myopia_{f}"] = round(p2.get(f, 0) / len(cln_tok) * 100, 6)
    rows.append(row)

    for i, w, f in idxl:
        conc.append({"code": code, "year": year, "word": w, "facet": "L",
                     "left": " ".join(cln_tok[max(0, i - 12):i]),
                     "right": " ".join(cln_tok[i + 1:i + 13])})
    for i, w, f in idx2:
        conc.append({"code": code, "year": year, "word": w, "facet": f,
                     "left": " ".join(cln_tok[max(0, i - 12):i]),
                     "right": " ".join(cln_tok[i + 1:i + 13])})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "routeA_dict.csv"), index=False, encoding="utf-8-sig")
cd = pd.DataFrame(conc)
cd.to_csv(os.path.join(RES, "concordance_all.csv"), index=False, encoding="utf-8-sig")

cmp = pd.DataFrame({"v1.0命中": pd.Series(word_hits1), "v2.0命中": pd.Series(word_hits2)}).fillna(0).astype(int)
cmp["剔除率%"] = (100 * (1 - cmp["v2.0命中"] / cmp["v1.0命中"].replace(0, np.nan))).round(1)
cmp = cmp.sort_values("v1.0命中", ascending=False)
cmp.to_csv(os.path.join(RES, "dict_v1_vs_v2.csv"), encoding="utf-8-sig")

print(f"\n[OK] 词典法完成：{len(df)} 篇文档")
print(f"v1.0 命中合计 {int(df.hits_v1.sum())} 次；v2.0 命中合计 {int(df.hits_v2.sum())} 次 "
      f"（剔除 {100*(1-df.hits_v2.sum()/df.hits_v1.sum()):.1f}%）")
print("\n主测度描述性统计：")
print(df[["Myopia_v1", "Myopia_dict", "Myopia_dict_allwords"]].describe().round(4).to_string())
print("\n逐词 v1→v2 剔除情况（前 20）：")
print(cmp.head(20).to_string())
print(f"短期词命中合计 {int(df.hits_v2.sum())}；长期词命中合计 {int(df.hits_long.sum())}")
print("\n净时间视野 NTH 描述性统计：")
print(df[["Rate_short", "Rate_long", "NTH_ratio", "NTH_diff"]].describe().round(4).to_string())
print(f"\nNTH_ratio<0（偏长期导向）占比：{(df.NTH_ratio<0).mean()*100:.1f}%")
print("\n侧面构成（v2.0）：")
tot = {f: int(df[f"hits_{f}"].sum()) for f in FACETS}
for f, v in tot.items():
    print(f"  {f}: {v:5d} ({v/max(1,sum(tot.values()))*100:5.2f}%)")
