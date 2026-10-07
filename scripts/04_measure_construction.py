"""
第四步 ④ 指标构建：文本向量 → 企业-年份面板
==============================================
包含：
  1) 计算路线三"路径 B"测度 LDA_Lex（LDA 扩展词典计数比）
  2) 三路测度与财务数据合并
  3) 样本筛选（可穷尽核对）
  4) 变量构造：被解释 / 解释 / 控制变量
  5) 缩尾处理
输出：results/panel_firm_year.csv      主面板
      results/sample_screen.csv        样本筛选过程表（论文用）
"""
import os, re, numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")

# ---------- 1) LDA 扩展词典计数 ----------
PROC = os.path.join(DATA, "processed")
lda_words = [w.strip() for w in
             open(os.path.join(DATA, "lexicon", "lexicon_lda_expanded.txt"), encoding="utf-8")
             if w.strip() and not w.startswith("#")]
diag = pd.read_csv(os.path.join(RES, "extraction_diagnostics.csv"), dtype={"code": str})

lda_lex_rows = []
for _, d in diag.iterrows():
    p = os.path.join(PROC, f"{d['code']}_{int(d['year'])}_clean.txt")
    if not os.path.exists(p):
        continue
    toks = open(p, encoding="utf-8").read().split()
    if len(toks) < 500:
        continue
    vc = pd.Series(toks).value_counts()
    hits = int(vc.reindex(lda_words).fillna(0).sum())
    lda_lex_rows.append({"code": d["code"], "year": int(d["year"]),
                         "LDA_Lex": round(hits / len(toks) * 100, 6),
                         "LDA_hits": hits})
ldalex = pd.DataFrame(lda_lex_rows)

# ---------- 2) 合并三路测度 ----------
A = pd.read_csv(os.path.join(RES, "routeA_dict.csv"), dtype={"code": str})
B = pd.read_csv(os.path.join(RES, "routeB_word2vec.csv"), dtype={"code": str})
C = pd.read_csv(os.path.join(RES, "routeC_lda.csv"), dtype={"code": str})
best_k = pd.read_json(os.path.join(RES, "lda_best.json"), typ="series")["best_k"] \
    if os.path.exists(os.path.join(RES, "lda_best.json")) else None
target_topic = None
if os.path.exists(os.path.join(RES, "lda_best.json")):
    tgt = pd.read_json(os.path.join(RES, "lda_best.json"), typ="series")
    target_topic = int(tgt["target_topic"])

measures = (A.merge(B, on=["code", "year"], how="outer")
             .merge(C, on=["code", "year"], how="outer")
             .merge(ldalex, on=["code", "year"], how="outer"))
if target_topic is not None and f"topic_{target_topic}" in measures.columns:
    measures = measures.rename(columns={f"topic_{target_topic}": "LDA_TopicProp"})

# ---------- 3) 样本筛选 ----------
fin = pd.read_csv(os.path.join(DATA, "financials", "financials_raw.csv"), dtype={"code": str})
frame = pd.read_csv(os.path.join(DATA, "sample_frame", "liaoning_raw.csv"), dtype={"code": str})
frame["listed_year"] = pd.to_datetime(frame["listed_date"], errors="coerce").dt.year
meta = frame[["code", "name", "industry", "market", "regaddr", "listed_year"]].copy()

log = []
def note(step, desc, n):
    log.append({"步骤": step, "处理内容": desc, "剩余观测数": n})

df = fin.copy();                             note(0, "新浪财务库初始企业-年份观测", len(df))
# ① 剔除金融业（报表口径不可比）
fin_ind = ("货币金融", "资本市场服务", "保险业", "其他金融")
mask_fin = meta.set_index("code").loc[df["code"], "industry"].fillna("").apply(
    lambda s: any(k in s for k in fin_ind)).values
df = df[~mask_fin];                          note(1, "剔除金融保险业", len(df))
# ② 剔除 ST / *ST 公司（交易异常、持续经营存在不确定性）
st_codes = set(meta.loc[meta["name"].str.contains(r"\*?ST", na=False), "code"])
df = df[~df["code"].isin(st_codes)];         note(2, "剔除 ST/*ST 公司", len(df))
# ③ 合并文本测度，剔除 MD&A 抽取失败或未抓取到年报的观测
df = df.merge(measures, on=["code", "year"], how="inner")
note(3, "与年报 MD&A 语料成功匹配", len(df))
df = df[df["tokens_clean"].fillna(0) >= 500]; note(4, "剔除 MD&A 有效词数<500 的抽取失败样本", len(df))
df = df[df["raw_chars"].fillna(0) >= 2000] if "raw_chars" in df else df
# ④ 剔除上市当年（年报披露不完整、IPO 效应）
tmp = df.merge(meta[["code", "listed_year"]], on="code", how="left")
df = df[(~df["year"].le(tmp["listed_year"].values)) | tmp["listed_year"].isna().values]
note(5, "剔除上市当年及以前观测", len(df))
# ⑤ 关键变量缺失
df = df[df["revenue"].notna() & df["total_assets"].notna() & (df["revenue"] > 0)]
note(6, "剔除收入/资产缺失或非正观测", len(df))
df = df.merge(meta, on="code", how="left")

# ---------- 4) 变量构造 ----------
df = df.sort_values(["code", "year"]).reset_index(drop=True)
df["RD_int"] = df["rd_expense"].fillna(0) / df["revenue"] * 100          # 研发强度 %
df["Capex"] = df["cash_paid_capex"].fillna(0) / df["total_assets"] * 100  # 资本支出强度 %
df["Size"] = np.log(df["total_assets"])
df["Lev"] = df["total_liab"] / df["total_assets"]
df["ROA"] = df["net_profit"] / df["total_assets"] * 100
df["Growth"] = df.groupby("code")["revenue"].pct_change() * 100
df["Age"] = np.log1p(df["year"] - df["listed_year"])
df["LnLen"] = np.log(df["tokens_clean"])
df["SOE_hint"] = np.nan  # 产权性质未采集，留待后续版本
df["ind_code"] = df["industry"].astype("category").cat.codes

# ---------- 5) 缩尾 1%/99% ----------
wins = ["RD_int", "Capex", "Size", "Lev", "ROA", "Growth", "Age", "LnLen",
        "Myopia_dict", "Myopia_dict_allwords", "Myopia_ln",
        "NTH_diff", "NTH_ratio", "NTH_log",
        "EmbNet", "EmbShort", "LDA_Lex", "LDA_TopicProp"]
wins = [w for w in wins if w in df.columns]
for w in wins:
    lo, hi = df[w].quantile(0.01), df[w].quantile(0.99)
    df[w + "_w"] = df[w].clip(lo, hi)
print("[INFO] 已缩尾变量：", ", ".join(wins))

df.to_csv(os.path.join(RES, "panel_firm_year.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(log).to_csv(os.path.join(RES, "sample_screen.csv"), index=False, encoding="utf-8-sig")

print("\n[样本筛选过程]")
print(pd.DataFrame(log).to_string(index=False))
print(f"\n最终面板：{df['code'].nunique()} 家公司 × {df['year'].nunique()} 年 = {len(df)} 个观测")
print(df.groupby("year").size().to_string())
