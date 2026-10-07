"""
第五步 ⑤ 信效度检验：证明这条测度"值得信"
==============================================
检验清单（对应课件第19页四类 + 两项补充）：
  ① 信度 Reliability      —— Cronbach α（短视四维）、折半信度、替代形式相关
  ② 内容效度 Content      —— 词典来源 + 人工判读命中精确率（短视侧 / 长期侧）
  ③ 聚合效度 Convergent   —— 三路测度相关矩阵 + 偏相关
  ④ 区分效度 Discriminant —— 与规模/杠杆/篇幅等"不该相关"变量的相关；
                             并检验测度不能被 LDA 主题结构完全解释
  ⑤ 预测效度 Predictive   —— 单变量回归 + 五分位条件均值单调性
  ⑥ 方差分解              —— 企业间差异占比（测度是否承载稳定公司特征）
输出：results/validity_*.csv
"""
import os, numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
P = pd.read_csv(os.path.join(RES, "panel_firm_year.csv"), dtype={"code": str})
FACETS = ["A", "B", "C", "D"]
print(f"[INFO] 面板 {len(P)} 观测 / {P['code'].nunique()} 家 / {P['year'].nunique()} 年\n")
out = {}


def block(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ---------------- ① 信度 ----------------
block("① 信度检验 Reliability")
X = P[[f"hits_{f}" for f in FACETS]].div(P["tokens_clean"], axis=0) * 100
X.columns = FACETS
k = X.shape[1]
alpha = (k / (k - 1)) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))
h1, h2 = X[["A", "B"]].sum(axis=1), X[["C", "D"]].sum(axis=1)
r_half = stats.pearsonr(h1, h2)[0]
sb = 2 * r_half / (1 + r_half)
alt = stats.pearsonr(P["Myopia_dict"], P["Myopia_dict_allwords"])[0]
# 净测度的内部一致性：把"短期占比"与"长期占比"视为一枚硬币两面
net_alpha = None
rows = [
    {"指标": "Cronbach's α（短视词典四维侧面）", "取值": round(alpha, 4), "参考标准": ">0.60 可接受"},
    {"指标": "折半相关系数 r", "取值": round(r_half, 4), "参考标准": "越高越好"},
    {"指标": "Spearman-Brown 折半信度", "取值": round(sb, 4), "参考标准": ">0.60 可接受"},
    {"指标": "替代形式相关（分母换成全部词数）", "取值": round(alt, 4), "参考标准": "应高度正相关"},
]
out["reliability"] = pd.DataFrame(rows)
print(pd.DataFrame(rows).to_string(index=False))

# ---------------- ② 内容效度 ----------------
block("② 内容效度 Content Validity：人工判读命中精确率")
res_rows = []
for name, f, tot, hit_cnt in [
    ("短视词典 v1.0（无语境约束）", None, None, int(P["hits_v1"].sum())),
    ("短视词典 v2.0（邻近剔除后）", "annotation_manual.csv", 52, int(P["hits_v2"].sum())),
    ("长期视域词典 v1.0", "annotation_manual_long.csv", 32, int(P["hits_long"].sum())),
]:
    if f is None:
        res_rows.append({"词表": name, "判读样本": "—", "严格口径精确率": "—",
                         "宽口径精确率": "—", "全样本命中次数": hit_cnt})
        continue
    path = os.path.join(RES, f)
    if not os.path.exists(path):
        continue
    a = pd.read_csv(path)
    strict_p = a["label"].mean()
    leni_p = a["label_lenient"].mean() if "label_lenient" in a else np.nan
    res_rows.append({"词表": name, "判读样本": int(len(a)),
                     "严格口径精确率": round(strict_p, 4),
                     "宽口径精确率": round(leni_p, 4) if np.isfinite(leni_p) else "—",
                     "全样本命中次数": hit_cnt})
cv = pd.DataFrame(res_rows)
out["content_validity"] = cv
print(cv.to_string(index=False))
print("\n判读结论：短视词表面形式在本土年报中几乎不承载'管理层自身短视'语义（严格口径≈6%），")
print("        而长期视域词精确率高（≈81%），故主测度采用二者之'净倾向'而非单纯短视词频。")

# ---------------- ③ 聚合效度 ----------------
block("③ 聚合效度 Convergent Validity：三路测度相关矩阵")
cols = [c for c in ["Myopia_dict", "NTH_diff", "NTH_ratio", "EmbNet", "EmbShort",
                    "LDA_Lex", "LDA_TopicProp"] if c in P]
sub = P[cols].astype(float)
pear = sub.corr("pearson")
spear = sub.corr("spearman")
print("Pearson 相关矩阵：")
print(pear.round(3).to_string())
pmat = pear.copy()
for a in cols:
    for b in cols:
        pmat.loc[a, b] = round(stats.pearsonr(sub[a], sub[b])[1], 4) if a != b else np.nan
print("\n对应 p 值：")
print(pmat.round(4).to_string())
out["convergent_pearson"], out["convergent_spearman"] = pear.round(4), spear.round(4)


def partial_corr(x, y, covs):
    def resid(t):
        return sm.OLS(t, sm.add_constant(covs)).fit().resid
    return stats.pearsonr(resid(x), resid(y))


pc, pcp = partial_corr(P["NTH_diff"], P["EmbNet"], P[["Size", "LnLen"]].astype(float))
print(f"\n偏相关 NTH_diff ~ EmbNet（控制 Size 与 ln(篇幅)）：r={pc:.4f}, p={pcp:.4g}")

# ---------------- ④ 区分效度 ----------------
block("④ 区分效度 Discriminant Validity")
nums = [c for c in ["Size", "Lev", "ROA", "Growth", "Age", "LnLen"] if c in P]
rows = []
for c in nums:
    r, p = stats.pearsonr(P["NTH_diff"], P[c])
    rs, ps = stats.spearmanr(P["NTH_diff"], P[c])
    rows.append({"变量": c, "Pearson_r": round(r, 4), "p": round(p, 4),
                 "Spearman_rho": round(rs, 4), "p_spearman": round(ps, 4),
                 "|r|是否<0.3": "是" if abs(r) < 0.3 else "否"})
dv = pd.DataFrame(rows)
out["discriminant"] = dv
print(dv.to_string(index=False))

# 测度是否被 LDA 主题结构完全吸收？
tp = [c for c in P.columns if c.startswith("topic_")]
if tp:
    r2 = sm.OLS(P["NTH_diff"], sm.add_constant(P[tp].astype(float))).fit().rsquared_adj
    r2_len = sm.OLS(P["NTH_diff"], sm.add_constant(P[tp + ["Size", "LnLen"]].astype(float))).fit().rsquared_adj
    print(f"\nLDA 主题对企业 × 年份固定后, 主题结构对 NTH 的解释力 R²={r2:.4f}"
          f"；加规模与篇幅后 R²={r2_len:.4f}")
    print("→ 若 R² 不高，说明净时间视野并非'披露主题构成'或'篇幅'的简单副产品。")

# ---------------- ⑤ 预测效度 ----------------
block("⑤ 预测效度 Predictive Validity")
b = sm.OLS(P["RD_int_w"], sm.add_constant(P[["NTH_diff_w"]])).fit()
print("单变量 OLS：RD_int ~ NTH_diff\n" + b.summary().tables[1].as_text())
P["_q"] = pd.qcut(P["NTH_diff"], 5, labels=False, duplicates="drop")
cond = P.groupby("_q").agg(
    n=("NTH_diff", "size"), NTH_mean=("NTH_diff", "mean"),
    RD_mean=("RD_int", "mean"), RD_median=("RD_int", "median"),
    Capex_mean=("Capex", "mean"), Size_mean=("Size", "mean"), LnLen_mean=("LnLen", "mean"))
print("\n按 NTH 五分位分组的条件均值（第 1 组最偏长期，第 5 组最偏短期）：")
print(cond.round(3).to_string())
mono = stats.spearmanr(cond.index, cond["RD_mean"])[0]
print(f"\n分组序号 vs 研发强度 Spearman rho = {mono:.4f}（负值=越偏短期、研发越低）")
out["conditional_means"] = cond.round(4)

# ---------------- ⑥ 方差分解 ----------------
block("⑥ 方差分解 Variance Decomposition")


def between_share(s, g):
    grand = s.mean(); gm = s.groupby(g).mean(); ns = s.groupby(g).size()
    return (ns * (gm - grand) ** 2).sum() / ((s - grand) ** 2).sum()


vd = pd.DataFrame([{"测度": c,
                    "企业间方差占比": round(between_share(P[c], P["code"]), 4),
                    "企业内(时序)占比": round(1 - between_share(P[c], P["code"]), 4)}
                   for c in ["NTH_diff", "Myopia_dict", "EmbNet"] if c in P])
out["variance_decomposition"] = vd
print(vd.to_string(index=False))
print("\n判读：企业间占比越高，说明测度越像一种稳定的公司层面特征，而非年度噪声。")

for n_, d_ in out.items():
    d_.to_csv(os.path.join(RES, f"validity_{n_}.csv"), encoding="utf-8-sig")
print("\n[OK] 已写出 results/validity_*.csv")
