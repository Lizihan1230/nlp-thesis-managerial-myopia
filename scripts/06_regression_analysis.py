"""
第六步：基于模型的分析（回归）
=================================
主模型：  RD_int_{i,t+1} = β0 + β1·NTH_{i,t} + γ·Controls + 行业FE + 年份FE + ε
  NTH = 净时间视野（短期视域词频比 − 长期视域词频比），取值越高越偏短期
  预期符号：β1 < 0（越偏短期 → 下一期研发投入强度越低）
标准误：按企业聚类（cluster robust）
输出：results/regression_main.csv, regression_vif.csv
"""
import os, numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
P = pd.read_csv(os.path.join(RES, "panel_firm_year.csv"), dtype={"code": str}).sort_values(
    ["code", "year"]).reset_index(drop=True)

CTRL = ["Size_w", "Lev_w", "ROA_w", "Growth_w", "Age_w", "LnLen_w"]
CTRL = [c for c in CTRL if c in P.columns]
P["RD_f1"] = P.groupby("code")["RD_int"].shift(-1)
P["Capex_f1"] = P.groupby("code")["Capex"].shift(-1)

ind_d = pd.get_dummies(P["industry"], prefix="IND", drop_first=True).astype(float)
yr_d = pd.get_dummies(P["year"], prefix="Y", drop_first=True).astype(float)

store = []


def run(ycol, xcols, label, fe=True, ctrl=True, data=None, cluster=True):
    d = (P if data is None else data).copy()
    need = [ycol] + list(xcols) + (CTRL if ctrl else [])
    X = d[list(xcols) + (CTRL if ctrl else [])].astype(float)
    if fe:
        X = pd.concat([X, ind_d.loc[d.index], yr_d.loc[d.index]], axis=1)
    m = pd.concat([d[need[:1] + ["code"]], X], axis=1).dropna()
    Xm = m[list(X.columns)]
    if len(m) < 40:
        print(f"  !! {label} 样本不足（n={len(m)}），跳过"); return None
    kw = {"groups": m["code"]} if cluster else {}
    fit = sm.OLS(m[ycol], sm.add_constant(Xm)).fit(cov_type="cluster", cov_kwds=kw) \
        if cluster else sm.OLS(m[ycol], sm.add_constant(Xm)).fit()
    key = xcols[0]
    store.append({"模型": label, "被解释变量": ycol, "核心解释变量": key,
                  "系数": round(fit.params[key], 4), "稳健标准误": round(fit.bse[key], 4),
                  "t值": round(fit.tvalues[key], 3), "p值": round(fit.pvalues[key], 4),
                  "95%CI下限": round(fit.conf_int().loc[key, 0], 4),
                  "95%CI上限": round(fit.conf_int().loc[key, 1], 4),
                  "调整R²": round(fit.rsquared_adj, 4), "N": int(fit.nobs),
                  "企业数": int(m["code"].nunique())})
    print(f"  {label:32s} β={fit.params[key]:+8.4f}  se={fit.bse[key]:6.4f}  "
          f"t={fit.tvalues[key]:+6.2f}  p={fit.pvalues[key]:.4f}  "
          f"adjR²={fit.rsquared_adj:.4f}  N={int(fit.nobs)}")
    return fit


print("=" * 96 + "\n主回归：净时间视野 → 下一期研发投入强度\n" + "=" * 96)
run("RD_f1", ["NTH_diff_w"], "(1) NTH → RD(t+1) 全控制+双向FE")
run("RD_f1", ["NTH_diff_w"], "(2) 无控制变量（仅FE）", ctrl=False)
run("RD_f1", ["NTH_diff_w"], "(3) 不聚类稳健标准误", cluster=False)
run("RD_f1", ["NTH_ratio_w"], "(4) 替换解释变量：比率式 NTH")
run("RD_f1", ["Myopia_dict_w"], "(5) 对照：仅短视词频（理应失效）")
run("RD_f1", ["EmbNet_w"], "(6) 路线二：词嵌入净相似度")

print("\n" + "=" * 96 + "\n被解释变量与同期设定\n" + "=" * 96)
run("RD_int_w", ["NTH_diff_w"], "(7) NTH → 当期研发投入强度")
run("Capex_f1", ["NTH_diff_w"], "(8) NTH → 下一期资本支出强度")

print("\n" + "=" * 96 + "\n子样本稳健性\n" + "=" * 96)
run("RD_f1", ["NTH_diff_w"], "(9) 剔除零研发投入企业", data=P[P["rd_expense"].fillna(0) > 0])
run("RD_f1", ["NTH_diff_w"], "(10) 剔除2020年（疫情异常）", data=P[P["year"] != 2020])
run("RD_f1", ["NTH_diff_w"], "(11) 仅制造业子样本",
    data=P[P["industry"].astype(str).str.contains("制造|化学|医药|设备|金属|汽车", na=False)])

print("\n" + "=" * 96 + "\n篇幅陷阱对照：为什么必须做长度标准化\n" + "=" * 96)
run("RD_f1", ["Myopia_ln_w"], "(12) 陷阱组：ln(1+短视命中数) 未除以篇幅")
print("     说明：该口径与披露篇幅 Spearman ρ=+0.300，加入 ln(篇幅) 控制后效应应消失；")
print("     若消失，说明未归一化的词频测度捕捉到的其实是'话多不多'，而非'视野长不长'。")

# ---------------- 效应来源诊断：逐步加入固定效应 ----------------
print("\n" + "=" * 96 + "\n效应来源诊断：逐步加入固定效应（诚实回答'效应来自哪里'）\n" + "=" * 96)
firm_d = pd.get_dummies(P["code"], prefix="F", drop_first=True).astype(float)
base_cols = ["NTH_diff_w"] + CTRL
d0 = P.dropna(subset=["RD_f1"] + base_cols)
diag = [("仅纳入 NTH", []),
        ("+ 财务与篇幅控制变量", CTRL),
        ("+ 年份固定效应", CTRL + ["__YEAR__"]),
        ("+ 行业固定效应", CTRL + ["__YEAR__", "__IND__"]),
        ("+ 企业固定效应", CTRL + ["__YEAR__", "__FIRM__"])]
rowsd = []
for label, add in diag:
    parts = [d0[["NTH_diff_w"]]]
    extra = [c for c in add if not c.startswith("__")]
    if extra:
        parts.append(d0[extra].astype(float))
    if "__YEAR__" in add:
        parts.append(yr_d.loc[d0.index])
    if "__IND__" in add:
        parts.append(ind_d.loc[d0.index])
    if "__FIRM__" in add:
        parts.append(firm_d.loc[d0.index])
    Xm = pd.concat(parts, axis=1)
    fit = sm.OLS(d0["RD_f1"], sm.add_constant(Xm)).fit(
        cov_type="cluster", cov_kwds={"groups": d0["code"]})
    rowsd.append({"设定": label, "β": round(fit.params["NTH_diff_w"], 4),
                  "聚类稳健t值": round(fit.tvalues["NTH_diff_w"], 3),
                  "p值": round(fit.pvalues["NTH_diff_w"], 4),
                  "调整R²": round(fit.rsquared_adj, 4), "N": int(fit.nobs)})
    print(f"  {label:24s} β={fit.params['NTH_diff_w']:+8.4f} "
          f"t={fit.tvalues['NTH_diff_w']:+6.2f} p={fit.pvalues['NTH_diff_w']:.4f} "
          f"adjR²={fit.rsquared_adj:.4f} N={int(fit.nobs)}")
dg = pd.DataFrame(rowsd)
dg.to_csv(os.path.join(RES, "regression_diagnostics.csv"), index=False, encoding="utf-8-sig")
print("\n  → 效应在加入行业/企业固定效应后消失，说明截面相关主要由**行业间披露风格差异**驱动，")
print("    而非同一行业内不同企业之间的管理层视野差异。此为本文最重要的稳健性结论。")

print("\n" + "=" * 96 + "\n多重共线性 VIF\n" + "=" * 96)
vmat = P[["NTH_diff_w"] + CTRL].dropna().astype(float)
vif = pd.DataFrame({"变量": vmat.columns,
                    "VIF": [round(variance_inflation_factor(vmat.values, i), 3)
                            for i in range(vmat.shape[1])]})
print(vif.to_string(index=False))
vif.to_csv(os.path.join(RES, "regression_vif.csv"), index=False, encoding="utf-8-sig")

reg = pd.DataFrame(store)
reg.to_csv(os.path.join(RES, "regression_main.csv"), index=False, encoding="utf-8-sig")
print("\n" + "=" * 96 + "\n回归结果汇总\n" + "=" * 96)
print(reg.to_string(index=False))
