"""生成论文插图 fig/: 流程、词频对比、LDA模型选择、分组均值"""
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(BASE, "results"), os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

# 中文字体
cands = ["/usr/share/fonts/truetype/arphic/uming.ttc",
         "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]
font = next((c for c in cands if os.path.exists(c)), None)
if font:
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
plt.rcParams["axes.unicode_minus"] = False
print("[font]", font or "default")

# 图1 词典命中 v1 → v2（对比主要误命中词）
cmp = pd.read_csv(os.path.join(RES, "dict_v1_vs_v2.csv"), index_col=0)
top = cmp.head(10).sort_values("v1.0命中")
fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(len(top))
ax.barh(y - 0.2, top["v1.0命中"], 0.4, label="v1.0 无语境约束", color="#c0392b")
ax.barh(y + 0.2, top["v2.0命中"], 0.4, label="v2.0 邻近剔除后", color="#2980b9")
ax.set_yticks(y); ax.set_yticklabels(top.index, fontsize=10)
ax.set_xlabel("命中次数"); ax.legend(fontsize=9)
ax.set_title("图3-1　短视词典修订前后的词条命中对比", fontsize=12)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{FIG}/fig3-1_dict_v1_v2.png", dpi=160); plt.close()

# 图2 LDA 模型选择
ld = pd.read_csv(os.path.join(RES, "lda_model_selection.csv"), index_col=0)
fig, ax1 = plt.subplots(figsize=(7, 3.8))
ax1.plot(ld.index, ld["coherence_c_v"], "o-", color="#16a085", label="主题连贯性 c_v")
ax1.set_ylabel("$c_v$ 连贯性", color="#16a085"); ax1.set_xlabel("主题数 K")
ax2 = ax1.twinx()
ax2.plot(ld.index, ld["perplexity"], "s--", color="#8e44ad", label="困惑度 perplexity")
ax2.set_ylabel("perplexity", color="#8e44ad")
ax1.axvline(15, ls=":", color="grey")
ax1.set_title("图3-2　LDA 主题数选择：困惑度与连贯性双判据", fontsize=12)
fig.legend(loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.02))
for s in ("top",): ax1.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{FIG}/fig3-2_lda_selection.png", dpi=160); plt.close()

# 图3 NTH 分布
P = pd.read_csv(os.path.join(RES, "panel_firm_year.csv"), dtype={"code": str})
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.hist(P["NTH_diff"], bins=40, color="#3498db", edgecolor="white")
ax.set_xlabel("净时间视野 NTH_diff"); ax.set_ylabel("企业-年份观测数")
ax.set_title("图3-3　净时间视野的分布（负值＝偏长期导向）", fontsize=12)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{FIG}/fig3-3_nth_dist.png", dpi=160); plt.close()

# 图4 分组均值
cm = pd.read_csv(os.path.join(RES, "validity_conditional_means.csv"), index_col=0)
fig, ax = plt.subplots(figsize=(6.8, 3.6))
x = np.arange(1, len(cm) + 1)
ax.bar(x, cm["RD_mean"], 0.55, color="#e67e22", label="研发强度均值")
ax.plot(x, cm["RD_median"], "o-", color="#2c3e50", lw=1.8, label="研发强度中位数")
ax.set_xticks(x); ax.set_xticklabels([f"第{i}组" for i in x])
ax.set_xlabel("按 NTH 由低到高的五分位组（1 最偏长期，5 最偏短期）")
ax.set_ylabel("研发投入强度 %"); ax.legend(fontsize=9)
ax.set_title("图5-1　净时间视野分组与研发投入强度", fontsize=12)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{FIG}/fig5-1_conditional_means.png", dpi=160); plt.close()

# 图5 MD&A 抽取篇幅诊断
dg = pd.read_csv(os.path.join(RES, "extraction_diagnostics.csv"), dtype={"code": str})
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.hist(dg["tokens_clean"].clip(upper=15000), bins=40, color="#9b59b6", edgecolor="white")
ax.set_xlabel("MD&A 有效词数"); ax.set_ylabel("文档数")
ax.set_title("图3-0　MD&A 抽取结果的篇幅分布", fontsize=12)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{FIG}/fig3-0_mda_length.png", dpi=160); plt.close()

print("[OK] 插图已生成：", sorted(os.listdir(FIG)))
