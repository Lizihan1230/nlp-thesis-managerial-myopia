"""
【内容效度辅助】抽取待人工判读的索引行样本
=========================================
从词典命中行中抽取随机样本，交由研究者按"该处语境是否确为短期视域表达"
逐条判定，据以计算**命中精确率 (hit precision)**，作为内容效度的量化支撑。
运行后请把判读结果写入 results/annotation_manual.csv（列：sn, label），
label ∈ {1=确为短视语义, 0=误命中}。
"""
import os, pandas as pd, numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
N = 150
RNG = np.random.default_rng(20260710)

cd = pd.read_csv(os.path.join(RES, "concordance_all.csv"), dtype={"code": str})
cd = cd[cd["left"].notna() & cd["right"].notna()]
# 分层：每个词条至多贡献若干条，避免热门词垄断样本
strata = cd.groupby("word", group_keys=False).apply(
    lambda g: g.sample(min(6, len(g)), random_state=7))
sample = strata.sample(min(N, len(strata)), random_state=7).reset_index(drop=True)
sample.insert(0, "sn", range(1, len(sample) + 1))
sample[["sn", "code", "year", "word", "left", "right"]].to_csv(
    os.path.join(RES, "annotation_sample.csv"), index=False, encoding="utf-8-sig")

print(f"[OK] 待判读样本 {len(sample)} 条 → results/annotation_sample.csv")
print("以下为待判读语料（…前为左语境，…后为右语境）：\n")
for _, r in sample.iterrows():
    print(f"[{r['sn']:>3}] ({r['code']} {r['year']})  …{r['left']}  【{r['word']}】  {r['right']}…")
