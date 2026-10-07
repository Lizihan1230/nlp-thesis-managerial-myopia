"""补全现金流量表"购建固定资产…所支付的现金"字段"""
import os, re, requests, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(BASE, "data", "financials", "financials_raw.csv")
NUM = re.compile(r"^-?\d+(\.\d+)?$")
URL = "http://money.finance.sina.com.cn/corp/go.php/vDOWN_CashFlow/displaytype/4/stockid/{c}/ctrl/all.phtml"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research)"}
df = pd.read_csv(F, dtype={"code": str})
KEYS = ["购建固定资产、无形资产和其他长期资产所支付的现金",
        "购建固定资产、无形资产和其他长期资产支付的现金"]
ok = 0
with requests.Session() as sess:
    for code, g in df.groupby("code"):
        txt = sess.get(URL.format(c=code), headers=HEADERS, timeout=60).content.decode("gb18030", "ignore")
        if "报表日期" not in txt:
            continue
        lines = [ln.split("\t") for ln in txt.strip().split("\n") if ln.strip()]
        head = [c.strip() for c in lines[0]]
        rows = {}
        for ln in lines[1:]:
            it = (ln[0] or "").strip()
            if it and it not in rows:
                rows[it] = [v.strip() for v in ln[1:]]
        if not rows:
            continue
        tab = pd.DataFrame.from_dict(rows, orient="index", columns=head[1:])
        for _, r in g.iterrows():
            p = f"{int(r['year'])}1231"
            if p not in tab.columns:
                continue
            v = np.nan
            for kw in KEYS:
                if kw in tab.index:
                    raw = str(tab.loc[kw, p])
                    v = float(raw) if NUM.match(raw) else np.nan
                    break
            if np.isfinite(v):
                df.loc[(df.code == code) & (df.year == r["year"]), "cash_paid_capex"] = v
                ok += 1
print(f"[OK] 补全资本支出字段 {ok} 条")
print("可得率:", df['cash_paid_capex'].notna().mean().round(4))
df.to_csv(F, index=False, encoding="utf-8-sig")
