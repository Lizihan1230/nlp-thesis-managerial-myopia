"""
第一步 ① 语料获取（续）：财务与治理数据采集
============================================
数据源：新浪财经财务报表下载接口（vDOWN_ProfitStatement / BalanceSheet / CashFlow）
用途：为被解释变量（研发投入强度、资本支出）与控制变量提供数值型数据
输出：data/financials/financials_raw.csv  （企业-年份原始财务变量）
"""
import os, re, time, warnings, requests, pandas as pd, numpy as np

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(BASE, "data", "sample_frame")
OUTDIR = os.path.join(BASE, "data", "financials")
os.makedirs(OUTDIR, exist_ok=True)

YEARS = list(range(2018, 2026))
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; low-frequency)"}
MAP = {
    "income": "http://money.finance.sina.com.cn/corp/go.php/vDOWN_ProfitStatement/displaytype/4/stockid/{c}/ctrl/all.phtml",
    "balance": "http://money.finance.sina.com.cn/corp/go.php/vDOWN_BalanceSheet/displaytype/4/stockid/{c}/ctrl/all.phtml",
    "cashflow": "http://money.finance.sina.com.cn/corp/go.php/vDOWN_CashFlow/displaytype/4/stockid/{c}/ctrl/all.phtml",
}
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def fetch_table(kind: str, code: str, session) -> pd.DataFrame:
    """下载新浪财报(制表符分隔)并转为 行项目 × 报告期 的 DataFrame。"""
    for attempt in range(3):
        try:
            r = session.get(MAP[kind].format(c=code), headers=HEADERS, timeout=60)
            txt = r.content.decode("gb18030", "ignore")
            if "报表日期" not in txt:
                return pd.DataFrame()
            lines = [ln.split("\t") for ln in txt.strip().split("\n") if ln.strip()]
            head = [c.strip() for c in lines[0]]
            rows = {}
            for ln in lines[1:]:
                item = (ln[0] or "").strip()
                vals = [v.strip() for v in ln[1:]]
                if item and item not in rows:
                    rows[item] = vals
            df = pd.DataFrame.from_dict(rows, orient="index", columns=head[1:])
            return df
        except Exception:
            time.sleep(1.2 * (attempt + 1))
    return pd.DataFrame()


def pick(df: pd.DataFrame, item_keywords, period: str):
    """按关键词取首个匹配行项目在该报告期的数值。"""
    if df.empty:
        return np.nan
    for kw in item_keywords:
        idx = [i for i in df.index if kw == i or kw in i]
        if idx:
            v = df.loc[idx[0], period]
            return float(v) if NUM.match(str(v)) else np.nan
    return np.nan


def main():
    frame = pd.read_csv(os.path.join(FRAME, "liaoning_raw.csv"), dtype={"code": str})
    rows = []
    with requests.Session() as s:
        for i, r in frame.iterrows():
            code, name = r["code"], r["name"]
            inc = fetch_table("income", code, s)
            bal = fetch_table("balance", code, s)
            cfs = fetch_table("cashflow", code, s)
            for yr in YEARS:
                p = f"{yr}1231"
                if inc.empty or p not in inc.columns:
                    continue
                rec = {
                    "code": code, "name": name, "year": yr,
                    "revenue": pick(inc, ["营业总收入"], p) or pick(inc, ["营业收入"], p),
                    "rd_expense": pick(inc, ["研发费用"], p) or pick(inc, ["研究开发费用", "研究与开发费"], p),
                    "net_profit": pick(inc, ["归属于母公司所有者的净利润"], p) or pick(inc, ["净利润"], p),
                    "total_assets": pick(bal, ["资产总计", "资产合计"], p),
                    "total_liab": pick(bal, ["负债合计", "负债总计"], p),
                    "fixed_assets": pick(bal, ["固定资产"], p),
                    "cash_paid_capex": pick(cfs, ["购建固定资产、无形资产和其他长期资产所支付的现金",
                                              "购建固定资产、无形资产和其他长期资产支付的现金"], p),
                }
                # 至少要能算被解释变量才纳入
                if np.isfinite(rec.get("revenue")) and np.isfinite(rec.get("total_assets")):
                    rows.append(rec)
            print(f"[{i+1}/{len(frame)}] {code} {name} 完成", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "financials_raw.csv"), index=False, encoding="utf-8-sig")
    print(f"[OK] 财务数据：{len(df)} 条企业-年份观测 → financials_raw.csv")


if __name__ == "__main__":
    main()
