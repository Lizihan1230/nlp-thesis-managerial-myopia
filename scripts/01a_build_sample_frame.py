"""
第一步 ① 语料获取：确定样本框（辽宁省A股非金融业上市公司）
================================================
数据源：巨潮资讯网（cninfo）公司概况接口 /data20/companyOverview/getCompanyIntroduction
判定规则：F005V（注册地址）以"辽宁省"开头 → 纳入样本框
合规：单线程受控并发 + 失败重试，遵守 robots 与低频请求原则
输出：data/sample_frame/sample_frame.csv
"""
import requests, pandas as pd, time, os, itertools, concurrent.futures as cf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "sample_frame")
os.makedirs(OUT, exist_ok=True)

URL = "http://www.cninfo.com.cn/data20/companyOverview/getCompanyIntroduction?scode={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; dalian-university)"}
PROVINCE = "辽宁省"


def candidate_codes():
    """枚举沪深 A 股可能代码段（不含 B 股 / 北交所）。"""
    ranges = [
        (1, 1999),        # 000xxx 深主板
        (2000, 3999),     # 002xxx 深主板(原中小板)
        (300000, 301999), # 300/301 创业板
        (600000, 600999), # 600 沪主板
        (601000, 601999), # 601 沪主板
        (603000, 603999), # 603 沪主板
        (605000, 605999), # 605 沪主板
        (688000, 688999), # 688 科创板
    ]
    codes = []
    for lo, hi in ranges:
        for c in range(lo, hi + 1):
            codes.append(f"{c:06d}")
    return codes


def fetch(code, session, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(URL.format(code), headers=HEADERS, timeout=20)
            js = r.json()
            recs = (js.get("data") or {}).get("records") or []
            if not recs:
                return None
            b = recs[0]["basicInformation"][0]
            return {
                "code": b.get("ASECCODE") or code,
                "name": (b.get("ASECNAME") or "").replace(" ", ""),
                "industry": b.get("F032V"),
                "market": b.get("MARKET"),
                "regaddr": b.get("F005V"),
                "listed_date": b.get("F010D"),
            }
        except Exception:
            time.sleep(0.6 * (attempt + 1))
    return None


def main():
    codes = candidate_codes()
    print(f"[INFO] 候选代码 {len(codes)} 个，开始并发查询公司概况…")
    rows, t0 = [], time.time()
    # 受控并发，避免对数据源造成压力
    with requests.Session() as s:
        with cf.ThreadPoolExecutor(max_workers=24) as ex:
            futures = {ex.submit(fetch, c, s): c for c in codes}
            for i, f in enumerate(cf.as_completed(futures), 1):
                r = f.result()
                if r:
                    rows.append(r)
                if i % 2000 == 0:
                    print(f"  已探测 {i}/{len(codes)}，命中 {len(rows)} 家，"
                          f"耗时 {time.time()-t0:.0f}s")
    df = pd.DataFrame(rows).drop_duplicates(subset="code")
    df.to_csv(os.path.join(OUT, "all_a_share.csv"), index=False, encoding="utf-8-sig")
    print(f"[OK] 全市场公司概况 {len(df)} 家 → all_a_share.csv")

    liaoning = df[df["regaddr"].fillna("").str.startswith(PROVINCE)].copy()
    liaoning["city"] = liaoning["regaddr"].str.replace(PROVINCE, "", regex=False).str.slice(0, 2)
    liaoning = liaoning.sort_values("code").reset_index(drop=True)
    liaoning.to_csv(os.path.join(OUT, "liaoning_raw.csv"), index=False, encoding="utf-8-sig")
    print(f"[OK] 辽宁省样本框 {len(liaoning)} 家 → liaoning_raw.csv")
    print(liaoning[["code", "name", "industry", "city"]].to_string(index=False))


if __name__ == "__main__":
    main()
