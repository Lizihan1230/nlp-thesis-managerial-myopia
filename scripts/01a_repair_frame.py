"""
第一步 ① 语料获取（补）：样本框修复扫描
=========================================
首轮高并发扫描存在超时丢帧，本脚本对"未命中代码"反复重扫直到收敛，
保证样本框的**穷尽性**（这是后续样本选择偏差讨论的前提）。
"""
import requests, pandas as pd, time, os, concurrent.futures as cf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "sample_frame")
URL = "http://www.cninfo.com.cn/data20/companyOverview/getCompanyIntroduction?scode={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; dalian-university)"}
PROVINCE = "辽宁省"


def candidate_codes():
    ranges = [(1, 1999), (2000, 3999), (300000, 301999), (600000, 600999),
              (601000, 601999), (603000, 603999), (605000, 605999), (688000, 688999)]
    return [f"{c:06d}" for lo, hi in ranges for c in range(lo, hi + 1)]


def fetch(code, session, retries=4):
    for a in range(retries):
        try:
            r = session.get(URL.format(code), headers=HEADERS, timeout=30)
            recs = (r.json().get("data") or {}).get("records") or []
            if not recs:
                return None
            b = recs[0]["basicInformation"][0]
            return {"code": b.get("ASECCODE") or code,
                    "name": (b.get("ASECNAME") or "").replace(" ", ""),
                    "industry": b.get("F032V"), "market": b.get("MARKET"),
                    "regaddr": b.get("F005V"), "listed_date": b.get("F010D")}
        except Exception:
            time.sleep(0.8 * (a + 1))
    return None


def pass_scan(codes, workers=12):
    rows, fails, t0 = [], [], time.time()
    with requests.Session() as s:
        s.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=workers + 8))
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch, c, s): c for c in codes}
            for i, f in enumerate(cf.as_completed(futs), 1):
                r = f.result()
                (rows if r else fails).append(r if r else futs[f])
                if i % 1500 == 0:
                    print(f"    {i}/{len(codes)} 命中+{len(rows)} 耗时{time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows), fails


def main():
    known_path = os.path.join(OUT, "all_a_share.csv")
    known = pd.read_csv(known_path, dtype={"code": str})
    known_codes = set(known["code"])
    n0 = len(known)
    print(f"[INFO] 当前已收录 {n0} 家，开始修复扫描…")

    for rnd in range(1, 5):
        todo = [c for c in candidate_codes() if c not in known_codes]
        print(f"  第 {rnd} 轮：待重扫 {len(todo)} 个代码", flush=True)
        if not todo:
            break
        new, _ = pass_scan(todo)
        if not new.empty:
            known = pd.concat([known, new], ignore_index=True).drop_duplicates("code")
            known_codes = set(known["code"])
        known.to_csv(known_path, index=False, encoding="utf-8-sig")
        print(f"  第 {rnd} 轮后累计 {len(known)} 家（新增 {len(known)-n0}）", flush=True)
        if len(new) < 30:
            print("  [收敛] 新增极少，停止。")
            break

    ln = known[known["regaddr"].fillna("").str.startswith(PROVINCE)].copy()
    ln["city"] = ln["regaddr"].str.replace(PROVINCE, "", regex=False).str.slice(0, 2)
    ln = ln.sort_values("code").reset_index(drop=True)
    ln.to_csv(os.path.join(OUT, "liaoning_raw.csv"), index=False, encoding="utf-8-sig")
    print(f"[OK] 辽宁省样本框最终 {len(ln)} 家 → liaoning_raw.csv")
    print(ln[["code", "name", "city"]].to_string(index=False))


if __name__ == "__main__":
    main()
