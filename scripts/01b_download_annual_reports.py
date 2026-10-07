"""
第一步 ① 语料获取（续）：下载 2018-2025 年年度报告 PDF
================================================================
数据源：巨潮资讯网 www.cninfo.com.cn，接口 /new/hisAnnouncement/query

**关键踩坑记录**（写入论文"数据来源与处理说明"）：
  该接口 stock 参数形如 "代码,orgId"，而 orgId 是站方内部 ID，
  不同板块编号规则并不统一：000xxx 为 gssz+7位，600xxx 为 gssh0+6位，
  002/300/301/601/603/688 板块则为 9900xxxxxx 形式。
  按代码前缀"猜" orgId 会导致大面积空结果（实测仅 000/600 段命中）。
  可靠做法：**先用公司简称检索拿到 orgId，再用 代码+orgId 精确定位**。

流程：
  Step1  searchkey=简称 → 找到 secCode 与本样本一致的条目，取其 orgId
         （必须按 secCode 过滤：简称存在重名，如"出版传媒"对应多家公司）
  Step2  stock=代码,orgId + category=category_ndbg_szsh → 该公司全部年度报告
  Step3  标题过滤与版本选择：同一年度可能有多版（摘要/英文/更正/更新后）
输出：data/raw_pdf/{code}_{year}.pdf + data/sample_frame/download_manifest.csv
"""
import os, re, time, requests, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(BASE, "data", "sample_frame")
PDFDIR = os.path.join(BASE, "data", "raw_pdf")
os.makedirs(PDFDIR, exist_ok=True)

YEARS = list(range(2018, 2026))
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; low-frequency)"}
QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
TAG = re.compile(r"<[^>]+>")
TITLE_PAT = re.compile(r"(\d{4})\s*年年度报告\s*(全文|正文)?\s*$")

# 版本惩罚：含这些字样的版本优先度低；"摘要"与"英文版"直接淘汰
SOFT_BAD = ("更正", "修订", "补充", "更新后", "取消", "二次")
HARD_BAD = ("摘要", "英文", "已取消", "H股", "境外")


def post(session, data, retries=3):
    for a in range(retries):
        try:
            r = session.post(QUERY, data=data, headers=HEADERS, timeout=30)
            return r.json()
        except Exception:
            time.sleep(0.8 * (a + 1))
    return {}


def clean_name(name):
    return re.sub(r"[\s\*]", "", str(name)).replace("Ａ", "A")


def get_orgid(session, code, names):
    """Step1：按简称检索，取 secCode 匹配的 orgId。"""
    for nm in names:
        js = post(session, {"pageNum": "1", "pageSize": "30", "column": "szse",
                            "tabName": "fulltext", "searchkey": nm,
                            "category": "category_ndbg_szsh", "isHLtitle": "true"})
        for a in (js.get("announcements") or []):
            if a.get("secCode") == code:
                return a.get("orgId"), nm
        time.sleep(0.2)
    return None, None


def list_reports(session, code, orgid):
    """Step2：代码+orgId 精确定位，翻页取全部年度报告公告。"""
    out = []
    for pg in range(1, 6):
        js = post(session, {"pageNum": str(pg), "pageSize": "30", "column": "szse",
                            "tabName": "fulltext", "stock": f"{code},{orgid}",
                            "category": "category_ndbg_szsh", "isHLtitle": "true"})
        anns = js.get("announcements") or []
        if not anns:
            break
        for a in anns:
            if a.get("secCode") != code:
                continue
            out.append((TAG.sub("", a["announcementTitle"]).strip(),
                        "http://static.cninfo.com.cn/" + a["adjunctUrl"].lstrip("/"),
                        int(a["announcementTime"])))
        if len(anns) < 30:
            break
        time.sleep(0.2)
    return out


def pick_best(cands, yr):
    """同一年度多版本择一：硬淘汰 → 软惩罚 → 公告时间最早者。"""
    pool = []
    for title, url, ts in cands:
        m = TITLE_PAT.search(title)
        if not m or int(m.group(1)) != yr:
            continue
        if any(b in title for b in HARD_BAD):
            continue
        pool.append((sum(1 for b in SOFT_BAD if b in title), ts, url, title))
    if not pool:
        return None, None
    pool.sort()
    return pool[0][2], pool[0][3]


def download(session, url, path, min_bytes=80_000):
    if os.path.exists(path) and os.path.getsize(path) > min_bytes:
        return "cached"
    for a in range(3):
        try:
            with session.get(url, headers=HEADERS, timeout=180, stream=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            return "ok" if os.path.getsize(path) > min_bytes else "too_small"
        except Exception:
            time.sleep(1.5 * (a + 1))
    return "failed"


def main():
    frame = pd.read_csv(os.path.join(FRAME, "liaoning_raw.csv"), dtype={"code": str})
    rows, t0 = [], time.time()
    with requests.Session() as s:
        for i, r in frame.iterrows():
            code = r["code"]; raw = clean_name(r["name"])
            names = [raw]
            stripped = re.sub(r"^[\*]*ST", "", raw)
            if stripped != raw:
                names.append(stripped)
            orgid, qn = get_orgid(s, code, names)
            if not orgid:
                print(f"[{i+1}/{len(frame)}] {code} {raw:10s} !! 未取到 orgId", flush=True)
                continue
            cands = list_reports(s, code, orgid)
            got = []
            for yr in YEARS:
                url, title = pick_best(cands, yr)
                if not url:
                    continue
                path = os.path.join(PDFDIR, f"{code}_{yr}.pdf")
                status = download(s, url, path)
                got.append(yr)
                rows.append({"code": code, "name": r["name"], "year": yr,
                             "ok": status in ("ok", "cached"), "status": status,
                             "size_mb": round(os.path.getsize(path) / 1e6, 2)
                             if os.path.exists(path) else 0,
                             "title": title, "orgId": orgid, "url": url})
            print(f"[{i+1}/{len(frame)}] {code} {raw:10s} orgId={orgid:12s} "
                  f"命中{got}  ({time.time()-t0:.0f}s)", flush=True)
            time.sleep(0.15)
    mf = pd.DataFrame(rows)
    mf.to_csv(os.path.join(FRAME, "download_manifest.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[OK] 成功 {mf['ok'].sum()}/{len(mf)} 份，覆盖 {mf[mf.ok]['code'].nunique()} 家公司")
    print(mf.pivot_table(index="code", columns="year", values="ok",
                         aggfunc="first").to_string())


if __name__ == "__main__":
    main()
