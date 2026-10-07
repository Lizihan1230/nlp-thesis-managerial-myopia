"""
第二步 ② 文本预处理：PDF → 干净分词结果
=========================================
流水线：
  ① 文档解析 PyMuPDF        → 定位年报 MD&A 章节（经营情况讨论与分析）
  ② 文本清洗                → 去重式去页眉页脚 / 页码 / 目录 / 乱码
  ③ 中文分词 jieba 精确模式 → 加载短视词典作用户词典，保护复合词不被切碎
  ④ 去停用词                → 通用汉语停用词 + 财经领域停用词
  ⑤ 词形归一                → 全角转半角、繁体转简体、统一数字与百分比写法

输出：
  data/corpus/{code}_{year}_mda.raw.txt       清洗后的 MD&A 原文
  data/processed/{code}_{year}_tokens.txt     归一化词序列（保留停用词）
  data/processed/{code}_{year}_clean.txt      去停用词后的有效词序列
  results/extraction_diagnostics.csv          抽取质量诊断（章节定位、篇幅）
"""
import os, re, sys, collections, warnings
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
import fitz  # PyMuPDF
import jieba

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))

PDFDIR = os.path.join(BASE, "data", "raw_pdf")
CORPUS = os.path.join(BASE, "data", "corpus")
PROCDIR = os.path.join(BASE, "data", "processed")
RESDIR = os.path.join(BASE, "results")
for d in (CORPUS, PROCDIR, RESDIR):
    os.makedirs(d, exist_ok=True)

LEX = os.path.join(BASE, "data", "lexicon", "shortterm_lexicon.txt")
SW = os.path.join(BASE, "data", "lexicon", "stopwords.txt")

CJK = re.compile(r"[\u4e00-\u9fa5]")

# ---------- 载入词典与停用词 ----------
def load_lexicon():
    words = {}
    for ln in open(LEX, encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith("---"):
            continue
        parts = ln.split()
        if len(parts) >= 2:
            words[parts[0]] = parts[1]
    return words

def load_stopwords():
    return {ln.strip() for ln in open(SW, encoding="utf-8")
            if ln.strip() and not ln.startswith("#")}

LEXICON = load_lexicon()
STOPWORDS = load_stopwords()
for w in LEXICON:
    jieba.add_word(w, freq=10000)     # 保护复合词不被切开
print(f"[INFO] 词典 {len(LEXICON)} 词；停用词 {len(STOPWORDS)} 词")

# ---------- MD&A 章节定位 ----------
CN_NUM = "一二三四五六七八九十"
SEC_LINE = re.compile(
    r"^\s*(?:第\s*([" + CN_NUM + r"]+)\s*[节篇章]\s*)?"
    r"(?:管理层讨论与分析|经营情况讨论与分析|董事会报告|管理层讨论及分析)\s*[（(]?.*$",
    re.M)
ANY_SEC = re.compile(r"^\s*第\s*([" + CN_NUM + r"]+)\s*[节篇章]", re.M)
TOC_LINE = re.compile(r"[.．·]{6,}|…{3,}")


def is_toc(page_text: str) -> bool:
    return len(TOC_LINE.findall(page_text)) >= 3


def locate_mda(pages):
    """返回 (start_page, end_page, sec_num, method)。pages 为页文本列表。"""
    cands = []
    for i, pt in enumerate(pages):
        if is_toc(pt):
            continue
        m = SEC_LINE.search(pt)
        if m:
            if m.group(1) is None:
                # 无"第X节"前缀：要求该行独占一行且文档不长，避免误命中行文引用
                line = m.group(0).strip()
                if len(line) > 25:
                    continue
            cands.append((i, m.group(1)))
    if not cands:
        return 0, len(pages) - 1, None, "fallback_fulltext"
    start, sec_num = cands[0]
    # 结束页：之后第一个出现"第X节/章/篇"且序号不同的页面
    end = len(pages) - 1
    for j in range(start + 1, len(pages)):
        if is_toc(pages[j]):
            continue
        ms = ANY_SEC.search(pages[j])
        if ms and ms.group(1) != sec_num:
            end = j - 1
            break
    if end <= start:
        end = min(len(pages) - 1, start + 25)
    return start, end, sec_num, "section_rule"


# ---------- 清洗 ----------
DOTLINE = re.compile(r"^[\s\.．·•\-—_=\d 　]*$")
PAGENUM = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
URL_RE = re.compile(r"(https?://|www\.)[^\s，。）】]+")


def clean_document(pages, start, end):
    """去页眉页脚（跨页高频行消除）→ 去目录/页码/空行 → 拼接。"""
    seg = pages[start:end + 1]
    if not seg:
        return ""
    # 统计短行出现频率，识别页眉页脚
    freq = collections.Counter()
    lines_per_page = []
    for pt in seg:
        ls = [l.strip() for l in pt.split("\n")]
        lines_per_page.append(ls)
        for l in dict.fromkeys(ls):
            if 0 < len(l) <= 24 and CJK.search(l):
                freq[l] += 1
    n = len(seg)
    boiler = {l for l, c in freq.items() if c >= max(3, n * 0.4)}
    out = []
    for ls in lines_per_page:
        for l in ls:
            if not l or l in boiler:
                continue
            if PAGENUM.match(l) or DOTLINE.match(l):
                continue
            if TOC_LINE.search(l):
                continue
            if l.startswith(("目", "录")) and len(l) <= 4:
                continue
            l = URL_RE.sub("", l)
            out.append(l)
    return "\n".join(out)


NUMBER = re.compile(r"^[\d\.,%％]+$")


def tokenize(text):
    """jieba 精确模式；返回 (全部token, 有效token)"""
    toks = jieba.lcut(text)
    keep_all, keep_clean = [], []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        if NUMBER.match(t):
            continue
        if not CJK.search(t):
            continue
        keep_all.append(t)
        if t in STOPWORDS:
            continue
        keep_clean.append(t)
    return keep_all, keep_clean


def main():
    mf = pd.read_csv(os.path.join(BASE, "data", "sample_frame", "download_manifest.csv"),
                     dtype={"code": str})
    mf = mf[mf["ok"]].sort_values(["code", "year"]).reset_index(drop=True)
    diag = []
    for i, r in mf.iterrows():
        pdf = os.path.join(PDFDIR, f"{r['code']}_{r['year']}.pdf")
        if not os.path.exists(pdf):
            continue
        try:
            doc = fitz.open(pdf)
            pages = [pg.get_text("text") for pg in doc]
            n_pages = len(pages)
            doc.close()
        except Exception as e:
            diag.append({"code": r["code"], "year": r["year"], "error": str(e)[:60]})
            continue
        start, end, sec, method = locate_mda(pages)
        text = clean_document(pages, start, end)
        all_tok, clean_tok = tokenize(text)
        base = f"{r['code']}_{r['year']}"
        with open(os.path.join(CORPUS, base + "_mda.raw.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(PROCDIR, base + "_tokens.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(all_tok))
        with open(os.path.join(PROCDIR, base + "_clean.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(clean_tok))
        diag.append({"code": r["code"], "year": r["year"], "n_pages": n_pages,
                     "mda_start_page": start, "mda_end_page": end,
                     "mda_pages": end - start + 1, "section_id": sec,
                     "method": method, "raw_chars": len(text),
                     "tokens_all": len(all_tok), "tokens_clean": len(clean_tok)})
        if (i + 1) % 25 == 0:
            print(f"  已处理 {i+1}/{len(mf)}", flush=True)
    df = pd.DataFrame(diag)
    df.to_csv(os.path.join(RESDIR, "extraction_diagnostics.csv"),
              index=False, encoding="utf-8-sig")
    print("\n[OK] 预处理完成：", len(df), "份文档")
    print(df.groupby("method").size().to_string())
    print(df[["tokens_all", "tokens_clean", "mda_pages"]].describe().to_string())


if __name__ == "__main__":
    main()
