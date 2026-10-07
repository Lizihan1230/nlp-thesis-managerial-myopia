# GitHub 发布说明

本仓库已在本机完成 `git init` 与首次提交，**只差最后一步推送**（需要你的 GitHub 凭据）。
以下三种方式任选其一。

---

## 方式一：用 GitHub CLI 登录后推送（推荐）

```bash
# 1. 登录（按提示浏览器授权，或粘贴 Personal Access Token）
gh auth login

# 2. 在 GitHub 上创建仓库并推送（仓库名可改）
cd nlp-thesis
gh repo create nlp-thesis --public --source=. --remote=origin --push
```

如果仓库已存在，改用：

```bash
git remote add origin https://github.com/<你的用户名>/nlp-thesis.git
git branch -M main
git push -u origin main
```

## 方式二：使用 Personal Access Token 直接推送

在 GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
创建一个具备 `Contents: Read and write` 与 `Administration: Read and write` 权限的令牌，然后：

```bash
cd nlp-thesis
git remote add origin https://<用户名>:<TOKEN>@github.com/<用户名>/nlp-thesis.git
git branch -M main
git push -u origin main
```

> 安全提示：不要把令牌写进任何提交文件。推送完成后可用
> `git remote set-url origin https://github.com/<用户名>/nlp-thesis.git` 清除令牌。

## 方式三：离线迁移（无网络凭据时）

本目录已附带 `nlp-thesis.bundle`，它是一个自包含的 git 仓库快照，可在任意机器上还原并推送：

```bash
git clone nlp-thesis.bundle nlp-thesis
cd nlp-thesis
git remote set-url origin https://github.com/<用户名>/nlp-thesis.git
git push -u origin main
```

或者直接解压 `nlp-thesis.zip`，重新 `git init` 后推送。

---

## 推送内容清单

| 类别 | 内容 | 是否入库 |
|---|---|---|
| 论文正文 | `docs/毕业论文预演稿.md`（中文 10 595 字） | 是 |
| Word 版 | `docs/毕业论文预演稿.docx`（14 表 5 图） | 是 |
| 数据说明 | `docs/数据来源与处理说明.md` | 是 |
| 五步流水线代码 | `scripts/01a` … `08_md_to_docx.py`（15 个脚本） | 是 |
| 词典与停用词 | `data/lexicon/`（5 份，含编制依据） | 是 |
| 样本框与财务数据 | `data/sample_frame/`、`data/financials/` | 是 |
| 全部结果表 | `results/`（24 个 CSV） | 是 |
| 论文插图 | `figures/`（5 张 PNG） | 是 |
| 原始年报 PDF | `data/raw_pdf/`（1.7 GB） | **否**，由 `01b` 脚本重新下载 |
| 中间语料 | `data/corpus/`、`data/processed/` | **否**，由 `02` 脚本重新生成 |

仓库体积约 1.6 MB（.git），适合直接公开。

---

## 建议的仓库设置

- **仓库名**：`nlp-thesis-managerial-myopia`（或 `nlp-measurement-thesis`）
- **可见性**：Public（便于导师与评审查看）或 Private（若担心正文被他人直接引用）
- **Topics**：`nlp` `text-analysis` `jieba` `word2vec` `lda` `corporate-finance` `thesis` `chinese-nlp`
- **Description**：`管理者短视的NLP测度构建与信效度检验｜辽宁省A股上市公司年报MD&A（2018-2025）`

## 发布前自查清单

- [ ] 确认论文正文中不包含个人身份信息（学号、身份证号、指导教师姓名与电话）
- [ ] 确认未把任何 API Token 写入脚本或配置文件
- [ ] 确认 `docs/数据来源与处理说明.md` 中的合规声明与实际抓取行为一致
- [ ] 若选择 Private，需为导师与评审单独授予访问权限
- [ ] 论文正文仅供教学参考，请勿直接作为正式学位论文提交
