#!/usr/bin/env python3
"""
Bug-fix resolution time vs. involvement of non-code game content in OSSGameBench.

Sub-commands (run in this order):
  select    scan the SMALL tables, pick projects, write the list of commit files to download
  sample    write a random sample of bug-labelled issues for manual validation
  analyze   build the analysis table and run all statistics -> out/results.json, out/*.png, out/*.csv
  validate  read the manually coded sample and compute label precision

Example:
  python analysis.py select  --data OSSGameBench --min-bug-fixes 20 --max-projects 150 --seed 42
  python analysis.py sample  --data OSSGameBench --n 100 --seed 42
  python analysis.py analyze --data OSSGameBench
  python analysis.py validate --file out/bug_label_sample.csv
"""
import argparse, glob, json, os, re, sys
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

OUT = "out"
SEED = 42

# ----------------------------------------------------------------------------- helpers
def col(df, *names, required=True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"none of {names} in columns {list(df.columns)}")
    return None


def to_list(x):
    """labels / files can be list, ndarray, JSON string or plain string."""
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        out = []
        for i in x:
            out.append(i.get("name", str(i)) if isinstance(i, dict) else str(i))
        return out
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("["):
            try:
                return to_list(json.loads(s))
            except Exception:
                pass
        return [s] if s else []
    return [str(x)]


# Bug label rule. Documented in the report (Section 3.2).
BUG_RE = re.compile(r"(^|[^a-z])(bug|bugs|defect|crash|crashes|regression|glitch)([^a-z]|$)")
# labels that say the issue was NOT a real, fixed defect
NOT_FIXED_RE = re.compile(
    r"(duplicate|invalid|wontfix|won't fix|wont-fix|not a bug|notabug|not-a-bug|question|"
    r"cannot reproduce|can't reproduce|works as intended|by design|stale|needs info)")

CODE_EXT = {
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".inl", ".cs", ".java", ".kt", ".kts",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".lua", ".rs", ".go", ".gd", ".gdscript", ".swift", ".m",
    ".mm", ".rb", ".php", ".hx", ".as", ".glsl", ".hlsl", ".shader", ".cginc", ".frag", ".vert",
    ".wgsl", ".cmake", ".gradle", ".sh", ".bat", ".dart", ".scala", ".pas", ".vb", ".cu",
}
MEDIA_EXT = {  # art / audio / model / font / video files edited by non-programmers
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".tga", ".dds", ".psd", ".xcf", ".ico", ".webp",
    ".ogg", ".wav", ".mp3", ".flac", ".mid", ".midi", ".opus",
    ".fbx", ".obj", ".blend", ".glb", ".gltf", ".dae", ".3ds", ".ttf", ".otf", ".woff", ".woff2",
    ".mp4", ".webm", ".avi", ".ogv",
}
ENGINE_EXT = {  # engine-specific content: scenes, prefabs, materials, tilemaps ...
    ".unity", ".prefab", ".mat", ".asset", ".anim", ".controller", ".meta", ".tscn", ".tres", ".res",
    ".scn", ".uasset", ".umap", ".tmx", ".tsx", ".tileset", ".atlas", ".fnt", ".spine", ".material",
}


def file_kind(path):
    ext = PurePosixPath(str(path).lower()).suffix
    if ext in CODE_EXT:
        return "code"
    if ext in MEDIA_EXT:
        return "media"
    if ext in ENGINE_EXT:
        return "engine"
    return "other"


def load(data, name):
    p = os.path.join(data, name)
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    return pd.read_parquet(p)


def load_issues(data):
    iss = load(data, "issue.parquet")
    c = {
        "project": col(iss, "project", "repo"),
        "issue_number": col(iss, "issue_number", "number"),
        "title": col(iss, "title", required=False),
        "labels": col(iss, "labels"),
        "created_at": col(iss, "created_at"),
        "closed_at": col(iss, "closed_at", required=False),
        "contributor": col(iss, "contributor", "author", required=False),
        "body": col(iss, "body", required=False),
    }
    df = pd.DataFrame({k: iss[v] for k, v in c.items() if v})
    df["labels"] = df["labels"].map(to_list)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    if "closed_at" in df:
        df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    else:
        df["closed_at"] = pd.NaT
    low = df["labels"].map(lambda L: [l.lower() for l in L])
    df["is_bug"] = low.map(lambda L: any(BUG_RE.search(l) for l in L))
    df["not_fixed_label"] = low.map(lambda L: any(NOT_FIXED_RE.search(l) for l in L))
    return df


def merged_links(data):
    """issue -> merged PR table (many-to-many)."""
    ip = load(data, "issue_pr.parquet")
    ip = ip.rename(columns={col(ip, "project", "repo"): "project"})
    ip = ip.rename(columns={col(ip, "issue_number"): "issue_number", col(ip, "pr_number", "number"): "pr_number"})
    ip = ip[["project", "issue_number", "pr_number"]].drop_duplicates()
    pr = load(data, "pr.parquet")
    pc = {"project": col(pr, "project", "repo"), "pr_number": col(pr, "number", "pr_number"),
          "merged_at": col(pr, "merged_at", required=False), "state": col(pr, "state", required=False)}
    prd = pd.DataFrame({k: pr[v] for k, v in pc.items() if v})
    if "merged_at" in prd:
        prd["merged"] = prd["merged_at"].notna()
    else:
        prd["merged"] = prd["state"].astype(str).str.lower().eq("merged")
    return ip.merge(prd[["project", "pr_number", "merged"]], on=["project", "pr_number"], how="left").query("merged == True")


def read_commit_cols(path, cols):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=cols).to_pandas()

def commit_files_for(data, project):
    # read only sha+files: the diff column makes commit files huge.
    # Names are commits|owner|repo.parquet (or ...|1.parquet when split). Windows cannot store '|',
    # so extraction turns it into '_'. Match both spellings, case-insensitively.
    d = os.path.join(data, "commits")
    if not os.path.isdir(d):
        return None
    owner, repo = project.split("/", 1)
    names = os.listdir(d)
    parts = []
    for sep in ("|", "_"):
        base = f"commits{sep}{owner}{sep}{repo}".lower()
        rx = re.compile(re.escape(base) + r"(?:" + re.escape(sep) + r"\d+)?\.parquet$")
        parts += [os.path.join(d, fn) for fn in names if rx.fullmatch(fn.lower())]
    parts = sorted(set(parts))
    if not parts:
        return None
    out = pd.concat([read_commit_cols(p, ["sha", "files"]) for p in parts], ignore_index=True)
    # the sha field can carry git decorations, e.g. '70243850b10d... bleed' or '... (HEAD -> master)';
    # keep only the leading hex id so it matches pr_commit.commit_sha
    out["sha"] = out["sha"].astype(str).str.extract(r"^\s*([0-9a-fA-F]{7,64})")[0].str.lower()
    out = out.dropna(subset=["sha"]).drop_duplicates("sha")
    out["files"] = out["files"].map(to_list)
    return out

# ----------------------------------------------------------------------------- select
def parse_listing(path):
    """Parse `remotezip -l URL > listing.txt`. Returns {project_lower: (total_bytes, [zip paths])}."""
    res = {}
    rx = re.compile(r"^\s*(\d+)\s+\d{4}-\d\d-\d\d\s+[\d:]+\s+(OSSGameBench/commits/(commits\|([^|]+)\|([^|]+?)(?:\|\d+)?\.parquet))\s*$")
    raw = open(path, "rb").read()
    # PowerShell's ">" writes UTF-16; accept UTF-16 (with BOM or NUL bytes) and UTF-8
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or raw.count(b"\x00") > len(raw) // 4:
        text = raw.decode("utf-16", errors="ignore")
    else:
        text = raw.decode("utf-8-sig", errors="ignore")
    for line in text.splitlines():
        m = rx.match(line)
        if m:
            key = (m.group(4) + "/" + m.group(5)).lower()
            size, paths = res.get(key, (0, []))
            res[key] = (size + int(m.group(1)), paths + [m.group(2)])
    return res


def cmd_select(a):
    os.makedirs(OUT, exist_ok=True)
    iss = load_issues(a.data)
    links = merged_links(a.data)
    bug = iss[iss.is_bug & ~iss.not_fixed_label]
    cand = bug.merge(links, on=["project", "issue_number"])
    per = cand.groupby("project").issue_number.nunique().rename("bug_fix_issues").reset_index()
    elig = per[per.bug_fix_issues >= a.min_bug_fixes].copy()
    listing = parse_listing(a.listing) if a.listing else None
    if listing is not None:
        elig["commit_mb"] = elig.project.map(lambda p: listing.get(p.lower(), (np.nan, []))[0] / 1e6)
        if not listing:
            sys.exit("could not read any commit file names from the listing file; check listing.txt")
        n0 = len(elig)
        elig = elig[elig.commit_mb <= a.max_commit_mb]      # drops projects with no commit file, too
        print(f"size filter: {len(elig)} of {n0} eligible projects have commit files <= {a.max_commit_mb} MB")
    rng = np.random.default_rng(a.seed)
    if len(elig) > a.max_projects:
        elig = elig.iloc[rng.choice(len(elig), a.max_projects, replace=False)]
    elig = elig.sort_values("bug_fix_issues", ascending=False)
    elig.to_csv(f"{OUT}/selected_projects.csv", index=False)
    names = []
    for p in elig.project:
        if listing is not None:
            names += listing[p.lower()][1]
        else:
            names.append("OSSGameBench/commits/commits|" + p.replace("/", "|") + ".parquet")
    with open(f"{OUT}/download_list.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(names))
    if listing is not None:
        print(f"total download size of selected commit files: {elig.commit_mb.sum():.0f} MB")
    print(f"projects with any bug issue linked to a merged PR: {len(per)}")
    print(f"eligible (>= {a.min_bug_fixes}): {int((per.bug_fix_issues >= a.min_bug_fixes).sum())}; selected: {len(elig)}")
    print(f"bug-fix issues in selection: {int(elig.bug_fix_issues.sum())}")
    print(f"-> {OUT}/selected_projects.csv and {OUT}/download_list.txt (download ONLY these commit files)")

# ----------------------------------------------------------------------------- sample
def cmd_sample(a):
    os.makedirs(OUT, exist_ok=True)
    iss = load_issues(a.data)
    sel = pd.read_csv(f"{OUT}/selected_projects.csv").project
    bug = iss[iss.is_bug & iss.project.isin(sel)]
    s = bug.sample(min(a.n, len(bug)), random_state=a.seed)
    s = s.assign(labels=s.labels.map(lambda L: "; ".join(L)),
                 body=s.body.fillna("").str.slice(0, 400) if "body" in s else "")
    s["is_real_bug_manual"] = ""   # fill with 1 (real defect report) or 0 (not a defect)
    s[["project", "issue_number", "title", "labels", "body", "is_real_bug_manual"]].to_csv(
        f"{OUT}/bug_label_sample.csv", index=False)
    print(f"wrote {OUT}/bug_label_sample.csv ({len(s)} rows). Code column is_real_bug_manual with 1/0, then run validate.")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cmd_validate(a):
    s = pd.read_csv(a.file)
    s = s[s.is_real_bug_manual.isin([0, 1])]
    k, n = int(s.is_real_bug_manual.sum()), len(s)
    lo, hi = wilson(k, n)
    res = {"n_coded": n, "n_real_bug": k, "precision": k / n if n else None, "ci95": [lo, hi]}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/label_validation.json", "w"), indent=2)
    print(res)

# ----------------------------------------------------------------------------- build table
def build_table(data):
    iss = load_issues(data)
    links = merged_links(data)
    sel = pd.read_csv(f"{OUT}/selected_projects.csv").project.tolist()
    bug = iss[iss.is_bug & ~iss.not_fixed_label & iss.project.isin(sel)].copy()
    bug = bug.merge(links, on=["project", "issue_number"])   # one row per issue-PR pair
    prc = load(data, "pr_commit.parquet")
    prc = prc.rename(columns={col(prc, "project", "repo"): "project", col(prc, "pr_number"): "pr_number",
                              col(prc, "commit_sha", "sha"): "sha"})[["project", "pr_number", "sha"]]
    prc["sha"] = prc["sha"].astype(str).str.extract(r"^\s*([0-9a-fA-F]{7,64})")[0].str.lower()
    prc = prc.dropna(subset=["sha"])
    cm = load(data, "issue_comment.parquet")
    cm = cm.rename(columns={col(cm, "project", "repo"): "project", col(cm, "issue_number", "pr_number"): "issue_number",
                            col(cm, "contributor", "author"): "commenter"})
    rows = []
    for proj, g in bug.groupby("project"):
        files = commit_files_for(data, proj)
        if files is None:
            continue
        fmap = dict(zip(files.sha, files.files))
        pcs = prc[prc.project == proj]
        shas_by_pr = pcs.groupby("pr_number").sha.apply(list).to_dict()
        for (inum), gi in g.groupby("issue_number"):
            shas = {s for pn in gi.pr_number for s in shas_by_pr.get(pn, [])}
            flist = set()
            for s in shas:
                flist.update(fmap.get(s, []))
            if not flist:
                continue
            kinds = pd.Series([file_kind(f) for f in flist]).value_counts()
            r = gi.iloc[0]
            rows.append(dict(project=proj, issue_number=inum, created_at=r.created_at, closed_at=r.closed_at,
                             reporter=r.get("contributor"), title=r.get("title", ""), n_prs=gi.pr_number.nunique(),
                             n_commits=len(shas), n_files=len(flist),
                             n_code=int(kinds.get("code", 0)), n_media=int(kinds.get("media", 0)),
                             n_engine=int(kinds.get("engine", 0)), n_other=int(kinds.get("other", 0)),
                             labels="; ".join(r.labels)))
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no rows: check that the commit parquet files for the selected projects are in <data>/commits/")
    df["has_code"] = df.n_code > 0
    df["has_content"] = (df.n_media + df.n_engine) > 0
    df["group"] = np.select([df.has_code & ~df.has_content, df.has_code & df.has_content, ~df.has_code & df.has_content],
                            ["code_only", "code_and_content", "content_only"], default="other")
    snap = max(df.created_at.max(), df.closed_at.max())
    df["event"] = df.closed_at.notna().astype(int)
    end = df.closed_at.fillna(snap)
    df["days"] = (end - df.created_at).dt.total_seconds() / 86400
    df = df[df.days >= 0].copy()
    df["log_days"] = np.log1p(df.days)
    df["log_files"] = np.log1p(df.n_files)
    df["year"] = df.created_at.dt.year.astype(int)
    # collaboration (observed before closure only)
    cm2 = cm.merge(df[["project", "issue_number", "closed_at", "created_at", "reporter"]], on=["project", "issue_number"])
    cm2["t"] = pd.to_datetime(cm2["created_at_x"] if "created_at_x" in cm2 else cm2["created_at"], utc=True, errors="coerce")
    cm2 = cm2[cm2.closed_at.isna() | (cm2.t <= cm2.closed_at)]
    agg = cm2.groupby(["project", "issue_number"]).agg(n_comments=("commenter", "size"),
                                                       n_commenters=("commenter", "nunique")).reset_index()
    df = df.merge(agg, on=["project", "issue_number"], how="left").fillna({"n_comments": 0, "n_commenters": 0})
    df["log_comments"] = np.log1p(df.n_comments)
    df.attrs["snapshot"] = str(snap)
    return df, iss, links

# ----------------------------------------------------------------------------- stats
def cliffs_delta(x, y):
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(x, y, alternative="two-sided")
    return 2 * u.statistic / (len(x) * len(y)) - 1, u.pvalue


def bootstrap_ci_median_diff(x, y, rng, B=2000):
    xs = rng.choice(x, (B, len(x))); ys = rng.choice(y, (B, len(y)))
    d = np.median(xs, axis=1) - np.median(ys, axis=1)
    return np.percentile(d, [2.5, 97.5]).tolist()


def cmd_analyze(a):
    import statsmodels.formula.api as smf
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df, iss, links = build_table(a.data)
    df.drop(columns=["title"]).to_csv(f"{OUT}/analysis_table.csv", index=False)
    R = {"snapshot_date": df.attrs["snapshot"], "seed": SEED}
    R["n_projects"] = int(df.project.nunique())
    R["n_issues_before_group_filter"] = int(len(df))
    R["group_counts_all"] = df.group.value_counts().to_dict()
    d = df[df.group != "other"].copy()
    R["n_issues"] = int(len(d))
    R["n_closed"] = int(d.event.sum())
    R["n_censored"] = int((1 - d.event).sum())
    R["group_counts"] = d.group.value_counts().to_dict()
    R["n_projects_final"] = int(d.project.nunique())
    d["content_any"] = d.group.isin(["code_and_content", "content_only"]).astype(int)

    # recall proxy for the bug-label rule: fix-like titles among unlabeled issues linked to merged PRs
    unl = iss[~iss.is_bug & iss.project.isin(d.project.unique())].merge(links, on=["project", "issue_number"])
    if "title" in unl:
        fixlike = unl.title.fillna("").str.contains(r"\b(?:bug|crash|fix|broken|error|doesn.t work|not working)\b", case=False, regex=True)
        R["unlabelled_linked_issues"] = int(len(unl)); R["unlabelled_fixlike_share"] = float(fixlike.mean()) if len(unl) else None

    # descriptives (closed issues only; days)
    closed = d[d.event == 1]
    def desc(x):
        return dict(n=int(len(x)), median=float(np.median(x)), q1=float(np.percentile(x, 25)),
                    q3=float(np.percentile(x, 75)), mean=float(np.mean(x)), p90=float(np.percentile(x, 90)))
    R["desc_days"] = {g: desc(x.days) for g, x in closed.groupby("group")}
    R["desc_days"]["all"] = desc(closed.days)
    covs = d.groupby("group")[["n_files", "n_commits", "n_prs", "n_comments", "n_commenters"]].median().to_dict("index")
    R["median_covariates"] = covs

    # nonparametric tests (closed issues)
    groups = {g: x.days.values for g, x in closed.groupby("group") if len(x) >= 5}
    if len(groups) >= 2:
        R["kruskal"] = dict(zip(["H", "p"], map(float, stats.kruskal(*groups.values()))))
    x = closed[closed.content_any == 1].days.values; y = closed[closed.content_any == 0].days.values
    if len(x) >= 5 and len(y) >= 5:
        dl, p = cliffs_delta(x, y)
        R["content_vs_codeonly"] = dict(cliffs_delta=float(dl), mwu_p=float(p), n_content=len(x), n_code=len(y),
                                        median_content=float(np.median(x)), median_code=float(np.median(y)),
                                        median_diff_ci95=bootstrap_ci_median_diff(x, y, rng))

    # project-level paired comparison (avoids pooling across very different projects)
    pm = closed.assign(c=closed.content_any).groupby(["project", "c"]).days.agg(["median", "size"]).unstack("c")
    pm = pm.dropna()
    pm = pm[(pm[("size", 0)] >= 3) & (pm[("size", 1)] >= 3)]
    if len(pm) >= 8:
        diff = np.log((pm[("median", 1)] + 0.01) / (pm[("median", 0)] + 0.01))
        w = stats.wilcoxon(diff)
        R["paired_projects"] = dict(n_projects=int(len(pm)), median_log_ratio=float(diff.median()),
                                    share_slower=float((diff > 0).mean()), wilcoxon_p=float(w.pvalue))
        pm.assign(log_ratio=diff).to_csv(f"{OUT}/project_level_comparison.csv")
    else:
        diff = None

    # regression: log time-to-closure, project fixed effects, cluster-robust SE
    closed = closed.copy()
    ref = 'C(group, Treatment("code_only"))'
    forms = {
        "M0_group": f"log_days ~ {ref} + C(project)",
        "M1_plus_size": f"log_days ~ {ref} + log_files + n_prs + C(year) + C(project)",
        "M2_plus_collab": f"log_days ~ {ref} + log_files + n_prs + C(year) + log_comments + n_commenters + C(project)",
    }
    def usable(df_, cands):
        keep = []
        for c in cands:
            if df_[c].nunique() > 1 and df_.groupby("project")[c].transform(lambda v: v - v.mean()).abs().sum() > 0:
                keep.append(c)
        return keep
    sz = usable(closed, ["log_files", "n_prs"]); co = usable(closed, ["log_comments", "n_commenters"])
    yr = ["C(year)"] if closed.year.nunique() > 1 else []
    forms = {
        "M0_group": f"log_days ~ {ref} + C(project)",
        "M1_plus_size": " + ".join([f"log_days ~ {ref}"] + sz + yr + ["C(project)"]),
        "M2_plus_collab": " + ".join([f"log_days ~ {ref}"] + sz + yr + co + ["C(project)"]),
    }
    R["ols"] = {}
    for name, f in forms.items():
        try:
            m = smf.ols(f, data=closed).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(closed.project)[0]})
            out = {}
            for k in m.params.index:
                if k.startswith("C(group") or k in ("log_files", "n_prs", "log_comments", "n_commenters"):
                    lo, hi = m.conf_int().loc[k]
                    out[k.replace('C(group, Treatment("code_only"))[T.', "group:").rstrip("]")] = dict(
                        coef=float(m.params[k]), ratio=float(np.exp(m.params[k])), ci95_ratio=[float(np.exp(lo)), float(np.exp(hi))],
                        p=float(m.pvalues[k]))
            R["ols"][name] = dict(n=int(m.nobs), r2=float(m.rsquared), terms=out)
        except Exception as e:  # keep running
            R["ols"][name] = {"error": str(e)}

    # sensitivity: drop top 1% largest changes; drop issues closed within 1 hour
    for tag, sub in {"drop_top1pct_files": closed[closed.n_files <= closed.n_files.quantile(0.99)],
                     "drop_closed_lt_1h": closed[closed.days >= 1 / 24],
                     "project_has_both_groups": closed[closed.groupby("project").group.transform("nunique") > 1]}.items():
        try:
            m = smf.ols(forms["M1_plus_size"], data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.project)[0]})
            t = {k.replace('C(group, Treatment("code_only"))[T.', "group:").rstrip("]"):
                 dict(ratio=float(np.exp(v)), p=float(m.pvalues[k]))
                 for k, v in m.params.items() if k.startswith("C(group")}
            R.setdefault("sensitivity", {})[tag] = dict(n=int(m.nobs), terms=t)
        except Exception as e:
            R.setdefault("sensitivity", {})[tag] = {"error": str(e)}

    # Cox PH with censoring of still-open issues, stratified by project
    try:
        from statsmodels.duration.hazard_regression import PHReg
        X = pd.get_dummies(d.group, dtype=float)[["code_and_content", "content_only"]] if "content_only" in set(d.group) else \
            pd.get_dummies(d.group, dtype=float)[["code_and_content"]]
        X["log_files"] = d.log_files.values
        if d.n_prs.nunique() > 1: X["n_prs"] = d.n_prs.values
        ph = PHReg(d.days.clip(lower=1e-3).values, X.values, status=d.event.values,
                   strata=pd.factorize(d.project)[0]).fit(disp=0)
        R["cox"] = {n: dict(hazard_ratio=float(np.exp(b)), p=float(p)) for n, b, p in zip(X.columns, ph.params, ph.pvalues)}
    except Exception as e:
        R["cox"] = {"error": str(e)}

    # share of issues closed within 7 / 30 / 180 days
    R["closed_within"] = {str(k): d.assign(w=(d.days <= k) & (d.event == 1)).groupby("group").w.mean().to_dict()
                          for k in (7, 30, 180)}

    # figures
    order = [g for g in ["code_only", "code_and_content", "content_only"] if g in R["group_counts"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for g in order:
        v = np.sort(closed[closed.group == g].days.values)
        ax.step(v + 0.01, np.arange(1, len(v) + 1) / len(v), where="post", label=f"{g} (n={len(v)})")
    ax.set_xscale("log"); ax.set_xlabel("days from issue creation to closure (log scale)")
    ax.set_ylabel("share of closed issues"); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ecdf.png", dpi=200); plt.close(fig)
    if diff is not None:
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        ax.hist(diff, bins=20, color="#4c72b0", edgecolor="white"); ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("log(median days content-involved / median days code-only), per project"); ax.set_ylabel("projects")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_project_diff.png", dpi=200); plt.close(fig)
    json.dump(R, open(f"{OUT}/results.json", "w"), indent=2, default=str)
    print(json.dumps({k: R[k] for k in ("n_projects_final", "n_issues", "group_counts")}, indent=1))
    print(f"all results -> {OUT}/results.json")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select"); s.add_argument("--data", required=True)
    s.add_argument("--min-bug-fixes", type=int, default=20); s.add_argument("--max-projects", type=int, default=150)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--listing", default=None, help="file made with: remotezip -l URL > listing.txt")
    s.add_argument("--max-commit-mb", type=float, default=100, help="skip projects whose commit file is larger")
    s = sp.add_parser("sample"); s.add_argument("--data", required=True); s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=SEED)
    s = sp.add_parser("analyze"); s.add_argument("--data", required=True)
    s = sp.add_parser("validate"); s.add_argument("--file", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "sample": cmd_sample, "analyze": cmd_analyze, "validate": cmd_validate}[a.cmd](a)


if __name__ == "__main__":
    main()