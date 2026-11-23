"""Builds a SYNTHETIC miniature dataset with the OSSGameBench schema to test the pipeline.
It contains a planted effect (content fixes ~1.5x slower). Its numbers are NOT real results."""
import json, os, sys
import numpy as np, pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "synth"
rng = np.random.default_rng(7)
os.makedirs(f"{out}/commits", exist_ok=True)
CODE = [".cpp", ".cs", ".h", ".py"]; MEDIA = [".png", ".ogg", ".svg"]; OTH = [".md", ".txt"]
issues, ip, prs, prc, cms = [], [], [], [], []
for p in range(40):
    proj = f"own{p}/game{p}"; base = rng.lognormal(1.5, 0.5)
    commits, t0 = [], pd.Timestamp("2021-01-01", tz="UTC")
    for i in range(1, 80):
        created = t0 + pd.Timedelta(days=float(rng.uniform(0, 1200)))
        bug = rng.random() < 0.7
        labels = ["bug"] if bug else ["enhancement"]
        if rng.random() < 0.05: labels.append("duplicate")
        grp = rng.choice(["code", "mixed", "content"], p=[.7, .2, .1])
        files = list(rng.choice(CODE, rng.integers(1, 4))) if grp != "content" else []
        if grp != "code": files += list(rng.choice(MEDIA, rng.integers(1, 3)))
        files = [f"src/f{j}{e}" for j, e in enumerate(files)] + ["README.md"] * int(rng.random() < .2)
        mult = 1.0 if grp == "code" else 1.5
        days = base * mult * rng.lognormal(0, 1.2) * (1 + .15 * len(files))
        closed = created + pd.Timedelta(days=float(days)) if rng.random() > .05 else pd.NaT
        issues.append(dict(project=proj, number=i, state="closed", contributor=f"u{rng.integers(0,30)}", title=f"issue {i}",
                           body="b", labels=json.dumps(labels), created_at=created, closed_at=closed, linked_prs=[i]))
        prs.append(dict(project=proj, number=i, contributor="x", state="closed", merged_at=created + pd.Timedelta(days=1)))
        ip.append(dict(project=proj, issue_number=i, pr_number=i))
        sha = f"{p:02d}{i:04d}"
        prc.append(dict(project=proj, pr_number=i, commit_sha=sha))
        commits.append(dict(sha=sha, repo=proj, files=json.dumps(files), diff="x" * 50))
        for k in range(rng.poisson(3)):
            cms.append(dict(project=proj, issue_number=i, contributor=f"u{rng.integers(0,30)}",
                            body="c", created_at=created + pd.Timedelta(hours=float(rng.uniform(0, 48)))))
    pd.DataFrame(commits).to_parquet(f"{out}/commits/commits|{proj.replace('/', '|')}.parquet")
pd.DataFrame(issues).to_parquet(f"{out}/issue.parquet")
pd.DataFrame(prs).to_parquet(f"{out}/pr.parquet")
pd.DataFrame(ip).to_parquet(f"{out}/issue_pr.parquet")
pd.DataFrame(prc).to_parquet(f"{out}/pr_commit.parquet")
pd.DataFrame(cms).to_parquet(f"{out}/issue_comment.parquet")
print("synthetic data in", out)
