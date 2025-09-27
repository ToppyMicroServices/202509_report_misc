This folder may contain large, transient datasets (e.g., HMDA_2018_originated_nationwide.csv) that should not be committed to Git.

How to use:
- Download data into this folder as needed.
- Do NOT commit gigabyte-scale CSVs. These are ignored by .gitignore.
- If you accidentally staged a large file, unstage it:
  git reset HEAD -- downloads/HMDA_*.csv
- If it was committed and pushed failed due to size, you must remove it from history (see below).

History cleanup options:
1) Remove from the last commit only (if not yet pushed):
   git rm --cached downloads/HMDA_*.csv
   git commit -m "remove large HMDA file from repo"
2) Rewrite history (if committed before) using git filter-repo (recommended):
   pipx install git-filter-repo  # or brew install git-filter-repo
   git filter-repo --force --path downloads/HMDA_2018_originated_nationwide.csv --invert-paths
   git push --force-with-lease

Alternative: Git LFS
- If you prefer to track large files, install Git LFS and track the pattern:
  git lfs install
  git lfs track "downloads/HMDA_*.csv"
  git add .gitattributes
  git commit -m "track HMDA CSV via LFS"
