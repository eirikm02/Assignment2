# Bug-fix closing time and non-code content in OSSGameBench

Code for the MSR assignment (Option A, bug resolution). Dataset: OSSGameBench v3, DOI 10.5281/zenodo.18359123.

## Requirements
Python 3.10 or newer.

    pip install pandas pyarrow numpy scipy statsmodels matplotlib python-docx remotezip lifelines

## Steps
Run everything from the folder that contains analysis.py.

1. Get the small tables. Download issue, issue_pr, pr, pr_commit and issue_comment (parquet) into a folder
   called OSSGameBench. They can be taken out of the remote zip with remotezip. Save the file listing of the zip as
   listing.txt with `remotezip -l "<zip url with dl=1>"`.
2. Select projects:

       python analysis.py select --data OSSGameBench --min-bug-fixes 20 --max-projects 150 --seed 42 --listing listing.txt --max-commit-mb 100

   This writes out/selected_projects.csv and out/download_list.txt.
3. Download the commit files in the list:

       python download_commits.py "<zip url with dl=1>" out/download_list.txt

   On Windows the character | in file names is saved as _. The code accepts both.
4. Draw the validation sample and code it by hand:

       python analysis.py sample --data OSSGameBench --n 100

   Fill in the column is_real_bug_manual (1 or 0) in out/bug_label_sample.csv, then run

       python analysis.py validate --file out/bug_label_sample.csv
5. Run the analysis:

       python analysis.py analyze --data OSSGameBench

   Output: out/results.json, two figures and two csv tables.

## Test without the data
`python tests/make_synthetic.py synth` makes a small fake dataset with the same columns. Use `--data synth` in steps 2 to 5.
The numbers from that run are only a test of the code and are not results.

## Use of AI
The code in this repository was written with help from an AI assistant (Claude). I have run it on the real data, checked the output and take responsibility for it. The AI assistant also made the first coding of the 100 sampled issues, which I then checked (see the report).

## Notes
The outcome is days from created to closed. It is not debugging effort. Fixes without a linked commit file are left out of the file-based groups.
