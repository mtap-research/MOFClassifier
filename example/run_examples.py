from MOFClassifier.CLscore import predict
import glob
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

input_pattern = "/path/to/folder/*.cif"
files = glob.glob(input_pattern)

scores_csv = "all_CLscores.csv"
score_csv  = "all_CLscore.csv"
fail_txt   = "fail_list.txt"

bag_cols = [f"bag_{i+1}" for i in range(100)]
with open(scores_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["cif_name"] + bag_cols)

with open(score_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["cif_name", "CLscore"])

open(fail_txt, "w").close()

scores_f = open(scores_csv, "a", newline="")
score_f  = open(score_csv,  "a", newline="")
fail_f   = open(fail_txt,   "a")

scores_writer = csv.writer(scores_f)
score_writer  = csv.writer(score_f)

with ProcessPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(predict, fn): fn for fn in files}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting"):
        fn = futures[future]
        try:
            cifid, CLscores, CLscore = future.result()
            scores_writer.writerow([cifid] + CLscores)
            scores_f.flush()
            score_writer.writerow([cifid, CLscore])
            score_f.flush()
        except Exception:
            fail_f.write(fn + "\n")
            fail_f.flush()

scores_f.close()
score_f.close()
fail_f.close()
