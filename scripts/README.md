## Plot AMRs

This requires an extra `pip install matplotlib`. Then you can plot random or
target laigned AMRs (following sentence order)

```
python scripts/plot_amr.py --in-amr DATA/AMR2.0/aligned/cofill/train.txt
```

Use `--indices` to select AMRs by the order they appear in the file. See
`--has-*` flags to select by graph properties

## Sanity check AMR 

You can check any AMR against any propbank frames and their rules

Extract all frames in separate `xml` format file into one single `json` format
```
python scripts/read_propbank.py /path/to/amr_2.0/data/frames/xml/ DATA/probank_amr2.0.json
```

Run sanity check, for example
```
python scripts/sanity_check.py /path/to/amr2.0/train.txt DATA/probank_amr2.0.json

36522 sentences 152897 predicates
401 role not in propbank
322 predicate not in propbank
25 missing required role
```

## Additional Gold Path based metric

This computes scores based on how many full paths in the gold AMR appear
in the predicted AMR. It has specific flags fow KBQA AMR metrics

To just extract the paths do
```
python scripts/gold_path_metric.py --in-amr /path/to/file.amr --out-paths file.paths
```

To compute path precision
```
python scripts/gold_path_metric.py --in-amr /path/to/file.amr --gold-amr /path/to/gold.amr
```

To restrict to paths involving unknons and named entities (KBQA) user the `--kb-only` flag
