## Plot AMRs

To plot in LaTex using tikz, you can use

```
python scripts/plot_amr.py --in-amr DATA/wiki25.jkaln --out-amr tmp.tex
```

Use `--indices` to select AMRs by the order they appear in the file. See
`--has-*` flags to select by graph properties

To plot using matplotlib (for e.g. notebooks) you can use `AMR.plot()` in the
AMR class

## JAMR to ISI notaion

To convert an AMR file aligned using JAMR (_Kevin) aligner into isi alignments format.

```
python scripts/jamr2isis.py --in-amr <jamr aligned amr> --out-amr <output in isi notaion>
```

## Understanding the Oracle

An oracle is a module that given a sentence and its AMR annotation (aligned,
right now) provides a sequence of actions, that played on a state machine
produce back the AMR. Current AMR oracle aka Oracle10 can be explored in
isolation running

```
bash tests/oracles/amr_o10.sh DATA/wiki25.jkaln
```

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
