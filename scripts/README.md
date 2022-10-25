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

To convert an AMR file aligned using JAMR (+Kevin) aligner into ISI alignments format.

```
python scripts/jamr2isi.py --in-amr <jamr_aligned_amr> --out-amr <output_in_isi_notaion>
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
python scripts/sanity_check_amr.py /path/to/amr2.0/train.txt DATA/probank_amr2.0.json

36522 sentences 152897 predicates
401 role not in propbank
322 predicate not in propbank
25 missing required role
```

## Paired Boostrap Significance Test

The following script implements the paired boostrap significance test after

    @Book{Nor89,
        author = {E. W. Noreen},
        title =  {Computer-Intensive Methods for Testing Hypotheses},
        publisher = {John Wiley Sons},
        year = {1989},
    }

to use you can call

```bash
python scripts/smatch_aligner.py \
    --in-reference-amr /path/to/gold.amr \
    --in-amrs \
        /path/to/predicted1.amr \ 
        /path/to/predicted2.amr \ 
        ...
        /path/to/predictedN.amr \ 
    --amr-labels 
        label1 \
        label2 \
        ...
        labelN \
    --bootstrap-test
```

for each pair of predicted amr files, it tests the hypothesis that the
prediction with larges Smatch is significantly greater than the smaller one.
Use `--bootstrap-test-restarts` to set the number of samples (default `10,000`,
note this has little effect on speed). Use `--out-boostrap-png
/path/to/file.png` to save the distribution of score differences for each pair.
Script calls the original `smatch` python module. In order to export components
it needs the main branch after version `1.0.4`.
