## Training a full model

This code is intended to train models from scratch on the CCC cluster but can
be repurposed for other task managers e.g. slurm. You can do a mini run to
check how this all works under

    bash tests/minimal_test_lsf.sh

First of all make sure you have installed according to README.md. Be sure to
activate your environment in `set_environment.sh` since this is called by the
different scripts

Then ensure you have unzipped the data from its location, you will need at least

1. the corpus you want to train for e.g. AMr2.0 (optionally already aligned)

2. the entity linking cache for that corpus

once you have unzipped these items we are ready to go. The code is though to be
latched from a **login node** not a compute node. You will need some app to
have a pervasive session on that login node (this is a good idea in general)
like tmux (recommended) or screen. From one of those do e.g.

    bash run/lsf/run_experiment.sh configs/amr2.0-structured-bart-large-neur-al.sh

this will launch all the needed jobs in a dependent fashion so that one is run
after another (seeds will be ran in parallel). It will also display the status
of the training. The script will hold until the first checkpoint is created to
launch the evaluation jobs. This is why this command line call needs to be kept
alive, after that it is no longer necessary.

At any point you can do

    bash run/status.sh -c configs/amr2.0-structured-bart-large-neur-al.sh

to check the status of that experiment. Once results start appearing, you can use

    bash run/status.sh --configs configs/amr2.0-structured-bart-large-neur-al.sh --results

to check progress. To compare models and get details of loss and Smatch, you
can plot a png and bring it locally with scp with

    python scripts/plot_results.py --in-configs configs/amr2.0-structured-bart-large-neur-al.sh --title my-training --out-png my-training.png

each step of the experiment has its own folder and it is completed it should
have a `.done` file. If you delete this the stage will be redone (not the
neural aligner has multiple of these files). The final model should be found under e.g.

    DATA/AMR2.0/models/amr2.0-structured-bart-large-neur-al/

We try to avoid running on the tests set to prevent corpus overfitting, this
can be done with

    bash run/lsf/final_test.sh configs/amr2.0-structured-bart-large-neur-al.sh

It will ask you to confirm.    

Once training is done you can save space by calling

    bash run/status.sh -c configs/amr2.0-structured-bart-large-neur-al.sh --final-remove

This will remove the optimizer from configs `DECODING_CHECKPOINT` and delete
all other. Save copies if you want further train later.

to save the minimal files needed for a model into a zip do

    bash scripts/export_model.sh configs/amr2.0-structured-bart-large-neur-al.sh

## Things that can go wrong

Code is built to be able to resume if it stops, just do 

    bash run/lsf/run_experiment.sh configs/amr2.0-structured-bart-large-neur-al.sh

But it should not die, so if it did it is important to find the reason first
before resuming.

The most common problem is that you hit your space quota and code dies halfway
while writing a checkpoint. You need to know how to check your quota to avoid
this. Also the jobs doing evaluation also take care of removing checkpoints. If
these die then your space can finish quickly. This should not happen and it is
best to fins the reason why this happened before relaunching evaluation. You
can do this with

    bash run/lsf/run_model_eval.sh configs/amr2.0-structured-bart-large-neur-al.sh

If you hit your quota, you need to fix that first, then you will also have to
find and delete corrupted checkpoints. For this you can use

    bash run/status.sh -c configs/amr2.0-structured-bart-large-neur-al.sh --remove-corrupted-checkpoints

the code automatically calls

    bash run/status.sh -c configs/amr2.0-structured-bart-large-neur-al.sh --link-best --remove

to find the best checkpoint and remove checkpoints not in the top n-best, but
it may come handy to run this yourself at some point. It is already a bad
state of affairs if some checkpoint got deleted without being evaluated, but
you can always ignore this by adding `--ignore-missing-checkpoints`
