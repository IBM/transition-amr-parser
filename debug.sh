# fairseq/fairseq_cli/train.py(125)train()
# fairseq/data/transition_based_parsing_dataset.py
# fairseq/tasks/fairseq_task.py

. set_environment.sh

rm -Rf TMP
mkdir -p TMP

arguments="
    DATA/amr/features/o3+Word100_RoBERTa-base/ 
    --max-epoch 1 
    --arch stack_transformer_6x6_nopos 
    --optimizer adam 
    --adam-betas '(0.9,0.98)' 
    --clip-norm 0.0 
    --lr-scheduler inverse_sqrt 
    --warmup-init-lr 1e-07 
    --warmup-updates 4000 
    --lr 0.0005 
    --min-lr 1e-09 
    --dropout 0.3 
    --weight-decay 0.0 
    --criterion label_smoothed_cross_entropy 
    --label-smoothing 0.01 
    --keep-last-epochs 40 
    --max-tokens 3584 
    --log-format json 
    --seed 42 
    --save-dir TMP
"

# normall run
# to list profile decorators
# grep -rl '@profile' fairseq/fairseq
# to switich off profile decorators
# grep -rl '@profile' fairseq/fairseq | xargs sed 's,@profile,#@profile,' -i
fairseq-train $arguments

# profiled
# to switch on profile decorators
# grep -rl '@profile' fairseq/fairseq | xargs sed 's,#@profile,@profile,' -i
#kernprof -l fairseq/train.py $arguments
#python -m line_profiler train.py.lprof
