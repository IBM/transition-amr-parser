# Set variables and environment for a give experiment
set -o errexit
set -o pipefail
set -o nounset

# AMR ORACLE
amr_train_file=/dccstor/multi-parse/amr/2016/jkaln_2016_scr.txt 
amr_dev_file=/dccstor/ykt-parse/AMR/2016data/dev.txt.removedWiki.noempty.JAMRaligned 
amr_test_file=/dccstor/ykt-parse/AMR/2016data/test.txt.removedWiki.noempty.JAMRaligned

# DATA
# 1 billion corpus
#extracted_oracle_folder=/dccstor/ramast1/_DATA/1B_Parsed/
#extracted_oracle_folder=/dccstor/ykt-parse/1B-Parsed/
#extracted_oracle_folder=/dccstor/ramast1/_DATA/
#extracted_oracle_folder=/dccstor/ykt-parse/ramast/stack-transformer/DATA/
extracted_oracle_folder=data/

# normal PTB
#prepro_tag=PTB
#data_set=PTB_SD_3_3_0
# PTB with BPE
#prepro_tag=PTB.BPE
#data_set=PTB_SD_3_3_0.BPE
# PTB with POS multi-task
#prepro_tag=PTB_POS
#data_set=PTB_SD_3_3_0+POS
# PTB with auto-encoding multi-task
#prepro_tag=PTB_Word100
#data_set=PTB_SD_3_3_0+Word100
# PTB with bigrams of actions
#prepro_tag=PTB_a2gram
#data_set=PTB_SD_3_3_0_action_2gram
# AMR
#prepro_tag=LDC2016_prepro_o2
#data_set=AMR_2016data_oracle2
# no prediction rules
#prepro_tag=LDC2016_prepro_o1
#data_set=AMR_2016data_oracle1
# multi-task 100 words
#prepro_tag=LDC2016_prepro_o2+Word100
#data_set=AMR_2016data_oracle2+Word100
# copy spacy lemmas/sense-01
prepro_tag=LDC2016_prepro_o3+Word100
data_set=AMR_2016data_oracle3+Word100
#realign
#prepro_tag=LDC2016_prepro_o4+Word100
#data_set=AMR_2016data_oracle4+Word100

# where the oracle data has bee extracted
# bash dcc/extract_AMR_data.sh
oracle_folder=$extracted_oracle_folder/${data_set}_extracted/

# use fp16=--fp16 for V100
#fp16=--fp16
fp16=""

# PREPROCESSING
# jbsub will solicit same number of cores. Use 10 for large corpora.
num_cores=1
# jbsub logs will be stored inside
features_folder=fairseq/data-bin/${data_set}_extracted/
fairseq_preprocess_args="
    --source-lang en
    --target-lang actions
    --trainpref ${oracle_folder}/train
    --validpref ${oracle_folder}/dev
    --testpref ${oracle_folder}/test
    --destdir $features_folder
    --workers $num_cores
    --machine-type AMR \
    --machine-rules $oracle_folder/train.rules.json \
    $fp16
"
# to restrict vocabulary
# --tgtdict $extracted_oracle_folder/${data_set}_extracted/external_dict.actions.txt

# model types defined in ./fairseq/models/transformer.py
#train_tag=t2x2
#base_model=transformer_2x2
#train_tag=stnp2x2
#base_model=stack_transformer_2x2_nopos

train_tag=stnp6x6
base_model=stack_transformer_6x6_nopos

#train_tag=st6x6
#base_model=stack_transformer_6x6

#train_tag=stnp6x6_tops
#base_model=stack_transformer_6x6_tops_nopos

#train_tag=stnp6x6_obuff
#base_model=stack_transformer_6x6_only_buffer_nopos

#train_tag=stnp6x6_ostack
#base_model=stack_transformer_6x6_only_stack_nopos

num_seeds=1
# use ="--lazy-load" for very large corpora (data does not fit into RAM)
do_lazy_load=""
# note that --save-dir is specified inside dcc/train.sh to account for the seed
# BERT backprop
# --bert-backprop
checkpoints_dir_root="fairseq/checkpoints/${base_model}-${prepro_tag}-${train_tag}"
fairseq_train_args="
    $features_folder
    --max-epoch 100
    --arch $base_model
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
    $do_lazy_load 
    $fp16
"

# TESTING
# this will be appended to $checkpoints_dir, which needs the seed given in the
# train.sh/test.sh scripts
beam_size=1
test_tag="${train_tag}_beam${beam_size}"
test_basename=checkpoint_best.pt
fairseq_generate_args="
    $features_folder 
    --gen-subset valid
    --machine-type AMR 
    --machine-rules $oracle_folder/train.rules.json \
    --beam ${beam_size}
    --batch-size 128
    --remove-bpe
    $fp16
"
#    --batch-size 128
