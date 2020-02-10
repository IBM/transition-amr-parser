. set_environment.sh

arguments="
    DATA/amr/features/o3+Word100_RoBERTa-base/ 
    --gen-subset train 
    --batch-size 128
"

python debug_data_iterator.py $arguments

#kernprof -l debug_data_iterator.py $arguments
#python -m line_profiler debug_data_iterator.py.lprof
