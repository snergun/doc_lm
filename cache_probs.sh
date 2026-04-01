python cal_ppl.py --data data/penn --save /home/jovyan/doc_lm/trainedmodel/ptb/additional_finetuned.pt --bptt 1000

for seed in 1 2 3 4 5; do
    echo "Running seed $seed
    python