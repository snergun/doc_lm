python cal_ppl.py --data data/penn --save /home/jovyan/doc_lm/trainedmodel/ptb/additional_finetuned.pt --bptt 1000

for seed in 0 1 2 3; do
    echo "Running seed $seed"
    python cal_ppl.py --data data/penn --save /home/jovyan/doc_lm/trainedmodel/ptb_ensemble/seed"$seed"_additional_finetuned.pt --bptt 1000
done