python cal_ppl.py --data data/penn --save /home/jovyan/doc_lm/trainedmodel/ptb/additional_finetuned --bptt 1000
for seed in 0 1 2 3; do
    echo "Running seed $seed"
    python cal_ppl.py --data data/penn --save /home/jovyan/doc_lm/trainedmodel/ptb_ensemble/seed"$seed"_additional_finetuned --bptt 1000
done

python cal_ppl.py --data data/wikitext-2 --save /home/jovyan/doc_lm/trainedmodel/wikitext2/additional_finetuned --bptt 1000
for seed in 0 1 2 3; do
    echo "Running seed $seed"
    python cal_ppl.py --data data/wikitext-2 --save /home/jovyan/doc_lm/trainedmodel/wikitext2_ensemble/seed"$seed"_additional_finetuned --bptt 1000
done