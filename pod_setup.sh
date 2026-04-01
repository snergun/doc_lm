echo "--- Creating Conda Environment 'doc_lm' ---"
conda create -n doc_lm python=3.8 -y
conda activate doc_lm

# 3. Install PyTorch and Dependencies
echo "--- Installing PyTorch and ML Stack ---"
# Using 1.12.1 for TITAN RTX / 3090 compatibility
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

# 4. Create the MKL/iJIT Stub Fix
echo "--- Applying Intel MKL Symbol Fix ---"
cat > "$CONDA_PREFIX/lib/itt_stub.c" <<'EOF'
void iJIT_NotifyEvent() {}
void iJIT_NotifyEventW() {}
int iJIT_IsProfilingActive() { return 0; }
int iJIT_GetNewMethodID() { return 1; }
EOF

gcc -shared -fPIC -O2 -o "$CONDA_PREFIX/lib/libittnotify.so" "$CONDA_PREFIX/lib/itt_stub.c"

# Add the fix to conda activation so it persists
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export LD_PRELOAD=$CONDA_PREFIX/lib/libittnotify.so" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

conda deactivate 
conda activate doc_lm

bash get_data.sh


# Download trained model
pip install gdown
gdown 1ug-6ISrXHEGcWTk5KIw8Ojdjuww-i-Ci

tar -xzvf trainedmodel.tar.gz

python cal_ppl.py --data data/penn --save /home/jovyan/doc_lm/trainedmodel/ptb/additional_finetuned.pt --bptt 1000