echo "--- Creating Conda Environment 'doc_lm' ---"
conda update -n base -c conda-forge conda -y
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

tar -xzvf trainedmodel.tar.gz# Function to reorganize checkpoint files

reorganize_checkpoints() {
    local checkpoint_dir="$1"
    
    # Find all .pt files in the directory
    find "$checkpoint_dir" -maxdepth 1 -name "*.pt" -type f | while read -r checkpoint; do
        # Get filename without extension
        local filename=$(basename "$checkpoint" .pt)
        
        # Create directory for the model
        local model_dir="$checkpoint_dir/$filename"
        mkdir -p "$model_dir"
        
        # Move checkpoint to model.pt inside the directory
        mv "$checkpoint" "$model_dir/model.pt"
        
        echo "Reorganized: $filename -> $model_dir/model.pt"
    done
}

# Reorganize existing checkpoints before running the script
reorganize_checkpoints "/home/jovyan/doc_lm/trainedmodel/ptb"
reorganize_checkpoints "/home/jovyan/doc_lm/trainedmodel/ptb_ensemble"