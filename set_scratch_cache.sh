cat > /work/pi_jensen_umass_edu/ikarn_umass_edu/olmo_cladder_test/set_scratch_cache.sh <<'EOF'
export WS=/scratch/workspace/ikarn_umass_edu-olmo_cladder_cache

mkdir -p $WS/.cache/huggingface $WS/.cache/torch $WS/.cache/pip

export HF_HOME=$WS/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

export TORCH_HOME=$WS/.cache/torch
export PIP_CACHE_DIR=$WS/.cache/pip

echo "[cache] HF_HOME=$HF_HOME"
EOF