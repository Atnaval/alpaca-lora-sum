mkdir /workspace/cache
export TRANSFORMERS_CACHE=/workspace/cache/
pip install -r requirements.txt
export NCCL_P2P_DISABLE=1
