srun -p mm_det \
    --gres=gpu:0 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=5 \
    --kill-on-bad-exit=1 \
    python tools/dataset_converters/create_depth_bin.py
