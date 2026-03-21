for b in libero_spatial libero_object libero_goal libero_10 libero_90; do
  python "/home/hubertchang/p-progress/lerobot-libero/notebooks/extract_libero_wm_data.py" \
    --benchmark-name "$b" \
    --task-ids all \
    --max-demos-per-task -1 \
    --fps 20 \
    --output-root "/tmp2/hubertchang/p-jepa/data/libero_wm_settle"
done

