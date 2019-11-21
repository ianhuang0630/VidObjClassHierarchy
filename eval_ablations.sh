echo ACTION_EMBEDDING
python evaluate_ablations.py --our_model_path models/hierarchy_discovery/ablation_hierarchy_discovery_run1/net_epoch49.pth --num_workers 5
echo FUNCTION_ENCODING
python evaluate_ablations.py --our_model_path models/hierarchy_discovery/ablation_hierarchy_discovery_run2/net_epoch9.pth --num_workers 5
echo UNKNOWN_VISUAL
python evaluate_ablations.py --our_model_path models/hierarchy_discovery/ablation_hierarchy_discovery_run3/net_epoch20.pth --num_workers 5
echo KNOWN_VISUAL
python evaluate_ablations.py --our_model_path models/hierarchy_discovery/ablation_hierarchy_discovery_run4/net_epoch29.pth --num_workers 5
echo HAND
python evaluate_ablations.py --our_model_path models/hierarchy_discovery/ablation_hierarchy_discovery_run5/net_epoch16.pth --num_workers 5
echo VIDEO_VISUAL
python evaluate_ablations.py --our_model_path models/hierarchy_discovery/ablation_hierarchy_discovery_run6/net_epoch25.pth --num_workers 5
