# Locality-in-ViT-for-histopathlogy-imgs

# Set Up

Must have Python 3.13 (or 3.11/3.12) installed, accessible via 'py' launcher on Windows.
I used GPU from my PC of Nividea 3070 - this is recomneded for similar results.
1. Create and activate a virtual environment:
   py -m venv venv
   venv\Scripts\activate
2. install dependencies:
   py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   py -m pip install transformers datasets scipy numpy matplotlib

All files must be run from the same folder with all .py files together since they import from eachother. 

Files: 
1. gpsa.py - The GPSA attention layer itself (content + positional attention, gated combination)
2. phikon_gpsa.py - injects gpsa into pretrained phikon backbone and transplanting pretrained Queires/key/value output weights. 
3. seed_sweep.py - runs one model across multiple seeds with checkpointing with optional LR
4. paired_seed_comparison.py - runs vinalla phikon and GPSA phikon together on same seeds plus the paired statistical test. 

Runner Scripts:
1. run_sweep_local.py - main 10 seed vinalla vs default-config GPSA. Takes around 20.6 hours
2. gpsa_hparam_search.py - cheap search only 5 seeds over gating init and new lr to get optimal parameters. Around 31 hours
3. optimal_parameters.py - using those optimal values and running it on gpsa and compares against vinalla - 10.3 hours.
4. ablation_epochs.py - does the epoch count ablation at the tuned config of (1/3/5) epochs and 3 seeds each. Around 24.3 hours. 
5. ablation_local_layers.py - ablation at the tuned config (4/6/8/10/12, 3 seeds each). Around 15.5 hours. 

running order:
1. py run_sweep_local.py - baseline comparison (vanilla vs untuned GPSA)
2. py gpsa_hparam_search.py - find best gating_init / new_lr
3. py run_combined_config.py - headline result at tuned config
4. py ablation_epochs.py - supplementary
5. py ablation_local_layers.py - supplementary
6. py analyze_results.py - tables + figures 

