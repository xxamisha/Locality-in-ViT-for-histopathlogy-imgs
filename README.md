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
4. ablation_test.py - Runs both supplementary ablations at the tuned config: epoch counts (1/3/5) and local-layer counts (4/6/8/10/12), with 3 seeds per condition. Results are saved separately under `checkpoints/ablation_epochs` and `checkpoints/ablation_local_layers`.
   The epoch-count runs also save per-epoch training loss and validation accuracy for the convergence plot. Re-running a completed epoch condition will retrain only seeds whose epoch histories are missing, so the plot has data for all three seeds.
   To fill in missing epoch histories without running the local-layer ablation, run `py ablation_test.py --only-epochs`, then `py analyze_results.py`. The convergence figure overlays the 1-, 3-, and 5-epoch conditions in distinct colors.

running order:
1. py run_sweep_local.py - baseline comparison (vanilla vs untuned GPSA) 
2. py gpsa_hparam_search.py - find best gating_init / new_lr [OPTIONAL EXPERIMENT]
3. py optimal_parameters.py - headline result at tuned config 
4. py ablation_test.py - Optional epoch-count and local-layer ablations
7. py analyze_results.py - tables + figures 

Output:
At the end of very seed run, you will result in a averaged accuracy and wilxscon mean after the full comparison seed run has taken place. 
The analyze_results.py will output every image that is contained in the Analysis_imgs along with tables. 
The terminal printed output: 

               model |        metric |   n |    mean |     std |     min |     max
--------------------------------------------------------------------------------
             vanilla |  test_ood_acc |  10 |  0.9427 |  0.0221 |  0.9078 |  0.9653
      GPSA (default) |  test_ood_acc |  10 |  0.9414 |  0.0162 |  0.9087 |  0.9588
        GPSA (tuned) |  test_ood_acc |  10 |  0.9511 |  0.0115 |  0.9304 |  0.9687
saved table to ./results_table_val_ood_acc.csv

               model |        metric |   n |    mean |     std |     min |     max
--------------------------------------------------------------------------------
             vanilla |   val_ood_acc |  10 |  0.9650 |  0.0037 |  0.9588 |  0.9715
      GPSA (default) |   val_ood_acc |  10 |  0.9301 |  0.0094 |  0.9146 |  0.9418
        GPSA (tuned) |   val_ood_acc |  10 |  0.9590 |  0.0027 |  0.9546 |  0.9621
        seeds with results in both: 10

vanilla-phikon: mean=0.9427 std=0.0210
GPSA-phikon:    mean=0.9414 std=0.0154

Wilcoxon signed-rank: statistic=19.0000, p=4.3164e-01
Paired t-test:        t=-0.1970, p=8.4818e-01
Cohen's d (paired):   -0.0623
seeds with results in both: 10

vanilla-phikon: mean=0.9427 std=0.0210
GPSA-phikon:    mean=0.9511 std=0.0109

Wilcoxon signed-rank: statistic=19.0000, p=4.3164e-01
Paired t-test:        t=0.9513, p=3.6630e-01
Cohen's d (paired):   0.3008

     sweep |    value |   n | val_acc mean | val sample SD | test_acc mean | test sample SD
--------------------------------------------------------------------------------
gating_init |      0.0 |   5 |       0.9527 |        0.0029 |        0.9590 |         0.0072
gating_init |      0.5 |   5 |       0.9391 |        0.0066 |        0.9559 |         0.0110
gating_init |      1.0 |   5 |       0.9283 |        0.0092 |        0.9385 |         0.0194
    new_lr |   0.0001 |   5 |       0.9312 |        0.0072 |        0.9462 |         0.0133
    new_lr |   0.0005 |   5 |       0.9435 |        0.0040 |        0.9513 |         0.0125
    new_lr |    0.001 |   5 |       0.9469 |        0.0095 |        0.9559 |         0.0128

local_layers |   n | val_acc mean | val sample SD | test_acc mean | test sample SD
---------------------------------------------------------------------------
           4 |   3 |       0.9606 |        0.0028 |        0.9474 |         0.0148
           6 |   3 |       0.9647 |        0.0041 |        0.9309 |         0.0113
           8 |   3 |       0.9570 |        0.0039 |        0.9638 |         0.0084
          10 |   3 |       0.9580 |        0.0011 |        0.9573 |         0.0075
          12 |   3 |       0.9561 |        0.0031 |        0.9493 |         0.0218

  epochs |   n | val_acc mean | val sample SD | test_acc mean | test sample SD
----------------------------------------------------------------------
       1 |   3 |       0.9580 |        0.0011 |        0.9573 |         0.0075
       3 |   3 |       0.9567 |        0.0029 |        0.9382 |         0.0224
       5 |   3 |       0.9630 |        0.0018 |        0.9496 |         0.0154
vanilla-phikon: 85.80M total params, 85.80M trainable
GPSA-phikon (tuned config): 85.80M total params, 85.80M trainable
        
