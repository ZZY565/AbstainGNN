ARCH=gcn        
LOSS=ours_proto
REWARDS=4.6    
MOM=0.9
M=0.99
K=100
T=0.07

DATASET=IMDBBINARY
SEED=1111       
PRETRAIN=100
EPOCHS=150
LR=0.01

RHO=0.5          
GAMMA=0.02      
ALPHA=0.1     
CONF_WEIGHT=1    


while getopts ":s:" flag; do
  case "${flag}" in
    s) SEED=${OPTARG};;
    :) echo "Error: -${OPTARG} requires an argument."; exit 1;;
    *) ;;
  esac
done

SAVE_DIR="./log/${DATASET}_${ARCH}_${LOSS}_rho${RHO}_gamma${GAMMA}_alpha${ALPHA}_conf${CONF_WEIGHT}_seed-${SEED}"
mkdir -p ${SAVE_DIR}

echo "🚀 [Stage 1] Training on ${DATASET} with ${ARCH}"
python -u train_AbstainGNN.py \
  --arch ${ARCH} \
  --loss ${LOSS} \
  --dataset ${DATASET} \
  --pretrain ${PRETRAIN} \
  --sat-momentum ${MOM} \
  --moco-m ${M} \
  --moco-k ${K} \
  --lr ${LR} \
  --rewards ${REWARDS} \
  --moco-t ${T} \
  --manualSeed ${SEED} \
  --epochs ${EPOCHS} \
  --train-batch 64 \
  --test-batch 128 \
  --save ${SAVE_DIR} \
  \
  --ours_rho ${RHO} \
  --ours_gamma ${GAMMA} \
  --ours_alpha_intra ${ALPHA} \
  --conf_weight ${CONF_WEIGHT} \
  2>&1 | tee -a ${SAVE_DIR}/train.log

echo "🧠 [Stage 2] Evaluation mode..."
python -u train_moco_dev_AbstainGNN.py \
  --arch ${ARCH} \
  --loss ${LOSS} \
  --dataset ${DATASET} \
  --manualSeed ${SEED} \
  --moco-m ${M} \
  --moco-k ${K} \
  --pretrain ${PRETRAIN} \
  --rewards ${REWARDS} \
  --moco-t ${T} \
  --evaluate \
  --epochs ${EPOCHS} \
  --save ${SAVE_DIR} \
  \
  --ours_rho ${RHO} \
  --ours_gamma ${GAMMA} \
  --ours_alpha_intra ${ALPHA} \
  --conf_weight ${CONF_WEIGHT} \
  \
  2>&1 | tee -a ${SAVE_DIR}/eval.log
