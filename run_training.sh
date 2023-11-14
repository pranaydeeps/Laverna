export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_PORT=29500

export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch \
 --nproc_per_node=$N_GPU_NODE \
 --master_port=$MASTER_PORT \
 --nnodes=$N_NODES \
 --node_rank=$NODE_RANK \
train.py --n_gpu $N_GPU_NODE --student_type distilbert --student_config training_configs/distilbert-base-multilingual-cased.json --teacher_type bert --teacher_name bert-base-multilingual-uncased \
--alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm --freeze_pos_embs --dump_path distilSL_mBERT --data_file data/binarized_sl.txt.bert-base-multilingual-uncased.pickle \
--token_counts data/token_counts.sl.bert-base-multilingual-uncased.pickle --force --batch_size 2 --student_pretrained_weights init_checkpoint/distil_mBERT_init.pth --n_epoch 20
