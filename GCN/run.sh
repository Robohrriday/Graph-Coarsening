#To overcome over-fitting, we fine-tuned the hyperparameter settings in the few-shot regime.
#To overcome over-smoothing, we fine-tuned the hyperparameter settings when the coarsening ratio is 0.1.
#example, the coarsening ratio is 0.5
# python train.py --dataset cora --experiment fixed --coarsening_ratio 0.5
# python train.py --dataset cora --experiment few --epoch 100 --coarsening_ratio 0.5
# python train.py --dataset citeseer --experiment fixed --epoch 200 --coarsening_ratio 0.5
# python train.py --dataset pubmed --experiment fixed --epoch 200 --coarsening_ratio 0.5
# python train.py --dataset pubmed --experiment few --epoch 60 --coarsening_ratio 0.5
# python train.py --dataset dblp --experiment random --epoch 50 --coarsening_ratio 0.5
# python train.py --dataset Physics --experiment random --epoch 200 --lr 0.001 --weight_decay 0 --coarsening_ratio 0.5


python train.py --hidden 512 --dataset cora --experiment fixed --coarsening_ratio 0 --runs 50
python train.py --hidden 512 --dataset cora --experiment fixed --coarsening_ratio 0.1 --runs 50
python train.py --hidden 512 --dataset cora --experiment fixed --coarsening_ratio 0.3 --runs 50
python train.py --hidden 512 --dataset cora --experiment fixed --coarsening_ratio 0.5 --runs 50
python train.py --hidden 512 --dataset cora --experiment fixed --coarsening_ratio 0.7 --runs 50


python train.py --hidden 512 --dataset citeseer --experiment fixed --coarsening_ratio 0 --runs 50
python train.py --hidden 512 --dataset citeseer --experiment fixed --coarsening_ratio 0.1 --runs 50
python train.py --hidden 512 --dataset citeseer --experiment fixed --coarsening_ratio 0.3 --runs 50
python train.py --hidden 512 --dataset citeseer --experiment fixed --coarsening_ratio 0.5 --runs 50
python train.py --hidden 512 --dataset citeseer --experiment fixed --coarsening_ratio 0.7 --runs 50


python train.py --hidden 512 --dataset pubmed --experiment fixed --coarsening_ratio 0 --runs 50
python train.py --hidden 512 --dataset pubmed --experiment fixed --coarsening_ratio 0.1 --runs 50
python train.py --hidden 512 --dataset pubmed --experiment fixed --coarsening_ratio 0.3 --runs 50
python train.py --hidden 512 --dataset pubmed --experiment fixed --coarsening_ratio 0.5 --runs 50
python train.py --hidden 512 --dataset pubmed --experiment fixed --coarsening_ratio 0.7 --runs 50


python train.py --hidden 512 --dataset dblp --experiment random --coarsening_ratio 0 --runs 50
python train.py --hidden 512 --dataset dblp --experiment random --coarsening_ratio 0.1 --runs 50
python train.py --hidden 512 --dataset dblp --experiment random --coarsening_ratio 0.3 --runs 50
python train.py --hidden 512 --dataset dblp --experiment random --coarsening_ratio 0.5 --runs 50
python train.py --hidden 512 --dataset dblp --experiment random --coarsening_ratio 0.7 --runs 50


python train.py --hidden 512 --dataset Physics --experiment random --coarsening_ratio 0 --runs 50
python train.py --hidden 512 --dataset Physics --experiment random --coarsening_ratio 0.1 --runs 50
python train.py --hidden 512 --dataset Physics --experiment random --coarsening_ratio 0.3 --runs 50
python train.py --hidden 512 --dataset Physics --experiment random --coarsening_ratio 0.5 --runs 50
python train.py --hidden 512 --dataset Physics --experiment random --coarsening_ratio 0.7 --runs 50