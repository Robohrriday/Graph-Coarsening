import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net
import numpy as np
from utils import load_data, coarsening
import os
import sys
import csv
from pympler import asizeof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_features, args.num_classes, candidate, C_list, Gc_list = coarsening(args.dataset, 1-args.coarsening_ratio, args.coarsening_method)
    model = Net(args).to(device)
    all_acc = []

    for i in range(args.runs):

        data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = load_data(
            args.dataset, candidate, C_list, Gc_list, args.experiment)
        # a = asizeof.asizeof(data)
        b = sys.getsizeof(coarsen_features[0][0])*coarsen_features.shape[0]*coarsen_features.shape[1]
        c = sys.getsizeof(coarsen_train_labels[0])*coarsen_train_labels.shape[0]
        d = sys.getsizeof(coarsen_train_mask[0])*coarsen_train_mask.shape[0]
        e = sys.getsizeof(coarsen_val_labels[0])*coarsen_val_labels.shape[0]
        f = sys.getsizeof(coarsen_val_mask[0])*coarsen_val_mask.shape[0]
        g = sys.getsizeof(coarsen_edge[0][0])*coarsen_edge.shape[0]*coarsen_edge.shape[1]
        # print(a)
        data = data.to(device)
        # print(coarsen_edge)
        # print(coarsen_edge.shape)
        # print(sys.getsizeof(data.x[0][0])*data.x.shape[0]*data.x.shape[1])
        h = sys.getsizeof(data.x[0][0])*data.x.shape[0]*data.x.shape[1]
        i = sys.getsizeof(data.y[0])*data.y.shape[0]
        j = sys.getsizeof(data.edge_index[0][0])*data.edge_index.shape[0]*data.edge_index.shape[1]
        dictt = {
        # "data": a,
        "coarsen features": b,
        "coarsen train labels": c,
        "coarsen train mask": d,
        "coarsen val labels": e,
        "coarsen val mask": f,
        "coarsen edge": g,
        "data.x": h,
        "data.y": i,
        "data.edge_index": j,
    }

    # Calculate sum
    total_size = sum(dictt.values())

    # Write to CSV
    with open('sizes.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dataset',args.dataset])
        writer.writerow(['Coarsening ratio',args.coarsening_ratio])
        # writer.writerow(['Name', 'Size'])
        # for name, size in dictt.items():
        #     writer.writerow([name, size])
        n = []
        sizee = []
        for name, size in dictt.items():
            n.append(name)
            sizee.append(size)

        # writer.writerow(['Total', total_size])
        n.append("Total")
        sizee.append(total_size)
        writer.writerow(n)
        writer.writerow(sizee)
        writer.writerow([])

        coarsen_features = coarsen_features.to(device)
        coarsen_train_labels = coarsen_train_labels.to(device)
        coarsen_train_mask = coarsen_train_mask.to(device)
        coarsen_val_labels = coarsen_val_labels.to(device)
        coarsen_val_mask = coarsen_val_mask.to(device)
        coarsen_edge = coarsen_edge.to(device)

        if args.normalize_features:
            coarsen_features = F.normalize(coarsen_features, p=1)
            data.x = F.normalize(data.x, p=1)

    #     model.reset_parameters()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #     best_val_loss = float('inf')
    #     val_loss_history = []
    #     for epoch in range(args.epochs):

    #         model.train()
    #         optimizer.zero_grad()
    #         out = model(coarsen_features, coarsen_edge)
    #         loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
    #         loss.backward()
    #         optimizer.step()

    #         model.eval()
    #         pred = model(coarsen_features, coarsen_edge)
    #         val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()

    #         if val_loss < best_val_loss and epoch > args.epochs // 2:
    #             best_val_loss = val_loss
    #             torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')

    #         val_loss_history.append(val_loss)
    #         if args.early_stopping > 0 and epoch > args.epochs // 2:
    #             tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
    #             if val_loss > tmp.mean().item():
    #                 break

    #     model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
    #     model.eval()
    #     pred = model(data.x, data.edge_index).max(1)[1]
    #     test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
    #     print(test_acc)
    #     all_acc.append(test_acc)

    # print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))

