import os
import torch
import argparse
import numpy as np
import torch.nn as nn

from dataset import DNLIDataset
from torch.utils.data import DataLoader
from model import RobertaClassifier
from utils import to_var
from tqdm import tqdm

from sklearn import metrics


def parse_arguments(parser):
    parser.add_argument('--data_root_path', type=str, default="../data/dnli/", help='root path of dataset')
    parser.add_argument('--save_path', type=str, default="./models", help='root path of saved model')
    parser.add_argument('--save_model_name', type=str, default="roberta_nli_classification_large.model",
                        help='name of saved model')
    parser.add_argument('--class_num', type=int, default=3, help='number of classes')
    parser.add_argument('--fnn_size', type=int, default=256, help='hidden size of the fc layer in the model')
    parser.add_argument('--pretrained_model_dim', type=int, default=1024, help='')
    # parser.add_argument('--pretrained_model_dim', type=int, default=768, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--contain_extra_data', type=bool, default=False, help='whether to leverage the extra data')
    parser.add_argument('--test_on_gold', type=bool, default=False,
                        help='whether to test on gold set (details listed in the paper)')
    parser.add_argument('--test_only', type=bool, default=True, help='test mode')
    parser.add_argument('--patient', type=int, default=100, help='patient epochs')
    parser.add_argument('--output', type=str, default='phase_2')
    parser.add_argument('--gpu', type=str, default='0')

    return parser


def test_model(args, model_path):
    test_dataset = DNLIDataset('test_phase_2_update.csv', mode=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print('# -------------------- building model -------------------- #')
    model = RobertaClassifier(args)

    if torch.cuda.is_available():
        print("use CUDA")
        model.cuda()

    # Loss and Optimizer
    print('# -------------------- Testing -------------------- #')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("No model is loaded from disk.", "Model path:", model_path)
    if torch.cuda.is_available():
        model.cuda()
    target = open(args.output + '.txt', 'w+')
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (text_data, mask, labels) in enumerate(tqdm(test_loader)):
        test_text, test_mask, test_labels = to_var(text_data), to_var(mask), to_var(labels)
        test_outputs = model(test_text, test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
        test_pred += test_argmax.data.cpu().tolist()
        test_true += test_labels.data.cpu().tolist()
        test_score += torch.softmax(test_outputs, dim=-1).data.cpu().tolist()
    for item in test_pred:
        target.write(str(item) + '\n')

    # test_accuracy = metrics.accuracy_score(test_true, test_pred)
    # test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    # test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    # test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    # test_auc_roc = metrics.roc_auc_score(test_true, test_score, average='macro', multi_class='ovo')
    #
    # test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)
    #
    # print("Classification Acc: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, AUC-ROC: %.4f"
    #       % (test_accuracy, test_precision, test_recall, test_f1, test_auc_roc))
    # print("Classification report:\n%s\n"
    #       % (metrics.classification_report(test_true, test_pred)))
    # print("Classification confusion matrix:\n%s\n"
    #       % (test_confusion_matrix))


def main(args):
    print("# -------------------- Load, tokenize data -------------------- #")
    train_dataset = DNLIDataset('./train_update_aug4.csv', mode='train')
    dev_dataset = DNLIDataset('./train_update_aug4.csv', mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=4, shuffle=False)
    test_dataset = DNLIDataset('test_phase_1_update.csv', mode=None)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    print('# -------------------- building model -------------------- #')
    model = RobertaClassifier(args)

    if torch.cuda.is_available():
        print("use CUDA")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr=args.learning_rate, weight_decay=0.1)

    print("loader size " + str(len(train_loader)))

    print('# -------------------- training -------------------- #')
    best_dev_acc = 0.000
    best_dev_dir = ""

    # Train the Model
    patient_counter = 0
    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        lr = args.learning_rate / (1. + 10 * p) ** 0.75

        optimizer.lr = lr
        accuracies = []
        train_losses = []
        dev_accuracies = []
        dev_losses = []

        model.train()
        for i, (text_data, mask, labels) in enumerate(tqdm(train_loader)):
            train_text, train_mask, train_labels = to_var(text_data), to_var(mask), to_var(labels)
            # start_time = time.time()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            class_outputs = model(train_text, train_mask)

            class_loss = criterion(class_outputs, train_labels)
            loss = class_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)

            accuracy = (train_labels == argmax.squeeze()).float().mean()

            train_losses.append(class_loss.item())
            accuracies.append(accuracy.item())

        model.eval()
        dev_accuracies_temp = []
        for i, (text_data, mask, labels) in enumerate(tqdm(dev_loader)):
            dev_text, dev_mask, dev_labels = to_var(text_data), to_var(mask), to_var(labels)
            dev_outputs = model(dev_text, dev_mask)
            _, dev_argmax = torch.max(dev_outputs, 1)
            dev_loss = criterion(dev_outputs, dev_labels)
            dev_accuracy = (dev_labels == dev_argmax.squeeze()).float().mean()
            dev_losses.append(dev_loss.item())
            dev_accuracies_temp.append(dev_accuracy.item())
        dev_acc = np.mean(dev_accuracies_temp)
        dev_accuracies.append(dev_acc)
        print('Epoch [%d/%d],  Loss: %.4f, Train_Acc: %.4f,  Development_Acc: %.4f.'
              % (
                  epoch + 1, args.num_epochs,
                  float(np.mean(train_losses)),
                  float(np.mean(accuracies)),
                  float(np.mean(dev_accuracies))
                )
              )

        if dev_acc > best_dev_acc:
            print("New best results:", dev_acc)
            best_dev_acc = dev_acc
            if not os.path.exists(os.path.join(args.save_path, args.output)):
                os.mkdir(os.path.join(args.save_path, args.output))

            best_dev_dir = os.path.join(args.save_path, args.output, args.save_model_name)
            torch.save(model.state_dict(), best_dev_dir)
            print("Model is saved in", best_dev_dir)
            patient_counter = 0
        else:
            patient_counter += 1
            print("Patient", patient_counter)
        if patient_counter >= args.patient:
            print("Out of patience.")
            break
    # Test
    print('# -------------------- Testing -------------------- #')
    if os.path.exists(best_dev_dir):
        model.load_state_dict(torch.load(best_dev_dir))
    else:
        print("No model is loaded from disk.", "Model path:", best_dev_dir)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    target = open(args.output + '.txt', 'w+')

    for i, (text_data, mask, labels) in enumerate(test_loader):
        test_text, test_mask, test_labels = to_var(text_data), to_var(mask), to_var(labels)
        test_outputs = model(test_text, test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
        test_pred += test_argmax.data.cpu().tolist()
        test_true += test_labels.data.cpu().tolist()
        test_score += torch.softmax(test_outputs, dim=-1).data.cpu().tolist()
    for item in test_pred:
        target.write(str(item) + '\n')

    # test_accuracy = metrics.accuracy_score(test_true, test_pred)
    # test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    # test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    # test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    # test_auc_roc = metrics.roc_auc_score(test_true, test_score, average='macro', multi_class='ovo')
    #
    # test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)
    #
    # print("Classification Acc: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, AUC-ROC: %.4f"
    #       % (test_accuracy, test_precision, test_recall, test_f1, test_auc_roc))
    # print("Classification report:\n%s\n"
    #       % (metrics.classification_report(test_true, test_pred)))
    # print("Classification confusion matrix:\n%s\n"
    #       % (test_confusion_matrix))


if __name__ == '__main__':

    classifier_parse = argparse.ArgumentParser()
    classifier_parser = parse_arguments(classifier_parse)
    args_all = classifier_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args_all.gpu
    if args_all.test_only:
        test_model(args_all, "/home/yuxiao/roberta/models/roberta_large_mnli_0.000002_30epoch_fnn128_drop60/roberta_nli_classification_large.model")
        # test_model(args_all, "../data/models/roberta_nli_classification_large.model")
    else:
        main(args_all)
