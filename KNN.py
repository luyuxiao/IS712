import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, BertModel, AutoModel, AutoModelForSequenceClassification
import os
from sklearn import svm
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


class RobertaClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fnn_size = args.fnn_size
        self.class_num = args.class_num
        self.roberta = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").base_model
        for param in self.roberta.parameters():
            param.requires_grad = True
        self.fc1 = nn.Linear(args.pretrained_model_dim, self.fnn_size)
        self.fc2 = nn.Linear(self.fnn_size, self.class_num)

    def forward(self, text_ids, mask):
        features = self.roberta(text_ids, mask).last_hidden_state  # batch_size, seq_len, dim

        return features


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
    parser.add_argument('--test_only', type=bool, default=False, help='test mode')
    parser.add_argument('--patient', type=int, default=100, help='patient epochs')
    parser.add_argument('--output', type=str, default='result')
    parser.add_argument('--gpu', type=str, default='0')

    return parser


def test_model(args, model_path):
    test_dataset = test_dataset = DNLIDataset('test_phase_1_update.csv', mode=None)
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


def main(args):
    print("# -------------------- Load, tokenize data -------------------- #")
    train_dataset = DNLIDataset('./train_update.csv', mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = DNLIDataset('test_phase_1_update.csv', mode=None)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    print('# -------------------- building model -------------------- #')
    model = RobertaClassifier(args)

    if torch.cuda.is_available():
        print("use CUDA")
        model.cuda()

    print("loader size " + str(len(train_loader)))

    print('# -------------------- training -------------------- #')
    best_dev_acc = 0.000
    best_dev_dir = ""

    # Train the Model
    patient_counter = 0

    features = []
    labels = []
    model.eval()
    for i, (text_data, mask, label) in enumerate(tqdm(train_loader)):
        train_text, train_mask, train_labels = to_var(text_data), to_var(mask), to_var(label)
        faeature = model(train_text, train_mask)
        features.append(faeature)
        labels.append(label)
        a = 0
    clf = svm.SVC()

    # Test
    print('# -------------------- Testing -------------------- #')

    target = open(args.output + '.txt', 'w+')
    test_pred = []
    for i, (text_data, mask, labels) in enumerate(test_loader):
        test_text, test_mask, test_labels = to_var(text_data), to_var(mask), to_var(labels)

    for item in test_pred:
        target.write(str(item) + '\n')


if __name__ == '__main__':

    classifier_parse = argparse.ArgumentParser()
    classifier_parser = parse_arguments(classifier_parse)
    args_all = classifier_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args_all.gpu
    if args_all.test_only:
        test_model(args_all, "/home/yuxiao/roberta/models/roberta_large_mnli_0.000002_30epoch_fnn128_augmentation/roberta_nli_classification_large.model")
        # test_model(args_all, "../data/models/roberta_nli_classification_large.model")
    else:
        main(args_all)
