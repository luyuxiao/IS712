import os
import json
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, AutoTokenizer
import time
import csv


class DNLIDataset(Dataset):
    def __init__(self, data_root_path, mode=None, contain_extra_data=False, test_on_gold=False):
        # if not os.path.exists(data_root_path):
        #     raise ValueError("Dataset not exists in", data_root_path)
        # if contain_extra_data:
        #     if test_on_gold and data_type == "test":
        #         data_path1 = os.path.join(data_root_path, "dialogue_nli", "dialogue_nli_verified_"+data_type+".jsonl")
        #     else:
        #         data_path1 = os.path.join(data_root_path, "dialogue_nli", "dialogue_nli_"+data_type+".jsonl")
        #     data_path2 = os.path.join(data_root_path, "dialogue_nli_extra", "dialogue_nli_EXTRA_uu_"+data_type+".jsonl")
        # else:
        #     if test_on_gold and data_type == "test":
        #         data_path1 = os.path.join(data_root_path, "dialogue_nli", "dialogue_nli_verified_"+data_type+".jsonl")
        #     else:
        #         data_path1 = os.path.join(data_root_path, "dialogue_nli", "dialogue_nli_"+data_type+".jsonl")
        #     data_path2 = ""
        # start_time = time.time()
        # json_data1 = self.reformat_data(data_path1)
        # print("load data1 time:", time.time()-start_time)
        # start_time = time.time()
        # json_data2 = self.reformat_data(data_path2) if data_path2 != "" else []
        # print("load data2 time:", time.time()-start_time)
        csv_reader = csv.reader(open(data_root_path))
        self.sentences1 = []
        self.sentences2 = []
        self.labels = []

        index = 0
        for item in csv_reader:
            if item[0] == 'ID':
                continue
            self.sentences1.append(item[1])
            self.sentences2.append(item[2])
            # self.sentences1.append(item[2])
            # self.sentences2.append(item[1])
            if len(item) < 4:
                item_label = '0'
            else:
                item_label = item[3]
            if item_label == "0":
                self.labels.append(0)
                # self.labels.append(0)
            elif item_label == "1":
                self.labels.append(1)
                # self.labels.append(1)
            else:
                assert item_label == "2"
                self.labels.append(2)
                # self.labels.append(2)
            index += 1
        # if not json_data2 == []:
        #     for item in json_data2:
        #         self.sentences1.append(item['sentence1'])
        #         self.sentences2.append(item['sentence2'])
        #         item_label = item['label']
        #         if item_label == "positive":
        #             self.labels.append(0)
        #         elif item_label == "neutral":
        #             self.labels.append(1)
        #         else:
        #             assert item_label == "negative", [item_label, item]
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        #     if mode == 'train':
        #         if index >= 1000:
        #             break
        #     else:
        #         csv_reader = csv_reader[1000:]
        # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        # tokenizer = AutoTokenizer.from_pretrained("gsarti/scibert-nli")
        # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        # tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-distilroberta-base")
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
        # tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-distilroberta-base")
        start_time = time.time()
        tokenized_text = tokenizer(self.sentences1, self.sentences2,
                                   padding=True, truncation=True, return_tensors="pt", max_length=256)
        print("Tokenization time:", time.time()-start_time)
        self.input_text_ids = tokenized_text["input_ids"]
        self.mask = tokenized_text["attention_mask"]
        print("Text:", len(self.input_text_ids), "mask:", len(self.mask), "Label:", len(self.labels))

    def __len__(self):
        return len(self.input_text_ids)

    def __getitem__(self, idx):
        return self.input_text_ids[idx], self.mask[idx], self.labels[idx]

    @classmethod
    def reformat_data(cls, data_path):
        with open(data_path) as f:
            data = f.readline()
            if "EXTRA_uu_" in data_path:
                data = data[:-44] + "]"
            data = data.strip()
            json_data = json.loads(data)
        return json_data


if __name__ == '__main__':
    # temp_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # temp_data = DNLIDataset("../data/dnli", "train")
    temp_data = DNLIDataset('test_phase_1_update.csv')
    train_temp_loader = DataLoader(dataset=temp_data, batch_size=64, shuffle=True)
    for i, (train_data, train_mask, train_label) in enumerate(train_temp_loader):
        print(1)
        print(train_data, train_mask, train_label)
        break

