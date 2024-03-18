"""
@ModuleName: predict
@Description: 
@Author: MRhu
@Date: 2024-03-18 9:35
"""
import os

import torch
from pytorch_pretrained_bert import BertForSequenceClassification, BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn import metrics

from data_util import convert_examples_to_features, get_data, get_label
import argparse
import logging
from func import compute_metrics

# TODO 日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
                    default='',
                    type=str,
                    required=True,
                    help="examples data path")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--eval_batch_size",
                    default=128,
                    type=int,
                    help="Total batch size for evaluation")
parser.add_argument("--BERT_PATH",
                    default='model/bert_uncased',
                    type=str,
                    help='BERT Model path')
args = parser.parse_args()

# TODO 使用gpu进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

# TODO 准备模型
label_list = get_label(args.data_dir)
num_labels = len(label_list)
model_load = BertForSequenceClassification.from_pretrained(args.BERT_PATH, num_labels=num_labels)
model_load.load_state_dict(torch.load('output/model_weights.pth'))
tokenizer = BertTokenizer.from_pretrained(args.output_dir)

# TODO 准备数据
eval_examples = get_data('test_examples', args.data_dir)
eval_features = convert_examples_to_features(eval_examples, label_list, 128, tokenizer)
logger.info("***** Running Prediction *****")
logger.info("  Num examples = %d", len(eval_examples))
logger.info("  Batch size = %d", 128)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=args.eval_batch_size)

model_load.to(device)
model_load.eval()
eval_loss = 0
nb_eval_steps = 0
preds = []

for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        logits = model_load(input_ids, segment_ids, input_mask, labels=None)

    loss_fct = CrossEntropyLoss()
    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

    eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if len(preds) == 0:
        preds.append(logits.detach().cpu().numpy())
    else:
        preds[0] = np.append(
            preds[0], logits.detach().cpu().numpy(), axis=0)

eval_loss = eval_loss / nb_eval_steps
preds = preds[0]
print(preds, preds.shape)

all_label_ids = all_label_ids.numpy()
preds = np.argmax(preds, axis=1)
result = compute_metrics(preds, all_label_ids)
result['eval_loss'] = eval_loss

output_eval_file = os.path.join(args.output_dir, "test_results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
print("Triple classification acc is : ")
print(metrics.accuracy_score(all_label_ids, preds))
