"""
@ModuleName: train
@Description: 
@Author: MRhu
@Date: 2024-03-15 14:44
"""
import os.path

from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DistributedSampler, DataLoader, SequentialSampler

from data_util import convert_examples_to_features, get_data, get_label
import torch
from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam
from transformers import BertModel, BertTokenizer
import logging
import argparse
from tqdm import tqdm, trange
import numpy as np
from func import compute_metrics

# 日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
# BERT路径
BERT_PATH = 'model/bert_uncased'


# TODO 添加参数
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
                    default='',
                    type=str,
                    required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--train_batch_size",
                    default=128,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=128,
                    type=int,
                    help="Total batch size for evaluation")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
args = parser.parse_args()

# TODO 准备模型
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
label_list = get_label(args.data_dir)
train_examples = get_data('train_examples', args.data_dir)
train_features = convert_examples_to_features(train_examples, label_list, 64, tokenizer)
num_labels = len(label_list)
model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=num_labels)

# TODO 使用gpu进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) \
                               * args.num_train_epochs
# TODO Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=args.learning_rate,
                     warmup=args.warmup_proportion,
                     t_total=num_train_optimization_steps)

# TODO train
global_step = 0
nb_tr_steps = 0
tr_loss = 0
logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", args.train_batch_size)
logger.info("  Num steps = %d", num_train_optimization_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).to(device)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).to(device)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).to(device)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long).to(device)

train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
model.train().to(device)
torch.cuda.empty_cache()

for _ in trange(int(args.num_train_epochs), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # define a new function to compute loss values for both output_modes
        logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
    print("Training loss: ", tr_loss, nb_tr_examples)
    # TODO 保存整个模型
    torch.save(model, os.path.join(args.output_dir, 'model.pth'))
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_weights.pth'))
    tokenizer.save_vocabulary(args.output_dir)

# TODO 评估模型
eval_examples = get_data('dev_examples', args.data_dir)
eval_features = convert_examples_to_features(
    eval_examples, label_list, 64, tokenizer)
logger.info("***** Running evaluation *****")
logger.info("  Num examples = %d", len(eval_examples))
logger.info("  Batch size = %d", args.eval_batch_size)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=args.eval_batch_size)

# Load a trained model and vocabulary that you have fine-tuned
model_load = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=num_labels)
model_load.load_state_dict(torch.load('output/model_weights.pth'))
tokenizer = BertTokenizer.from_pretrained(args.output_dir)
model_load.to(device)

model_load.eval()
eval_loss = 0.0
nb_eval_steps = 0
preds = []

for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        logits = model_load(input_ids, segment_ids, input_mask, labels=None)

    # create eval loss and other metric required by the task
    loss_fct = CrossEntropyLoss()
    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    print(label_ids.view(-1))

    eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if len(preds) == 0:
        preds.append(logits.detach().cpu().numpy())
    else:
        preds[0] = np.append(
            preds[0], logits.detach().cpu().numpy(), axis=0)

eval_loss = eval_loss / nb_eval_steps
preds = preds[0]

preds = np.argmax(preds, axis=1)
result = compute_metrics(preds, all_label_ids.numpy())
loss = tr_loss / nb_tr_steps

result['eval_loss'] = eval_loss
result['global_step'] = global_step
result['loss'] = loss

output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
