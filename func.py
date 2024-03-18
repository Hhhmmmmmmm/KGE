"""
@ModuleName: func
@Description: 
@Author: MRhu
@Date: 2024-03-18 10:25
"""


# TODO 计算准确率
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(preds, labels):
    return {"acc": simple_accuracy(preds, labels)}