"""
@ModuleName: until_test
@Description: 
@Author: MRhu
@Date: 2024-03-13 15:04
"""

import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

logger.info(device)
logger.info("gpu num is: {}".format(n_gpu))
