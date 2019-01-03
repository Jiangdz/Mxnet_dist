#-------------------------------------------------
# input process id
# logging memory, cpu, gpu
# output pid.log
#-------------------------------------------------

import logging
import psutil
import os
import time
import pynvml
pynvml.nvmlInit()

def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename='Log/'+output_dir+'.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

p = input("input process id: ")
logger = init_log('pid'+str(p))
_print = logger.info

p1 = psutil.Process(int(p))

M=1024**2.

while(1):

    _print('virtual memory percent: {}, pid memory percent {}'.format(
        (str)(psutil.virtual_memory().percent) + '%',str(round(p1.memory_percent()*100,1))+'%'))

    _print('cpu percent: {}, pid cpu percent: {}'.format((str)(psutil.cpu_percent()) + '%',
                                                        (str)(p1.cpu_percent()) + '%'))
    for i in range(8):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total=(meminfo.total)/M
        used=(meminfo.used)/M
        free=(meminfo.free)/M
        perc=round(used/total*100,3)
        _print('GPU {}: total {}, used {}, free {}, used percent {}'.format(i,total,used,free,perc))

    time.sleep(60)





