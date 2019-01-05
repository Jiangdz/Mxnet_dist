#-------------------------------------------------
# input process id
# logging memory, cpu, gpu
# output pid.log
# 2018-01-05 取消输入，监控全部cifar10相关
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
                        filename='Log_2/'+output_dir+'.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

# p = input("input process id: ")
head=int(time.time())
head='all_process_'+str(head)
logger = init_log(head)
_print = logger.info

# p1 = psutil.Process(int(p))

M=1024**2.

while(1):

    for pid in psutil.pids():
        try:
            p = psutil.Process(pid)
            if p.username() == 'weifeng':
                if p.name() == 'python':
                    if len(p.cmdline()) > 1:
                        if p.cmdline()[1] == 'cifar10.py':

                            _print('virtual memory percent: {} ,pid {} ,pid memory percent {}'.format(
                                (str)(psutil.virtual_memory().percent) + '%',str(pid),str(round(p.memory_percent()*100,1))+'%'))

                            _print('cpu percent: {} ,pid {} ,pid cpu percent: {}'.format(
                                (str)(psutil.cpu_percent()) + '%',str(pid),(str)(p.cpu_percent(1)) + '%'))
        except:
            pass
    for i in range(8):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total=(meminfo.total)/M
        used=(meminfo.used)/M
        free=(meminfo.free)/M
        perc=round(used/total*100,3)
        _print('GPU {}: total {}, used {}, free {}, used percent {}'.format(i,total,used,free,perc))

    time.sleep(30)





