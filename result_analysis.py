import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time

def read_model_result(path):
    timestamp = []
    epoch = []
    train_time = []
    test_acc = []
    time_per = []
    time_total = []
    with open(path) as f:
        line = f.readline()
        while line != '':
            sp_line = line.split(' ')
            epoch.append(int(sp_line[2][:-1]))
            train_time.append(round(float(sp_line[5][:-1]), 2))
            line = f.readline()
            sp_line = line.split(' ')
            timestamp.append(time.mktime(time.strptime(sp_line[0], "%Y%m%d-%H:%M:%S")))
            test_acc.append(float(sp_line[4]))
            time_per.append(round(float(sp_line[6]), 2))
            time_total.append(round(float(sp_line[8][:-1]), 2))
            line = f.readline()
    model_result = pd.DataFrame({'timestamp': timestamp, 'epoch': epoch, 'train_time': train_time, 'test_acc': test_acc,
                                 'time_per': time_per, 'time_total': time_total})
    return model_result

def read_resource(path):
    timestamp=[]
    timestamp_gpu=[]
    total_memo_perc=[]
    p_memo_perc=[]
    total_cpu_per=[]
    p_cpu_per=[]
    pid=[]
    gpu_0=[]
    gpu_1=[]
    gpu_2=[]
    gpu_3=[]
    gpu_4=[]
    gpu_5=[]
    gpu_6=[]
    gpu_7=[]
    with open(path) as f:
        line = f.readline()
        while line:
            sp_line = line.split(' ')
            if sp_line[1] == 'virtual':
                pid.append(int(sp_line[6]))
                timestamp.append(time.mktime(time.strptime(sp_line[0], "%Y%m%d-%H:%M:%S")))
                total_memo_perc.append(float(sp_line[4][:-1]))
                p_memo_perc.append(float(sp_line[-1][:-2]))
                line = f.readline()
                sp_line = line.split(' ')
                total_cpu_per.append(float(sp_line[3][:-1]))
                p_cpu_per.append(float(sp_line[-1][:-2]))
                line = f.readline()
            elif sp_line[1] == 'GPU':
                timestamp_gpu.append(time.mktime(time.strptime(sp_line[0], "%Y%m%d-%H:%M:%S")))
                for gpu in [gpu_0, gpu_1, gpu_2, gpu_3, gpu_4, gpu_5, gpu_6, gpu_7]:
                    sp_line = line.split(' ')
                    gpu.append(float(sp_line[-1][:-1]))
                    line = f.readline()
            else:
                print('Error!',line)
                break

    cpu_memory=pd.DataFrame({'timestamp':timestamp,'pid':pid,'total_memo_perc':total_memo_perc,'p_memo_perc':p_memo_perc,
                           'total_cpu_per':total_cpu_per,'p_cpu_per':p_cpu_per})

    gpu=pd.DataFrame({'timestamp':timestamp_gpu,'gpu_0':gpu_0,'gpu_1':gpu_1,'gpu_2':gpu_2,
                           'gpu_3':gpu_3,'gpu_4':gpu_4,'gpu_5':gpu_5,'gpu_6':gpu_6,'gpu_7':gpu_7})
    return cpu_memory,gpu

if __name__== '__main__':
    zhfont1 = matplotlib.font_manager.FontProperties(family = 'Hiragino Sans GB',fname='/System/Library/Fonts/Hiragino Sans GB.ttc')
    path='/Users/dongzhejiang/Downloads/log_save/cifar10365.log'
    model_result=read_model_result(path)

    path='/Users/dongzhejiang/Downloads/Log_1/all_process.log'
    cpu_memory,gpu=read_resource(path)

    # 画训练集句长
    length1=count_length(train_path=r"./data1/train.txt")
    length2 = count_length (train_path=r"./data2/train.txt")
    length3 = count_length (train_path=r"./data3/train.txt")
    plt.figure()
    for length in [length1,length2,length3]:
        length = pd.Series(length)
        length.plot(kind='kde')
    # hist, bin_edges = np.histogram(length1,bins=200)
    # cdf = np.cumsum(hist / sum(hist))
    # plt.plot(bin_edges[1:], cdf,label='data 1')
    # hist, bin_edges = np.histogram(length2, bins=100)
    # cdf = np.cumsum(hist / sum(hist))
    # plt.plot(bin_edges[1:], cdf, label='data 2')
    # hist, bin_edges = np.histogram(length3, bins=100)
    # cdf = np.cumsum(hist / sum(hist))
    # plt.plot(bin_edges[1:], cdf, label='data 3')
    plt.xlim([-1,200])
    plt.ylim([0, 0.06])
    plt.grid()
    label=['data 1','data 2','data 3']
    plt.legend(label,fontsize=12)
    plt.ylabel('Density',fontsize=18)
    plt.xlabel('训练集句长', fontproperties=zhfont1, fontsize=18)
    plt.title('密度分布函数图', fontproperties=zhfont1,fontsize=20)
    plt.savefig('/Users/dongzhejiang/Downloads/word_length.png', dpi=300)
    plt.show()

    # 画三个model的f1
    f1_1=pd.read_table('data1_save/1545714654/summaries/report.txt',header=None)
    f1_2=pd.read_table('data2_save/1546036157/summaries/report.txt',header=None)
    f1_3=pd.read_table('data3_save/1546046955/summaries/report.txt',header=None)
    plt.figure()
    plt.plot(f1_1[1],f1_1[2],label='model 1')
    plt.plot(f1_2[1],f1_2[2], label='model 2')
    plt.plot(f1_3[1],f1_3[2], label='model 3')
    plt.ylabel('f1',fontsize=18)
    plt.xlabel('epoch',fontsize=18)
    plt.title('f1随epoch变化情况',fontproperties=zhfont1,fontsize=20)
    plt.legend(fontsize=12)
    plt.grid()
    plt.xlim([0, 90])
    plt.ylim([0, 0.9])
    plt.savefig('/Users/dongzhejiang/Downloads/f1.png', dpi=300)
    plt.show()

    # 画model 3的accuracy变化情况
    plt.figure()
    plt.plot(f1_3[1], f1_3[3], label='model 3')
    plt.ylabel('accuracy', fontsize=18)
    plt.xlabel('epoch', fontsize=18)
    plt.title('accuracy随epoch变化情况', fontproperties=zhfont1, fontsize=20)
    plt.grid()
    plt.xlim([0, 60])
    plt.ylim([0, 1])
    plt.savefig('/Users/dongzhejiang/Downloads/accuracy.png', dpi=300)
    plt.show()

    #获取时间间隔从而得到平均时间
    from datetime import datetime
    for f1 in [f1_1,f1_2,f1_3]:
        start_time=datetime.strptime(f1.iloc[0][0],"%Y/%m/%d %H:%M:%S")
        end_time=datetime.strptime(f1.iloc[len(f1)-1][0],"%Y/%m/%d %H:%M:%S")
        dur=((end_time-start_time).seconds)/60./len(f1)
        print(dur)

    #获取f1
    for f1 in [f1_1, f1_2, f1_3]:
        print(f1.iloc[len(f1)-1][2])

    #获取accuracy
    for f1 in [f1_1, f1_2, f1_3]:
        print(f1.iloc[len(f1)-1][3])

# for pid in psutil.pids():
#     p=psutil.Process(pid)
#     if p.username()=='weifeng':
#         if p.name()=='python':
#             if len(p.cmdline())>1:
#                 if p.cmdline()[1]=='cifar10.py':
#                     print(pid)
#
# import os
# def isRunning(pid):
#     try:
#         process = len(os.popen('ps aux | grep ' + str(pid) + ' | grep -v grep').readlines())
#         if process >= 1:
#             return True
#         else:
#             return False
#     except:
#         print("Check process ERROR!!!")
#         return False