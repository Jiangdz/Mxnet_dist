#-------------------------------------------------
# 结果分析
#-------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
import os

def read_model_result(path):
    timestamp = []
    epoch = []
    train_time = []
    test_acc = []
    time_per = []
    time_total = []
    with open(path) as f:
        line = f.readline()
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

def read_model_result_1(path):
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

def read_resource_1(path):
    timestamp=[]
    timestamp_gpu=[]
    total_memo_perc=[]
    p_memo_perc=[]
    total_cpu_per=[]
    p_cpu_per=[]
    # pid=[]
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
                # pid.append(int(sp_line[6]))
                timestamp.append(time.mktime(time.strptime(sp_line[0], "%Y%m%d-%H:%M:%S")))
                total_memo_perc.append(float(sp_line[4][:-2]))
                p_memo_perc.append(float(sp_line[-1][:-2]))
                line = f.readline()
                sp_line = line.split(' ')
                total_cpu_per.append(float(sp_line[3][:-2]))
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

    cpu_memory=pd.DataFrame({'timestamp':timestamp,'total_memo_perc':total_memo_perc,'p_memo_perc':p_memo_perc,
                           'total_cpu_per':total_cpu_per,'p_cpu_per':p_cpu_per})

    gpu=pd.DataFrame({'timestamp':timestamp_gpu,'gpu_0':gpu_0,'gpu_1':gpu_1,'gpu_2':gpu_2,
                           'gpu_3':gpu_3,'gpu_4':gpu_4,'gpu_5':gpu_5,'gpu_6':gpu_6,'gpu_7':gpu_7})
    return cpu_memory,gpu

def tran_per(num):
    num=round(num*100,2)
    num=str(num)+'%'
    return num

def plot_xy(x,y,min_x,ax):
    x=int(x)
    ax.text(x=min_x, y=y, s=tran_per(y),fontsize=12)
    ax.text(x=x, y=0, s=x,fontsize=12)
    ax.plot([x, x], [-0.5, y], linestyle='--',color='grey')
    ax.plot([min_x, x], [y, y], linestyle='--',color='grey')
    ax.scatter([x, ], [y, ],color='grey')

if __name__== '__main__':
    zhfont1 = matplotlib.font_manager.FontProperties(family='Hiragino Sans GB',
                                                     fname='/System/Library/Fonts/Hiragino Sans GB.ttc')

    # 遍历所有数据
    pid = []
    sync = []
    data_path = []
    gpus = []
    models = []
    cpus = []
    for root, dirs, files in os.walk('/Users/dongzhejiang/Documents/learning/云计算/图/Log_2'):
        for file in files:
            if file[0] == 'c':
                pass
            else:
                continue
            path = '/Users/dongzhejiang/Documents/learning/云计算/图/Log_2/' + file
            with open(path) as f:
                line = f.readline()
            sp_line = line.split(' ')
            pid.append(int(sp_line[2][:-1]))
            if sp_line[4] == 'dist_async_device,':
                sync.append(0)
            else:
                sync.append(1)
            gpus.append(int(sp_line[8][0]))
            models.append(sp_line[10][:-1])
            cpus.append(int(sp_line[-1][0]))
            data_path.append(path)
    data = pd.DataFrame({'pid': pid, 'sync': sync, 'gpus': gpus, 'models': models, 'cpus': cpus, 'path': data_path})

    #resnet acc
    model='resnet'
    fig, ax = plt.subplots()
    temp_data = data[data['models'] == model]
    for tag in ['GPU = 4', 'GPU = 1', 'GPU = 2']:
        if tag == 'GPU = 4':
            path = '/Users/dongzhejiang/Downloads/all_Log/resnet/cifar10362.log'
            model_result_1 = read_model_result_1(path)
            path = '/Users/dongzhejiang/Downloads/all_Log/resnet/cifar10365.log'
            model_result_2 = read_model_result_1(path)
            # data_path = temp_data[temp_data['cpus'] == 1]['path']
        elif tag == 'GPU = 1':
            data_path = temp_data[temp_data['gpus'] == 1]['path']
            path = data_path.iloc[0]
            model_result_1 = read_model_result(path)
            path = data_path.iloc[1]
            model_result_2 = read_model_result(path)
        elif tag == 'GPU = 2':
            path = '/Users/dongzhejiang/Downloads/all_Log/resnet/cifar1010887.log'
            model_result_1 = read_model_result_1(path)
            path = '/Users/dongzhejiang/Downloads/all_Log/resnet/cifar1010888.log'
            model_result_2 = read_model_result_1(path)
        model_result = (model_result_1 + model_result_2) / 2
        min_time = model_result['timestamp'].min()
        x = (model_result['timestamp'] - min_time) / 60
        y = model_result['test_acc']
        plt.plot(x, y, label=tag)
        plot_xy(x.max(), y.max(), 0, ax)
    plt.ylabel('accuracy', fontsize=12)
    plt.xlabel('time(min)', fontsize=12)
    plt.tick_params(labelsize=12)
    plt.title(model, fontproperties=zhfont1, fontsize=20)
    plt.xlim(-0.5, )
    plt.ylim([0, 1])
    plt.legend(fontsize=12)
    plt.savefig('/Users/dongzhejiang/Downloads/resnet_acc.png', dpi=300)
    plt.show()

    #cnn accuracy
    fig, ax = plt.subplots()
    model = 'cnn'
    temp_data = data[data['models'] == model]
    for tag in ['GPU = 0', 'GPU = 1', 'GPU = 2','GPU = 4']:
        if tag == 'GPU = 0':
            data_path = temp_data[temp_data['cpus'] == 1]['path']
            path = data_path.iloc[0]
            model_result_1 = read_model_result(path)
            path = data_path.iloc[1]
            model_result_2 = read_model_result(path)
            # continue
        elif tag == 'GPU = 1':
            data_path = temp_data[temp_data['gpus'] == 1]['path']
            path = data_path.iloc[0]
            model_result_1 = read_model_result(path)
            path = data_path.iloc[1]
            model_result_2 = read_model_result(path)
        elif tag == 'GPU = 2':
            path = '/Users/dongzhejiang/Downloads/all_Log/cnn/cifar1038112.log'
            model_result_1 = read_model_result_1(path)
            path = '/Users/dongzhejiang/Downloads/all_Log/cnn/cifar1038114.log'
            model_result_2 = read_model_result_1(path)
        elif tag == 'GPU = 4':
            data_path = temp_data[(temp_data['gpus'] == 4) & (temp_data['cpus']==0)]['path']
            path = data_path.iloc[0]
            model_result_1 = read_model_result(path)
            path = data_path.iloc[1]
            model_result_2 = read_model_result(path)
        model_result = (model_result_1 + model_result_2) / 2
        min_time = model_result['timestamp'].min()
        x = (model_result['timestamp'] - min_time) / 60
        y = model_result['test_acc']
        plt.plot(x, y, label=tag)
        plot_xy(x.max(), y.max(), 0, ax)

        # break
    plt.ylabel('accuracy', fontsize=12)
    plt.xlabel('time(min)', fontsize=12)
    plt.tick_params(labelsize=12)
    plt.title(model, fontproperties=zhfont1, fontsize=20)
    plt.xlim(0, )
    plt.ylim([0, 1])
    plt.legend(fontsize=12)
    plt.savefig('/Users/dongzhejiang/Downloads/cnn_acc.png', dpi=300)
    plt.show()

    #mlp accuracy
    fig, ax = plt.subplots()
    model='mlp'
    temp_data = data[data['models'] == model]
    for tag in ['GPU = 0', 'GPU = 1', 'GPU = 2']:
        if tag == 'GPU = 0':
            data_path = temp_data[temp_data['cpus'] == 1]['path']
            path = data_path.iloc[0]
            model_result_1 = read_model_result(path)
            path = data_path.iloc[1]
            model_result_2 = read_model_result(path)
            # continue
        elif tag == 'GPU = 1':
            data_path = temp_data[temp_data['gpus'] == 1]['path']
            path = data_path.iloc[0]
            model_result_1 = read_model_result(path)
            path = data_path.iloc[1]
            model_result_2 = read_model_result(path)
        elif tag == 'GPU = 2':
            path='/Users/dongzhejiang/Downloads/all_Log/mlp/cifar1021625.log'
            model_result_1 = read_model_result_1(path)
            path = '/Users/dongzhejiang/Downloads/all_Log/mlp/cifar1021627.log'
            model_result_2 = read_model_result_1(path)
        model_result = (model_result_1 + model_result_2) / 2
        min_time = model_result['timestamp'].min()
        x = (model_result['timestamp'] - min_time) / 60
        y = model_result['test_acc']
        plt.plot(x, y, label=tag)
        plot_xy(x.max(), y.max(), 0, ax)
    plt.ylabel('accuracy', fontsize=12)
    plt.xlabel('time(min)', fontsize=12)
    plt.tick_params(labelsize=12)
    plt.title(model, fontproperties=zhfont1, fontsize=20)
    plt.xlim(0, )
    plt.ylim([0, 1])
    plt.legend(fontsize=12)
    plt.savefig('/Users/dongzhejiang/Downloads/mlp_acc.png', dpi=300)
    plt.show()

    # 画accuracy图
    for model in ['resnet','mlp','cnn']:
        fig, ax = plt.subplots()
        # model = 'resnet'
        temp_data=data[data['models']==model]
        for tag in ['GPU = 0','GPU = 1','GPU = 2']:
            if tag=='GPU = 0':
                data_path=temp_data[temp_data['cpus']==1]['path']
            elif tag=='GPU = 1':
                data_path=temp_data[temp_data['gpus']==1]['path']
            elif tag=='GPU = 2':
                data_path=temp_data[temp_data['gpus']==2]['path']
            if len(data_path)==0:
                continue
            path=data_path.iloc[0]
            model_result_1 = read_model_result(path)
            path=data_path.iloc[1]
            model_result_2 = read_model_result(path)
            model_result = (model_result_1 + model_result_2) / 2
            min_time = model_result['timestamp'].min()
            x = (model_result['timestamp'] - min_time)/60
            y = model_result['test_acc']
            plt.plot(x, y, label=tag)
            plot_xy(x.max(), y.max(), 0, ax)

            # break
        plt.ylabel('accuracy', fontsize=18)
        plt.xlabel('time(min)', fontsize=18)
        plt.tick_params(labelsize=18)
        plt.title(model, fontproperties=zhfont1, fontsize=20)
        plt.xlim(-0.5, )
        plt.ylim([0, 1])
        plt.legend(fontsize=12)
        plt.show()

    # resnet cpu
    path='/Users/dongzhejiang/Downloads/Log_2/all_process_1546666945.log'
    cpus,gpus=read_resource(path)
    model='resnet'
    plt.rcParams['figure.figsize'] = (6.0, 4.0)
    fig, ax = plt.subplots()
    temp_data = data[data['models'] == model]
    cpu_boxplt = pd.DataFrame()
    for tag in [ 'GPU = 1', 'GPU = 2','GPU = 4']:
        if tag == 'GPU = 1':
            pid = temp_data[temp_data['gpus'] == 1]['pid']
            cpu = cpus[cpus['pid'].isin(pid)]
        elif tag == 'GPU = 2':
            path = '/Users/dongzhejiang/Downloads/all_Log/resnet/pid10887.log'
            cpu_1, gpu_1 = read_resource_1(path)
            path = '/Users/dongzhejiang/Downloads/all_Log/resnet/pid10888.log'
            cpu_2, gpu_2 = read_resource_1(path)
            cpu = pd.concat([cpu_1, cpu_2],ignore_index=True)
            cpu = cpu.sort_values(by=['timestamp'])
        elif tag == 'GPU = 4':
            path = '/Users/dongzhejiang/Downloads/all_Log/resnet/pid362.log'
            cpu_1, gpu_1 = read_resource_1(path)
            path = '/Users/dongzhejiang/Downloads/all_Log/resnet/pid365.log'
            cpu_2, gpu_2 = read_resource_1(path)
            cpu = pd.concat([cpu_1, cpu_2],ignore_index=True)
            cpu = cpu.sort_values(by=['timestamp'])
        cpu[tag] = cpu['p_memo_perc'] / 4000
        cpu_boxplt=pd.concat([cpu_boxplt,cpu[tag]],axis=1)
        # cpu_boxplt[tag] = cpu['p_cpu_per']

    cpu_boxplt.boxplot()
    plt.ylabel('cpu percent', fontsize=12)
    plt.title(model, fontproperties=zhfont1, fontsize=16)
    plt.tick_params(labelsize=10)
    plt.savefig('/Users/dongzhejiang/Downloads/resnet_cpu.png', dpi=300)
    plt.show()

    # cnn cpu
    model = 'cnn'
    plt.rcParams['figure.figsize'] = (5.0,4.0)
    fig, ax = plt.subplots()
    temp_data = data[data['models'] == model]
    cpu_boxplt = pd.DataFrame()
    for tag in ['GPU = 0', 'GPU = 1', 'GPU = 2','GPU = 4']:
        if tag == 'GPU = 0':
            pid = temp_data[temp_data['cpus'] == 1]['pid']
            cpu = cpus[cpus['pid'].isin(pid)]
        elif tag == 'GPU = 1':
            pid = temp_data[temp_data['gpus'] == 1]['pid']
            cpu = cpus[cpus['pid'].isin(pid)]
        elif tag == 'GPU = 2':
            path = '/Users/dongzhejiang/Downloads/all_Log/cnn/pid38112.log'
            cpu_1, gpu_1 = read_resource_1(path)
            path = '/Users/dongzhejiang/Downloads/all_Log/cnn/pid38114.log'
            cpu_2, gpu_2 = read_resource_1(path)
            cpu = pd.concat([cpu_1, cpu_2], ignore_index=True)
            cpu = cpu.sort_values(by=['timestamp'])
        elif tag == 'GPU = 4':
            pid = temp_data[(temp_data['gpus'] == 4) & (temp_data['cpus'] == 0)]['pid']
            cpu = cpus[cpus['pid'].isin(pid)]
        cpu[tag] = cpu['p_memo_perc'] / 4000
        cpu_boxplt = pd.concat([cpu_boxplt, cpu[tag]], axis=1)
        # cpu_boxplt[tag] = cpu['p_cpu_per']

    cpu_boxplt.boxplot()
    plt.ylabel('cpu percent', fontsize=12)
    plt.title(model, fontproperties=zhfont1, fontsize=20)
    plt.tick_params(labelsize=10)
    plt.savefig('/Users/dongzhejiang/Downloads/cnn_cpu.png', dpi=300)
    plt.show()

    # mlp cpu
    model = 'mlp'
    fig, ax = plt.subplots()
    temp_data = data[data['models'] == model]
    cpu_boxplt = pd.DataFrame()
    for tag in ['GPU = 0', 'GPU = 1', 'GPU = 2']:
        if tag == 'GPU = 0':
            pid = temp_data[temp_data['cpus'] == 1]['pid']
            cpu = cpus[cpus['pid'].isin(pid)]
        elif tag == 'GPU = 1':
            pid = temp_data[temp_data['gpus'] == 1]['pid']
            cpu = cpus[cpus['pid'].isin(pid)]
        elif tag == 'GPU = 2':
            path = '/Users/dongzhejiang/Downloads/all_Log/mlp/pid21625.log'
            cpu_1, gpu_1 = read_resource_1(path)
            path = '/Users/dongzhejiang/Downloads/all_Log/mlp//pid21627.log'
            cpu_2, gpu_2 = read_resource_1(path)
            cpu = pd.concat([cpu_1, cpu_2], ignore_index=True)
            cpu = cpu.sort_values(by=['timestamp'])

        cpu[tag] = cpu['p_memo_perc'] / 4000
        cpu_boxplt = pd.concat([cpu_boxplt, cpu[tag]], axis=1)
        # cpu_boxplt[tag] = cpu['p_cpu_per']
    cpu_boxplt.boxplot()
    plt.ylabel('cpu percent', fontsize=12)
    plt.title(model, fontproperties=zhfont1, fontsize=20)
    plt.tick_params(labelsize=10)
    plt.savefig('/Users/dongzhejiang/Downloads/mlp_cpu.png', dpi=300)
    plt.show()


    # 画cpu箱线图
    fig, ax = plt.subplots()
    cpu_boxplt=pd.DataFrame()
    path='/Users/dongzhejiang/Downloads/all_Log/resnet/pid10887.log'
    cpu_1,gpu_1=read_resource_1(path)
    path = '/Users/dongzhejiang/Downloads/all_Log/resnet/pid10888.log'
    cpu_2, gpu_2 = read_resource_1(path)
    cpu=pd.concat([cpu_1,cpu_2])
    cpu=cpu.sort_values(by=['timestamp'])
    cpu['p_cpu_per']=cpu['p_cpu_per']/4000
    cpu_boxplt['GPU=2']=cpu['p_cpu_per'].copy()
    cpu_boxplt.boxplot()
    plt.ylabel('cpu percent', fontsize=18)
    plt.title('resnet', fontproperties=zhfont1, fontsize=20)
    plt.tick_params(labelsize=18)
    plt.show()

    path='/Users/dongzhejiang/Downloads/Log_1/all_process.log'
    cpus,gpus=read_resource(path)
    for model in ['resnet','mlp','cnn']:
        fig, ax = plt.subplots()
        temp_data=data[data['models']==model]
        cpu_boxplt = pd.DataFrame()
        for tag in ['GPU = 0','GPU = 1','GPU = 2']:
            if tag=='GPU = 0':
                pid=temp_data[temp_data['cpus']==1]['pid']
            elif tag=='GPU = 1':
                pid=temp_data[temp_data['gpus']==1]['pid']
            elif tag=='GPU = 2':
                pid=temp_data[temp_data['gpus']==2]['pid']
            if len(pid)==0:
                continue
            cpu=cpus[cpus['pid'].isin(pid)]
            cpu['p_cpu_per'] = cpu['p_memo_perc'] / 4000
            cpu_boxplt[tag] = cpu['p_cpu_per'].copy()

        cpu_boxplt.boxplot()
        plt.ylabel('cpu percent', fontsize=18)
        plt.title(model, fontproperties=zhfont1, fontsize=20)
        plt.tick_params(labelsize=18)
        plt.show()

    #resnet gpu
    fig, ax = plt.subplots()
    model='resnet'
    temp_data = data[data['models'] == model]
    gpu_boxplot = pd.DataFrame()
    for tag in [ 'GPU = 1', 'GPU = 2']:
        if tag == 'GPU = 1':
            pid = temp_data[temp_data['gpus'] == 1]['pid']
            cpu = cpus[cpus['pid'].isin(pid)]
            min_time = cpu['timestamp'].min()
            max_time = cpu['timestamp'].max()
            gpu = gpus[(gpus['timestamp'] <= max_time) & (gpus['timestamp'] >= min_time)]
            gpu['GPU = 1']=gpu['gpu_3'].copy()
            gpu_boxplot = pd.concat([gpu_boxplot, gpu['GPU = 1']], axis=1)
        elif tag == 'GPU = 2':
            path = '/Users/dongzhejiang/Downloads/all_Log/resnet/pid10887.log'
            cpu_1, gpu_1 = read_resource_1(path)
            path = '/Users/dongzhejiang/Downloads/all_Log/resnet/pid10888.log'
            cpu_2, gpu_2 = read_resource_1(path)
            gpu = pd.concat([gpu_1, gpu_2], ignore_index=True)
            gpu['GPU = 2 (1)']= gpu['gpu_1'].copy()
            gpu_boxplot=pd.concat([gpu_boxplot,gpu['GPU = 2 (1)']],axis=1)
            gpu['GPU = 2 (2)']=gpu['gpu_2'].copy()
            gpu_boxplot = pd.concat([gpu_boxplot, gpu['GPU = 2 (2)']], axis=1)
    gpu_boxplot.boxplot()
    plt.ylabel('gpu percent', fontsize=12)
    plt.title(model, fontproperties=zhfont1, fontsize=20)
    plt.tick_params(labelsize=12)
    plt.savefig('/Users/dongzhejiang/Downloads/resnet_gpu.png', dpi=300)
    plt.show()

    #cnn gpu
    model='cnn'
    fig, ax = plt.subplots()
    temp_data = data[data['models'] == model]
    gpu_boxplot = pd.DataFrame()
    for tag in ['GPU = 1', 'GPU = 2']:
        if tag == 'GPU = 1':
            pid = temp_data[temp_data['gpus'] == 1]['pid']
            if len(pid) == 0:
                continue
            cpu = cpus[cpus['pid'].isin(pid)]
            min_time = cpu['timestamp'].min()
            max_time = cpu['timestamp'].max()
            gpu = gpus[(gpus['timestamp'] <= max_time) & (gpus['timestamp'] >= min_time)]
            gpu['GPU = 1'] = gpu['gpu_3'].copy()
            gpu_boxplot = pd.concat([gpu_boxplot, gpu['GPU = 1']], axis=1)
        elif tag == 'GPU = 2':
            path = '/Users/dongzhejiang/Downloads/all_Log/cnn/pid38112.log'
            cpu_1, gpu_1 = read_resource_1(path)
            path = '/Users/dongzhejiang/Downloads/all_Log/cnn/pid38114.log'
            cpu_2, gpu_2 = read_resource_1(path)
            gpu = pd.concat([gpu_1, gpu_2], ignore_index=True)
            gpu['GPU = 2 (1)'] = gpu['gpu_1'].copy()
            gpu_boxplot = pd.concat([gpu_boxplot, gpu['GPU = 2 (1)']], axis=1)
            gpu['GPU = 2 (2)'] = gpu['gpu_2'].copy()
            gpu_boxplot = pd.concat([gpu_boxplot, gpu['GPU = 2 (2)']], axis=1)
    gpu_boxplot.boxplot()
    plt.ylabel('gpu percent', fontsize=12)
    plt.title(model, fontproperties=zhfont1, fontsize=20)
    plt.tick_params(labelsize=12)
    plt.savefig('/Users/dongzhejiang/Downloads/cnn_gpu.png', dpi=300)
    plt.show()

    # mlp gpu
    fig, ax = plt.subplots()
    model = 'mlp'
    temp_data = data[data['models'] == model]
    gpu_boxplot = pd.DataFrame()
    for tag in ['GPU = 1', 'GPU = 2']:
        if tag == 'GPU = 1':
            pid = temp_data[temp_data['gpus'] == 1]['pid']
            cpu = cpus[cpus['pid'].isin(pid)]
            min_time = cpu['timestamp'].min()
            max_time = cpu['timestamp'].max()
            gpu = gpus[(gpus['timestamp'] <= max_time) & (gpus['timestamp'] >= min_time)]
            gpu['GPU = 1'] = gpu['gpu_3'].copy()
            gpu_boxplot = pd.concat([gpu_boxplot, gpu['GPU = 1']], axis=1)
        elif tag == 'GPU = 2':
            path = '/Users/dongzhejiang/Downloads/all_Log/mlp/pid21627.log'
            cpu_1, gpu_1 = read_resource_1(path)
            path = '/Users/dongzhejiang/Downloads/all_Log/mlp/pid21625.log'
            cpu_2, gpu_2 = read_resource_1(path)
            gpu = pd.concat([gpu_1, gpu_2], ignore_index=True)
            gpu['GPU = 2 (1)'] = gpu['gpu_1'].copy()
            gpu_boxplot = pd.concat([gpu_boxplot, gpu['GPU = 2 (1)']], axis=1)
            gpu['GPU = 2 (2)'] = gpu['gpu_2'].copy()
            gpu_boxplot = pd.concat([gpu_boxplot, gpu['GPU = 2 (2)']], axis=1)
    gpu_boxplot.boxplot()
    plt.ylabel('gpu percent', fontsize=12)
    plt.title(model, fontproperties=zhfont1, fontsize=20)
    plt.tick_params(labelsize=12)
    plt.savefig('/Users/dongzhejiang/Downloads/mlp_gpu.png', dpi=300)
    plt.show()

    #画gpu箱线图
    for model in ['resnet', 'mlp', 'cnn']:
        fig, ax = plt.subplots()
        temp_data = data[data['models'] == model]
        gpu_boxplot = pd.DataFrame()
        for tag in ['GPU = 0', 'GPU = 1', 'GPU = 2']:
            if tag == 'GPU = 1':
                pid = temp_data[temp_data['gpus'] == 1]['pid']
                if len(pid) == 0:
                    continue
                cpu = cpus[cpus['pid'].isin(pid)]
                min_time = cpu['timestamp'].min()
                max_time = cpu['timestamp'].max()
                gpu = gpus[(gpus['timestamp'] <= max_time) & (gpus['timestamp'] >= min_time)]
                gpu_boxplot['GPU = 1'] = gpu['gpu_3'].copy()
            elif tag == 'GPU = 2':
                pid = temp_data[temp_data['gpus'] == 2]['pid']
                if len(pid) == 0:
                    continue
                cpu = cpus[cpus['pid'].isin(pid)]
                min_time = cpu['timestamp'].min()
                max_time = cpu['timestamp'].max()
                gpu = gpus[(gpus['timestamp'] <= max_time) & (gpus['timestamp'] >= min_time)]
                gpu_boxplot['GPU = 2 (1)'] = gpu['gpu_3'].copy()
                gpu_boxplot['GPU = 2 (2)'] = gpu['gpu_4'].copy()
        gpu_boxplot.boxplot()
        plt.ylabel('gpu percent', fontsize=18)
        plt.title(model, fontproperties=zhfont1, fontsize=20)
        plt.tick_params(labelsize=18)
        plt.show()

    # 同步异步
    #accuracy
    fig, ax = plt.subplots()
    model = 'cnn'
    #同步
    path = '/Users/dongzhejiang/Documents/learning/云计算/图/all_Log/cnn/cifar1038112.log'
    model_result_1 = read_model_result_1(path)
    path = '/Users/dongzhejiang/Documents/learning/云计算/图/all_Log/cnn/cifar1038114.log'
    model_result_2 = read_model_result_1(path)
    model_result = (model_result_1 + model_result_2) / 2
    min_time = model_result['timestamp'].min()
    x = (model_result['timestamp'] - min_time) / 60
    y = model_result['test_acc']
    plt.plot(x, y, label='sync')
    plot_xy(x.max(), y.max(), 0, ax)
    # 异步
    temp_data = data[data['models'] == model]
    data_path = temp_data[temp_data['sync'] == 0]['path']
    path = data_path.iloc[0]
    model_result_1 = read_model_result(path)
    path = data_path.iloc[1]
    model_result_2 = read_model_result(path)
    model_result = (model_result_1 + model_result_2) / 2
    min_time = model_result['timestamp'].min()
    x = (model_result['timestamp'] - min_time) / 60
    y = model_result['test_acc']
    plt.plot(x, y, label='async')
    plot_xy(x.max(), y.max(), 0, ax)
    plt.ylabel('accuracy', fontsize=12)
    plt.xlabel('time(min)', fontsize=12)
    plt.tick_params(labelsize=12)
    plt.title(model, fontproperties=zhfont1, fontsize=20)
    plt.xlim(0, )
    plt.ylim([0, 1])
    plt.legend(fontsize=12)
    plt.savefig('/Users/dongzhejiang/Downloads/sync_acc.png', dpi=300)
    plt.show()

    #cpu
    model = 'cnn'
    fig, ax = plt.subplots()
    temp_data = data[data['models'] == model]
    cpu_boxplt = pd.DataFrame()
    #异步
    pid = temp_data[temp_data['sync'] == 0]['pid']
    cpu = cpus[cpus['pid'].isin(pid)]
    cpu['async'] = cpu['p_memo_perc'] / 4000
    cpu_boxplt = pd.concat([cpu_boxplt, cpu['async']], axis=1)
    #同步
    path = '/Users/dongzhejiang/Downloads/all_Log/cnn/pid38112.log'
    cpu_1, gpu_1 = read_resource_1(path)
    path = '/Users/dongzhejiang/Downloads/all_Log/cnn/pid38114.log'
    cpu_2, gpu_2 = read_resource_1(path)
    cpu = pd.concat([cpu_1, cpu_2], ignore_index=True)
    cpu = cpu.sort_values(by=['timestamp'])
    cpu['sync'] = cpu['p_memo_perc'] / 4000
    cpu_boxplt = pd.concat([cpu_boxplt, cpu['sync']], axis=1)
    cpu_boxplt.boxplot()
    plt.ylabel('cpu percent', fontsize=12)
    plt.title(model, fontproperties=zhfont1, fontsize=20)
    plt.tick_params(labelsize=10)
    plt.savefig('/Users/dongzhejiang/Downloads/sync_cpu.png', dpi=300)
    plt.show()

    #gpu
    model = 'cnn'
    fig, ax = plt.subplots()
    # temp_data = data[data['models'] == model]
    gpu_boxplot = pd.DataFrame()
    #同步
    path = '/Users/dongzhejiang/PycharmProjects/my_mxnet/all_process_1546764850.log'
    cpus, gpus = read_resource(path)
    gpu=gpus.copy()
    gpu['sync(1)'] = gpu['gpu_5'].copy()
    gpu['sync(1)']=gpu['sync(1)']/100
    gpu_boxplot = pd.concat([gpu_boxplot, gpu['sync(1)']], axis=1)
    gpu['sync(2)'] = gpu['gpu_7'].copy()
    gpu['sync(2)']=gpu['sync(2)']/100
    gpu_boxplot = pd.concat([gpu_boxplot, gpu['sync(2)']], axis=1)

    #异步
    path='/Users/dongzhejiang/PycharmProjects/my_mxnet/all_process_1546765782.log'
    cpus, gpus = read_resource(path)
    gpu = gpus.copy()
    gpu['async(1)'] = gpu['gpu_6'].copy()
    gpu['async(1)'] = gpu['async(1)'] / 100
    gpu_boxplot = pd.concat([gpu_boxplot, gpu['async(1)']], axis=1)
    gpu['async(2)'] = gpu['gpu_0'].copy()
    gpu['async(2)'] = gpu['async(2)'] / 100
    gpu_boxplot = pd.concat([gpu_boxplot, gpu['async(2)']], axis=1)
    gpu_boxplot.boxplot()
    plt.ylabel('gpu percent', fontsize=12)
    plt.title(model, fontsize=20)
    plt.tick_params(labelsize=12)
    plt.savefig('/Users/dongzhejiang/Downloads/sync_gpu.png', dpi=300)
    plt.show()



    fig, ax = plt.subplots()
    gpu_boxplot=pd.DataFrame()
    gpu=pd.concat([gpu_1,gpu_2])
    gpu=gpu.sort_values(by=['timestamp'])
    gpu_boxplot['GPU = 2 (1)']=gpu['gpu_1'].copy()
    gpu_boxplot['GPU = 2 (2)']=gpu['gpu_2'].copy()
    gpu_boxplot.boxplot()
    plt.ylabel('gpu percent', fontsize=18)
    plt.title('resnet', fontproperties=zhfont1, fontsize=20)
    plt.tick_params(labelsize=18)
    plt.show()


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