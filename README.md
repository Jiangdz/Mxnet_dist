# Mxnet_dist


## Step 1
Download https://github.com/apache/incubator-mxnet

## Step 2
No passwd login

```
ssh-copy-id localhost

```

## Step 3
Install requirements
```
pip install mxnet
pip install gluoncv
```


## Step 4
Create a file "host" and the content of the file is ips (one ip per line)

For example:
```
127.0.0.1
```

## Step 5
Run command
```
python incubator-mxnet/tools/launch.py -n 1 -s 1 -H host  --launcher ssh "export DMLC_PS_ROOT_URI=10.0.0.1 ;  python cifar10_dist.py"
```


