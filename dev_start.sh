#!/bin/bash
WORKDIR=/ToT
CONTAINER=tree-of-thoughts-dev-container
IMAGE=tree-of-thoughts:dev
sudo docker stop $(sudo docker ps -a -q)  #stop停止所有容器
sudo docker rm $(sudo docker ps -a -q)  #remove删除所有容器
sudo docker run  --name $CONTAINER -it -v $PWD:$WORKDIR -w $WORKDIR $IMAGE /bin/bash
