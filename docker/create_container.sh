#!/bin/bash

CURRENT_USER=$(whoami)
CONTAINER_NAME=$1
ID=`id -u`

if [ -z "$CONTAINER_NAME" ]; then
    CONTAINER_NAME=$CURRENT_USER"_roll_dev"
    echo "No container name specified; using the default name: $CONTAINER_NAME"
fi

echo "current user: $CURRENT_USER"

sudo docker run -dit \
     --cap-add SYS_ADMIN \
     --gpus all \
     --name=$CONTAINER_NAME \
     --ipc=host \
     --net=host \
     -v /dev/fuse:/dev/fuse \
     -v /dev/shm:/dev/shm \
     -v /home/$CURRENT_USER/:/home/$CURRENT_USER/ \
     -v /var/run/docker.sock:/var/run/docker.sock \
     -v /mnt/ram:/mnt/ram \
     roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-25.06-py3-torch280-vllm0110 \
     /bin/bash

sudo docker exec -i $CONTAINER_NAME groupadd sdev
sudo docker exec -i $CONTAINER_NAME /usr/sbin/useradd -MU -G sdev -u $ID $CURRENT_USER
sudo docker exec -it $CONTAINER_NAME /usr/bin/su $CURRENT_USER
