VPATH="/csha"
# UI permisions
xhost +si:localuser:root
CONTAINER="hacs2"
IMAGE="csha/cs-od"
DISPLAY=1
docker run \
            -ti \
            --gpus all \
            --ipc=host \
            --name $CONTAINER \
            --env="DISPLAY" \
            --env="QT_X11_NO_MITSHM=1" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            --volume="$HOME/.Xauthority:/home/$USER/.Xauthority:rw" \
            --volume="${PWD}/..:$VPATH/csod1" \
            --volume="/nas_data/EPL/csha:$VPATH/dataset" \
            --network host \
            $IMAGE:latest
                        # -p 52438:8888 \
            # -p 52436:6006 \
#            --network host  #DONT USE WITH OPT PORT '-p' \
#xhost -local:root  # resetting permissions
# --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
            # --volume="/data/rvi/dataset/GOPRO/GOPRO:$VPATH/GOPRO" \
