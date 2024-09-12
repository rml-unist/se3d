CONTAINER="hacs2"

# UI permisions
xhost +si:localuser:root

docker start $CONTAINER
# Git pull orbslam and compile
docker exec -it $CONTAINER /bin/bash
