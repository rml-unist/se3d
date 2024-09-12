# chmod +x build_image.sh

IMAGE="csha/cs-od"

docker build --pull=true -t $IMAGE:latest .

# docker build -t $IMAGE:latest .