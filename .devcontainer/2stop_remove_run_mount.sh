docker stop pyspark

docker rm pyspark

docker run \
    -p 8888:8888 -p 4040:4040 \
    --name pyspark \
    -v "$(pwd)/notebooks:/home/notebooks" \
    lee/pyspark \
    jupyter lab --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --allow-root

