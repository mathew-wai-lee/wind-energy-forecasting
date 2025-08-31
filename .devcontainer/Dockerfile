FROM apache/zeppelin:0.12.0

ENV SPARK_VERSION=4.0.0
ENV HADOOP_VERSION=3

USER root
RUN apt-get update && apt-get install -y curl unzip wget grep sed vim tzdata && apt-get clean

USER root
RUN apt-get update && apt-get install -y openjdk-17-jdk && apt-get clean
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# auto upload zeppelin interpreter lib
RUN rm -rf /zeppelin

RUN rm -rf /spark
# RUN wget https://www-us.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
RUN wget https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
RUN tar zxvf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
RUN mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark
RUN rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# docker build -t zep_spark .devcontainer --platform linux/amd64
# docker run -u $(id -u) -p 8080:8080 --rm -v ./notebooks:/notebooks  --platform linux/amd64 \
# -e SPARK_HOME=/opt/spark  -e ZEPPELIN_NOTEBOOK_DIR='/notebooks' --name zeppelin zep_spark:latest