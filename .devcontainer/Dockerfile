FROM quay.io/jupyter/pyspark-notebook:latest

COPY environment.yml /tmp/environment.yml

USER root

WORKDIR /home/jovyan/work

RUN conda env update --prune -f /tmp/environment.yml

RUN conda clean -afy

EXPOSE 8888 4040 8080 8081
