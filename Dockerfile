FROM ubuntu:20.04 as base

ENV TZ=Europe/Minsk
ARG DEBIAN_FRONTEND=noninteractive

# install sudo
RUN apt-get update && apt install -y sudo

# install dependencies
RUN apt-get install -y gcc libc6-dev gfortran libgfortran5 libsuitesparse-dev

# install python packages and update pip
RUN apt-get install -y python3-pip python3-tk python3-venv
RUN python3 -m pip install --upgrade pip


# install things that aren't in conda
COPY ./requirements.txt requirements.txt

# install NF-iSAM
COPY ./setup.py /app/setup.py

RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer

USER developer

# set up virtual environment
RUN python3 -m venv /venv
RUN /venv/bin/pip install --no-cache-dir numpy==1.19.5
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]
