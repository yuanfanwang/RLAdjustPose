ARG TARGET_IMAGE
FROM ${TARGET_IMAGE}

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=rl

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev sudo x11-apps && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt /app

RUN groupadd --gid=${GROUP_ID} ${USER_NAME}  && useradd -l -r --gid=${GROUP_ID} --uid=${USER_ID} ${USER_NAME} \
    &&  echo ${USER_NAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER_NAME} \
    &&  chmod 0440 /etc/sudoers.d/${USER_NAME}

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . /app
RUN chown -R ${USER_ID}:${GROUP_ID} /app
USER ${USER_NAME}

CMD ["tail", "-f", "/dev/null"]