

## setup
- copy .env.example to .env
  ```bash
  $ cp .env.example .env
  ```
  - If you want to enable GPUs, set the .env file as follows
    ```bash
    TARGET_ARCH=gpu
    RUNTIME=nvidia
    TARGET_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
    ```
  - If you use the CPU, set the .env file as follows
    ```bash
    TARGET_ARCH=cpu
    ```

- build docker image
  ```bash
  $ docker-compose build
  ```

- run docker container
  ```bash
  $ docker-compose up -d
  ```

## run
- exec learning
  ```bash
  $ docker-compose exec rl_adjust_pose bash
  $ python3 src/learning.py
  ```

- exec tensorboard
  ```
  $ docker exec rl_adjust_pose tensorboard --logdir logs
  ```
  - http://localhost:6006/


## note
- If X forwading is failing, try the following command on the host side.
  ```
  $ xhost +local:
  ```
