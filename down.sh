#!/bin/bash

sudo docker compose \
    -f compose.yaml \
    -f compose.gpu.yaml \
    down
