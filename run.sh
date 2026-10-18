#!/bin/bash

if [ "$1" = "--cpu" ]; then
    sudo env \
        RAG_DEVICE=cpu \
        MLOPS_DEVICE=cpu \
        AUTOCOMPLETE_DEVICE=cpu \
        docker compose -f compose.yaml up -d --build --force-recreate
else
    sudo docker compose \
        -f compose.yaml \
        -f compose.gpu.yaml \
        up -d --build --force-recreate
fi
