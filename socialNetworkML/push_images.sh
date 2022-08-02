#!/bin/bash

DOCKER_REGISTRY="${DOCKER_REGISTRY:-"harbor.harbor.svc.service.wpax.intel.com/dev/zzhou"}"

docker push ${DOCKER_REGISTRY}/media-frontend:xenial
docker push ${DOCKER_REGISTRY}/openresty-thrift-ml:xenial
docker push ${DOCKER_REGISTRY}/thrift-microservice-deps:xenial

docker push ${DOCKER_REGISTRY}/jaegertracing-all-in-one:latest
docker push ${DOCKER_REGISTRY}/memcached:1.6.7
docker push ${DOCKER_REGISTRY}/mongo:4.4.6
docker push ${DOCKER_REGISTRY}/redis:6.2.4

docker push ${DOCKER_REGISTRY}/social-network-ml-microservices:latest
docker push ${DOCKER_REGISTRY}/social-network-text-filter-service:latest
docker push ${DOCKER_REGISTRY}/social-network-media-filter-service:latest
docker push ${DOCKER_REGISTRY}/social-network-text-filter-service-opt-avx2:latest
docker push ${DOCKER_REGISTRY}/social-network-media-filter-service-opt-avx2:latest
docker push ${DOCKER_REGISTRY}/social-network-text-filter-service-opt-avx512:latest
docker push ${DOCKER_REGISTRY}/social-network-media-filter-service-opt-avx512:latest

docker push ${DOCKER_REGISTRY}/social-network-ml-client:latest
docker push ${DOCKER_REGISTRY}/locust-ml:latest
