#!/bin/bash

# get replica
function usage() {
  echo "scale.sh <replica>"
  exit 1
}

replica=$1
[ "x$replica" == "x" ] && usage


# scale nginx
nginx_podAntiAffinity_patch=" \
spec:
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - nginx-thrift
            topologyKey: kubernetes.io/hostname
"

kubectl -n deathstarbench-social-network patch deployments.apps nginx-thrift --patch "$nginx_podAntiAffinity_patch"
# scheduel nginx-thrift to  each k8s node
nodes=$(kubectl get nodes | grep -v 'master' | grep 'Ready' | wc -l)

# not work if only --replicas=$nodes!!
kubectl  -n deathstarbench-social-network scale --replicas=0 deployment/nginx-thrift
nginxpod=$(kubectl get pods -n deathstarbench-social-network -o wide |grep nginx-thrift)
while [ "x$nginxpod" != "x" ]; do
    echo 'wait all nginx pod gone, sleep 1'
    echo $nginxpod
    sleep 1
    nginxpod=$(kubectl get pods -n deathstarbench-social-network  -o wide |grep nginx-thrift)
done
kubectl  -n deathstarbench-social-network scale --replicas=$nodes deployment/nginx-thrift
kubectl get pods -n deathstarbench-social-network -o wide |grep nginx-thrift
echo "Patched and scaled nginx-thrift"

# scale
srv="compose-post-service \
home-timeline-service \
media-service \
post-storage-service \
social-graph-service \
text-service \
unique-id-service \
url-shorten-service \
user-mention-service \
user-service \
user-timeline-service"


echo "Scale all services to $replica"
for s in ${srv}
do
kubectl -n deathstarbench-social-network scale --replicas=$replica deployment/$s
done

