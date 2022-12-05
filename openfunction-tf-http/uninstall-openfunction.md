k get -n dapr-system deployments.apps |  awk '{print $1}' | xargs kubectl delete -n dapr-system deployments.apps

k get -n dapr-system pods |  awk '{print $1}' | xargs kubectl delete -n dapr-system pods

k get pod | awk '{print $1}' | xargs kubectl delete pod

kubectl get namespace keda  -o json | tr -d "\n" | sed "s/\"finalizers\": \[[^]]\+\]/\"finalizers\": []/"  | kubectl replace --raw /api/v1/namespaces/kube-node-lease/finalize -f - 

kubectl api-resources --verbs=list --namespaced -o name | xargs -n 1 kubectl get --show-kind --ignore-not-found -n openfunction

kubectl api-resources --verbs=list --namespaced -o name | xargs -n 1 kubectl get --show-kind  --ignore-not-found -n openfunction

cat <<EOF | curl -X PUT \
  localhost:8080/api/v1/namespaces/test/finalize \
  -H "Content-Type: application/json" \
  --data-binary @-
{
  "kind": "Namespace",
  "apiVersion": "v1",
  "metadata": {
    "name": "openfunction"
  },
  "spec": {
    "finalizers": null
  }
}
EOF

kubectl delete -f - <<EOF
  kind: Gateway
  apiVersion: gateway.networking.k8s.io/v1alpha2
  metadata:
    name: contour
    namespace: projectcontour
  spec:
    gatewayClassName: contour
    listeners:
      - name: http
        protocol: HTTP
        port: 80
        allowedRoutes:
          namespaces:
            from: All
EOF

kubectl delete -f - <<EOF
  kind: GatewayClass
  apiVersion: gateway.networking.k8s.io/v1alpha2
  metadata:
    name: contour
  spec:
    controllerName: projectcontour.io/gateway-controller
EOF

kubectl delete -f https://projectcontour.io/quickstart/contour-gateway-provisioner.yaml

 kubectl delete -f https://github.com/knative/serving/releases/download/knative-v1.3.2/serving-default-domain.yaml

 kubectl delete -f https://github.com/knative/net-kourier/releases/download/knative-v1.3.0/kourier.yaml

  kubectl delete -f https://github.com/knative/serving/releases/download/knative-v1.3.2/serving-core.yaml

  kubectl delete -f https://github.com/knative/serving/releases/download/knative-v1.3.2/serving-crds.yaml

  kubectl delete --filename https://github.com/shipwright-io/build/releases/download/v0.6.0/release.yaml

  kubectl delete --filename https://github.com/tektoncd/pipeline/releases/download/v0.28.1/release.yaml

  
  kubectl delete --filename https://github.com/jetstack/cert-manager/releases/download/v1.5.4/cert-manager.yaml


 kubectl delete namespace hotel-res --force --grace-period=0

 kubectl get ns hotel-res -o json | jq '.spec.finalizers=[]' > ns-without-finalizers.json