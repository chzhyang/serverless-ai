SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $SCRIPT_DIR

NGINX_ADDR=`kubectl get svc -n deathstarbench-social-network|grep nginx-thrift|awk '{print $3}'`
echo $NGINX_ADDR
