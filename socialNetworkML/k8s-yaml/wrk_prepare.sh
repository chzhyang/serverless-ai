SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR=$SCRIPT_DIR/..
# init data
sudo apt -y install python3
sudo apt -y install python3-pip
pip3 install asyncio && pip3 install aiohttp
# check: kubectl get pods -o wide -A |grep social-network  |grep -v Running
notready=$(kubectl -n deathstarbench-social-network  get pods | grep -v 'NAME' | grep -v 'Running')
while [ "x$notready" != "x" ]; do
    echo 'Scaling, sleep 1'
    sleep 1
    notready=$(kubectl -n deathstarbench-social-network  get pods | grep -v 'NAME' | grep -v 'Running')
done
echo 'Scaling done!'

NGINX_ADDR="$(kubectl get svc -n deathstarbench-social-network|grep nginx-thrift|awk '{print $3}')"

python3  $BASE_DIR/scripts/init_social_graph.py --ip http://${NGINX_ADDR} --port 8080 --graph $BASE_DIR/datasets/social-graph/socfb-Reed98/socfb-Reed98.mtx

# max open files
setOpenfiles=$(sudo grep 'soft nofile 1048576' /etc/security/limits.conf)
if [[ $setOpenfiles == '' ]]
then
    sudo bash -c 'echo  "* hard nofile 1048576" >>/etc/security/limits.conf'
    sudo bash -c 'echo  "* soft nofile 1048576" >>/etc/security/limits.conf'
fi

# install wrk
cd $BASE_DIR/wrk2/
if [ -e wrk ]; then
    echo "wrk is there"
else
    sudo apt-get -y update && \
        sudo apt-get -y install dnsutils python3 python3-pip python3-aiohttp libssl-dev libz-dev luarocks iputils-ping lynx build-essential gcc bash curl wget vim && \
        pip3 install asyncio && \
        pip3 install aiohttp && \
        sudo luarocks install luasocket && make clean && make
fi
cd -

