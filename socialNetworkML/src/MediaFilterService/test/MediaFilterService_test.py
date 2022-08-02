import sys
import time
import uuid
from pathlib import Path

from thrift import Thrift
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket, TTransport

gen_py_dir = Path(__file__).resolve().parent.parent.parent.parent / 'gen-py'
sys.path.append(str(gen_py_dir))
import argparse
import base64
import io
import json
from pathlib import Path

from social_network import MediaFilterService


def main():
    socket = TSocket.TSocket(host='localhost', port=9090)
    transport = TTransport.TFramedTransport(socket)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = MediaFilterService.Client(protocol)

    image_path = Path(__file__).resolve().parent.parent / 'data' / '0cXfzlu.jpg'
    base64_path = Path(__file__).resolve().parent.parent / 'data' / '1.png'

    with open(image_path, 'rb') as f:
        base64_str_1 = base64.b64encode(f.read()).decode('ascii')

    with open(base64_path, 'r') as f:
        base64_str_2 = f.read()

    transport.open()
    req_id = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
    media_ids = [1, 2]
    media_types = ['jpg', 'jpg']
    media_data_list = [base64_str_1, base64_str_2]
    carrier = {}
    for _ in range(20):
        start = time.time()
        print(client.MediaFilter(req_id, media_ids, media_types, media_data_list, carrier))
        end = time.time()
        duration = end - start
        print('inference time = {0:.1f}ms'.format(duration * 1000))
    transport.close()


if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        print('%s' % tx.message)
