import time
import zmq
from SecureConnection import root_server
from SecureConnection import server
from SecureConnection import monitor
import threading
import torch
import numpy as np
import heapq
import json
import os
from collections import deque
from util.model_card import available_models, ModelCard, retrieve_sending_dir, retrieve_sending_info, retrieve_file_cfg
from system_pipeline.onnx_backend.optimization import Optimizer

monitor_receive_interval = 5  # set intervals for receiving monitor info from clients
monitor_port = "34567"  # set server port to receive monitor info
TIMEOUT = 15  # Time to wait for new devices to connect to servers
MODEL_EXIST_ON_DEVICE = True  # set True if the model exists on the mobile device, will skip model creation and transmission
runtime_option = False  # set True if the load balance is runtime
split_size = 2
device_number = 2
task = "Generation"
root_dir = os.path.dirname(os.path.abspath(__file__))
residual_connection_option = True


# def round_robin_module_arrangement(num_devices, num_modules):
#     arrangement = [[0 for _ in range(num_modules)] for _ in range(num_devices)]
#     modules_per_device = num_modules // num_devices
#     extra_modules = num_modules % num_devices
#     start = 0
#     for i in range(num_devices):
#         end = start + modules_per_device + (1 if i < extra_modules else 0)
#         for j in range(start, end):
#             arrangement[i][j] = 1
#         start = end
#     return np.array(arrangement)
# Quntization_Option=True

# model_card = ModelCard('bloom3b', quantization_option=Quntization_Option, task_type=task,
#                        residual_connection=residual_connection_option, load_balancing_option=False,
#                        split_size=split_size)

# mem_util, out_size_map, bytearray_path, flop_module_path, num_flop, module_flop_map, num_modules \
#     = model_card.prepare_optimization_info()

# # initial_module_arrangement = round_robin_module_arrangement(device_number, split_size)
# # overlapping_module_arrangement = initial_module_arrangement  # Assuming no dynamic arrangement needed

# requested_model = 'bloom3b'
# to_send_path = retrieve_sending_dir(root_dir, requested_model, quantization_option=Quntization_Option,
#                                             residual_connection=residual_connection_option)
# initial_module_arrangement=[[1 ,1 ,1 ,1, 1, 1 ,1 ,1, 1, 1 ,1 ,1 ,1 ,1 ,1 ,1, 1 ,1 ,1 ,0 ,0 ,0 ,0 ,0, 0],
#                             [0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0 ,0 ,0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,1 ,1 ,1, 1 ,1, 1 ]]
# print("initial_module_arrangement")
# print(initial_module_arrangement)

# model_dirs = model_card.prepare_model_to_send(module_arrangement=initial_module_arrangement)

if __name__ == "__main__":
    start = time.time()
    context = zmq.Context()
    send = server.establish_connection(context, zmq.ROUTER, 234567)

    # start receiving the ips sent from android devices
    # once all ips are received, broadcast messages to all android devices saying all ip received
    # continue listen for whether there is new ip address arrives from android device
    devices = deque()
    ip_graph_requested = []  # Buffer to store addresses from devices
    last_received_time = time.time()
    continue_listening = False
    # requested_model = 'bloom560m'
    requested_model = 'bloom3b'
    #

    ##################################################################################
    ####################### 1. Devices-Server Connection Section #####################
    ##################################################################################
    while continue_listening:
        if send.poll(1000):
            print("start listening")
            identifier, action, msg_content = send.recv_multipart()
            print(f"action: {action.decode()}")
            print(f"msg_content: {msg_content.decode()}")
            print("message received")
            print("\n-----------------------------------\n")
            if action.decode() == "GET_IP_ADDRESSES":
                print("sending out ips...")
                ip_graph_requested.append(identifier)
            elif action.decode() == "RegisterIP":
                print("waiting for all device to be connected...")
                ip_graph_requested.append(identifier)
                jsonObject = json.loads(msg_content.decode())
                ip = jsonObject.get("ip")
                role = jsonObject.get("role")
                model_request = jsonObject.get("model", None)
                if role == "header":
                    devices.appendleft({"ip": ip, "role": role})
                    if model_request:
                        requested_model = model_request
                else:
                    devices.append({"ip": ip, "role": role})
                last_received_time = time.time()

        if time.time() - last_received_time > TIMEOUT:
            print("No new devices connected in the last", TIMEOUT, "seconds. Broadcasting message.")
            continue_listening = False
                                                    
    print(f'devices in root.py: {devices}')
    # requested_model = "opt125m"
    print(f'request model = {requested_model}')

    header_device = None
    tail_device = None

    tokenizer_dir = None
    session = []
    file_cfg = {}
    directory_path = None
    ip_graph = []
    ip_module = []

    ##################################################################################
    ####################### 2. Server-only Model Preparation Section #################
    ##################################################################################

    # read model request and conduct relevant model processing
    if requested_model:
        print(f'start preparing model: {requested_model}')
        if requested_model == "bloom3b":
            Quntization_Option = False
        if requested_model == "bloom3b-int8":
            Quntization_Option = True

        requested_model='bloom3b'
        to_send_path = retrieve_sending_dir(root_dir, requested_model, quantization_option=Quntization_Option,
                                            residual_connection=residual_connection_option)

        if   os.path.isdir(to_send_path):
            print('to_send dir exists')
            # Load the JSON string from the file
            with open(os.path.join(to_send_path, 'ip_module.json'), 'r') as file:
                ip_module_json = file.read()

            with open(os.path.join(to_send_path, 'session.json'), 'r') as file:
                session_index_json = file.read()

            ip_module = json.loads(ip_module_json)
            session = json.loads(session_index_json)
            file_cfg = retrieve_file_cfg(ip_module)

            # sending monitor initiation signal to all the devices
            for ip in ip_graph_requested:
                send.send_multipart([ip, b"False"])

        else:
            # sending monitor initiation signal to all the devices
            for ip in ip_graph_requested:
                send.send_multipart([ip, b"True"])

            # By default, we set "transformer_option" and "quantization option" to True
            # and we set "task_type" as "Generation" when creating ModelCard object.
            # we can set "task_type" as "Classification" if it's needed.
            
            model_card = ModelCard(requested_model, quantization_option=Quntization_Option, task_type=task,
                                   residual_connection=residual_connection_option, load_balancing_option=False,split_size=split_size)



            mem_util, out_size_map, bytearray_path, flop_module_path, num_flop, module_flop_map, num_modules \
                = model_card.prepare_optimization_info()



            tokenizer_dir = model_card.retreive_tokenizer_path()
            directory_path = os.path.dirname(bytearray_path)

            print(f'bytearray_path: {bytearray_path}')
            print(f'flop_module_path: {flop_module_path}')
            print(f'num_flop: {num_flop}')
            print(f'out_size_map: {out_size_map}')

            for ip in ip_graph_requested:
                send.send_multipart([ip, b"ready for monitor"])

            # # start monitor
            monitor = monitor.Monitor(monitor_receive_interval, monitor_port, devices, requested_model, \
                                      bytearray_path, flop_module_path, num_flop, runtime_option)
            thread = threading.Thread(target=monitor.start)
            thread.start()

            num_devices = len(devices)
            monitor.is_monitor_ready.wait()

            # 参数
            ping_latency, bandwidths, TotalMem, AvailMem, flop_speed = monitor.get_monitor_info()


            mem_threshold = .7  # set threshold for memory
            TotalMem = [m * mem_threshold for m in TotalMem]
            AvailMem = [m * mem_threshold for m in AvailMem]
            print("-----------------Test Optimizer Function----------------------")
            print("num_devices")
            print(num_devices)
            print("latency")
            print(ping_latency)
            print("bandwidth")
            print(bandwidths)
            print("totalMem")
            print(TotalMem)
            print("AvailMem")
            print(AvailMem)
            print("flop")
            print(flop_speed)

            if model_card.split_size:
                print("model_card.split_size: ", model_card.split_size)
                # load_balancer = Optimizer(num_devices=num_devices, num_modules=model_card.split_size)
                print("we use a round-robin approach")
            else:
                raise RuntimeError("The number of modules cannot be None! Check model_card.prepare_to_split().")
            # load_balancer.process_initial_info(num_flop=module_flop_map,
            #                                    flop_speed=flop_speed,
            #                                    ping_latency=ping_latency,
            #                                    bandwidths=bandwidths,
            #                                    m2m=out_size_map,
            #                                    model_size=mem_util,
            #                                    total_mem=TotalMem,
            #                                    ava_mem=AvailMem)
            # initial_module_arrangement = load_balancer.initial_module_arrangement()
            # overlapping_module_arrangement = load_balancer.dynamic_module_arrangement()
            def round_robin_module_arrangement(num_devices, num_modules):
                arrangement = [[0 for _ in range(num_modules)] for _ in range(num_devices)]
                modules_per_device = num_modules // num_devices
                extra_modules = num_modules % num_devices
                start = 0
                for i in range(num_devices):
                    end = start + modules_per_device + (1 if i < extra_modules else 0)
                    for j in range(start, end):
                        arrangement[i][j] = 1
                    start = end
                return np.array(arrangement)

            initial_module_arrangement = round_robin_module_arrangement(split_size, split_size)
            overlapping_module_arrangement = initial_module_arrangement  # Assuming no dynamic arrangement needed
            print("initial_module_arrangement")
            print(initial_module_arrangement)

            model_dirs = model_card.prepare_model_to_send(module_arrangement=initial_module_arrangement)
            device_module_order = model_card.device_module_arrangement  # [[0], [2], [1]]
            device_dir_map = {tuple(device_module_order[i]): model_dirs[i] for i in
                              range(len(model_dirs))}  # {(0): "..../model/.."}
            ip_device_module_map = {}
            for i in range(len(devices)):
                ip_device_module_map[devices[i]["ip"].encode("utf-8")] = device_module_order[
                    i]  # .26: [0], .19: [2], ..

            # retreive session for inference
            session = [str(j) for i in device_module_order for j in i]  # [0, 2, 1]

            # sort the order of ip graph for transmission
            ip_module_map = {}
            sorted_device_module_order = sorted(device_module_order)
            final_sorted_device_module = [[0]] * len(sorted_device_module_order)  # [[ip, [0]], [ip, [1]], [ip, [2]]]
            for ip, val in ip_device_module_map.items():
                if sorted_device_module_order.index(val) == 0:  # for header
                    final_sorted_device_module[0] = [ip, device_dir_map[tuple(val)]]
                elif sorted_device_module_order.index(val) != 0 and \
                        sorted_device_module_order.index(val) != len(sorted_device_module_order) - 1:
                    insert_index = sorted_device_module_order.index(val)
                    final_sorted_device_module[insert_index] = [ip, device_dir_map[tuple(val)]]
                else:  # for tailer
                    final_sorted_device_module[-1] = [ip, device_dir_map[tuple(val)]]

            print(f"session index: {session}")

            for d in range(len(final_sorted_device_module)):
                ip_encode = final_sorted_device_module[d][0]
                # current only retrieve single module path
                if final_sorted_device_module[d][1]:
                    print(f"{ip_encode}:{final_sorted_device_module[d][1][0]}")
                    file_cfg[ip_encode] = final_sorted_device_module[d][1][0]
                    ip_graph.append(ip_encode.decode("utf-8"))
                    ip_module.append([ip_encode.decode("utf-8"), file_cfg[ip_encode]])

            to_send_model_path = retrieve_sending_dir(root_dir, requested_model, quantization_option=Quntization_Option,
                                                      residual_connection=residual_connection_option)
            ip_module_json = json.dumps(ip_module)
            session_index_json = json.dumps(session)

            # Save the JSON string to a file
            with open(os.path.join(to_send_model_path, "ip_module.json"), 'w') as file:
                file.write(ip_module_json)

            with open(os.path.join(to_send_model_path, "session.json"), 'w') as file:
                file.write(session_index_json)

    else:
        raise RuntimeError("requested model cannot be None!")

    ##################################################################################
    ####################### 3. Sending models and tokenizer to devices ###############
    ##################################################################################
    print("------file_cfg--------")
    print(file_cfg)


    # 修改file_cfg json文件中的ip地址
    pathLists = []
    for index ,i in enumerate(devices):
        ip = i["ip"]
        role = i['role']

        if not Quntization_Option:
            pathList = [str(ip), "/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom3b_unquantized_res/device{}/module{}/module.zip".format(index,index)]

        else:
              pathList = [str(ip),
                    "/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom3b_quantized_int8_res/device{}/module{}/module.zip".format(
                        index, index)]
        pathLists.append(pathList)

    with open(os.path.join(to_send_path, 'ip_module.json'), 'w') as file:

        json.dump(pathLists, file)

    with open(os.path.join(to_send_path, 'ip_module.json'), 'r') as file:
        ip_module_json = file.read()
    ip_module = json.loads(ip_module_json)
    file_cfg = retrieve_file_cfg(ip_module)
    ip_graph, dependencyMap = retrieve_sending_info(root_dir, requested_model, ip_module_list=ip_module,
                                                    quantization_option=Quntization_Option,
                                                    residual_connection=residual_connection_option)



    print(f'\ngraph: {ip_graph}')
    print(f"session index: {session}")
    MODEL_EXIST_ON_DEVICE = True
    config = {"file_path": file_cfg,
              "num_sample": b'1000',
              "num_device": len(devices),
              "max_length": b'40',
              "task_type": "generation".encode('utf-8'),
              "core_pool_size": b'1',
              "head_node": ip_graph[0],
              "tail_node": ip_graph[-1],
              "dependency": dependencyMap,
              "session_index": ";".join(session).encode('utf-8'),
              "graph": ",".join(ip_graph).encode('utf-8'),
              "skip_model_transmission": MODEL_EXIST_ON_DEVICE,
              "model_name": requested_model,
              "reload_sampleId": None,
              "onnx": True,
              "ids": {}}

    # Read Dep json file
    for idx, fPath in dependencyMap.items():
        file = open(fPath, "r")
        data = json.load(file)
        config["dependency"][idx] = data

    msg_identifier = [0] * config["num_device"]

    print("finish configuration")

    status = {}
    threads = []

    lock = threading.Lock()
    locks = [threading.Lock(), threading.Lock()]
    conditions = [threading.Condition() for i in range(len(devices) + 1)]

    for i in range(config["num_device"]):
        t = threading.Thread(target=root_server.communication_open_close, args=(send, config, status, conditions, locks))
        threads.append(t)

    for i in threads:
        i.start()

    for t in threads:
        t.join()
        if t.exception:
            print(f"线程 {t.name} 出现异常: {t.exception}")

    send.close()
    context.term()
