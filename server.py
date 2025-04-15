import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# os.environ["DS_ACCELERATOR"]='cpu'
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
import socket
import traceback

monitor_receive_interval = 10  # set intervals for receiving monitor info from clients
monitor_port = "34567"  # set server port to receive monitor info
TIMEOUT =10 # Time to wait for new devices to connect to servers
MODEL_EXIST_ON_DEVICE = False  # set True if the model exists on the mobile device, will skip model creation and transmission
runtime_option = False  # set True if the load balance is runtime
split_size = 2
device_number =2
task = "Generation"
root_dir = os.path.dirname(os.path.abspath(__file__))
residual_connection_option = True

# 添加全局设备池和相关锁
all_devices_pool = deque()  # 全局设备池，存储所有已注册的设备
active_tasks = {}  # 格式: {task_id: {"devices": devices_list, "status": status}}
devices_pool_lock = threading.Lock()  # 设备池的线程锁

# 添加设备池管理类
class DevicePoolManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.device_pool = deque()  # 所有已注册设备
        self.active_devices = {}  # {task_id: device_list} 当前活跃任务使用的设备
        self.failed_devices = deque()  # 故障设备池
        self.task_counter = 0
        self.device_heartbeats = {}  # 记录设备最后心跳时间
        self.heartbeat_timeout = 30  # 心跳超时时间(秒)
        self.heartbeat_check_interval = 10  # 心跳检查间隔(秒)
    
    def update_device_heartbeat(self, device_id):
        """更新设备心跳时间"""
        with self.lock:
            self.device_heartbeats[device_id] = time.time()
    
    def check_device_heartbeats(self):
        """检查所有设备的心跳状态"""
        with self.lock:
            current_time = time.time()
            failed_devices = []
            
            # 检查所有设备的心跳
            for device in self.device_pool:
                device_id = device.get("device_id") or device["ip"]
                last_heartbeat = self.device_heartbeats.get(device_id, 0)
                
                if current_time - last_heartbeat > self.heartbeat_timeout:
                    failed_devices.append(device)
                    print(f"设备 {device_id} 心跳超时，可能已故障")
            
            # 处理故障设备
            for device in failed_devices:
                self.handle_device_failure(device)
    
    def handle_device_failure(self, device):
        """处理设备故障"""
        with self.lock:
            device_id = device.get("device_id") or device["ip"]
            
            # 从设备池中移除
            if device in self.device_pool:
                self.device_pool.remove(device)
                print(f"设备 {device_id} 已从设备池中移除")
            
            # 从活跃任务中移除
            for task_id, devices in list(self.active_devices.items()):
                if device in devices:
                    devices.remove(device)
                    print(f"设备 {device_id} 已从任务 {task_id} 中移除")
                    if not devices:  # 如果任务没有设备了，删除该任务
                        del self.active_devices[task_id]
                        print(f"任务 {task_id} 已删除，因为没有可用设备")
            
            # 添加到故障设备池
            device["failure_time"] = time.time()
            device["failure_reason"] = "heartbeat_timeout"
            self.failed_devices.append(device)
            print(f"设备 {device_id} 已添加到故障设备池")
            
            # 从心跳记录中移除
            if device_id in self.device_heartbeats:
                del self.device_heartbeats[device_id]
    
    def get_failed_devices(self):
        """获取故障设备列表"""
        with self.lock:
            return list(self.failed_devices)
    
    def get_device_status(self, device_id):
        """获取设备状态"""
        with self.lock:
            # 检查是否在故障设备池中
            for device in self.failed_devices:
                if (device.get("device_id") == device_id) or (device["ip"] == device_id):
                    return {
                        "status": "failed",
                        "failure_time": device.get("failure_time"),
                        "failure_reason": device.get("failure_reason")
                    }
            
            # 检查是否在活跃设备池中
            for device in self.device_pool:
                if (device.get("device_id") == device_id) or (device["ip"] == device_id):
                    last_heartbeat = self.device_heartbeats.get(device_id, 0)
                    return {
                        "status": "active",
                        "last_heartbeat": last_heartbeat,
                        "role": device.get("role")
                    }
            
            return {"status": "unknown"}

    def check_device_pool_integrity(self):
        """检查设备池中是否有重复项"""
        with self.lock:
            device_ids = set()
            ip_without_id = set()
            duplicates = []
            
            for device in self.device_pool:
                if "device_id" in device:
                    if device["device_id"] in device_ids:
                        duplicates.append(f"Duplicate device_id: {device['device_id']}")
                    device_ids.add(device["device_id"])
                else:
                    if device["ip"] in ip_without_id:
                        duplicates.append(f"Duplicate IP without device_id: {device['ip']}")
                    ip_without_id.add(device["ip"])
            
            if duplicates:
                print("WARNING: Found duplicates in device pool:")
                for dup in duplicates:
                    print(f"  - {dup}")
                return False
            return True
    
    def register_device(self, device_info):
        """注册新设备到设备池"""
        with self.lock:
            # 先获取设备的标识信息
            device_id = device_info.get("device_id")
            ip = device_info.get("ip")
            
            # 检查是否存在完全匹配的设备（基于设备ID和IP地址）
            exact_match_exists = False
            for device in self.device_pool:
                # 设备ID和IP完全匹配才视为同一设备
                if (device_id and device.get("device_id") == device_id and 
                    device["ip"] == ip):
                    # 更新设备信息
                    device.update(device_info)
                    print(f"Device updated: ID={device_id}, IP={ip}")
                    exact_match_exists = True
                    break
            
            if not exact_match_exists:
                # 检查是否存在相同ID但不同IP的设备（可能是同一物理设备的新IP）
                if device_id:
                    id_match_exists = False
                    for device in self.device_pool:
                        if device.get("device_id") == device_id:
                            if device["ip"] != ip:
                                print(f"WARNING: Same device ID but different IP detected. Old IP: {device['ip']}, New IP: {ip}")
                                id_match_exists = True
                                # 注意：我们不更新该设备，而是作为新设备添加
                
                # 查看是否有相同IP但不同ID的设备（可能是新设备使用了已存在的IP）
                ip_match_exists = False
                for device in self.device_pool:
                    if device["ip"] == ip and device.get("device_id") != device_id:
                        print(f"WARNING: Same IP but different device ID detected. Old ID: {device.get('device_id', 'None')}, New ID: {device_id or 'None'}")
                        ip_match_exists = True
                        # 注意：我们不更新该设备，而是作为新设备添加
                
                # 添加为新设备
                self.device_pool.append(device_info)
                print(f"New device registered: ID={device_id or 'None'}, IP={ip}, Role={device_info.get('role')}")
                
                # 最后打印当前设备池状态
                print(f"Current device pool has {len(self.device_pool)} devices.")
    
    def get_all_devices(self):
        """获取所有已注册设备"""
        with self.lock:
            return list(self.device_pool)
    
    def get_device_count(self):
        """获取设备总数"""
        with self.lock:
            return len(self.device_pool)
    
    def get_active_task_count(self):
        """获取活跃任务数量"""
        with self.lock:
            return len(self.active_devices)

    def get_main_thread_device_count(self):
        """获取主线程中的设备数量（非活跃设备）"""
        with self.lock:
            active_device_count = sum(len(devices) for devices in self.active_devices.values())
            return len(self.device_pool) - active_device_count
    
    def get_available_devices(self, required_count=None):
        """获取可用的设备列表"""
        with self.lock:
            # 找出未被分配给活跃任务的设备
            busy_devices = set()
            for devices in self.active_devices.values():
                for device in devices:
                    # 使用设备ID或IP作为标识
                    if "device_id" in device:
                        busy_devices.add(device["device_id"])
                    else:
                        busy_devices.add(device["ip"])
            
            available = []
            for d in self.device_pool:
                if "device_id" in d and d["device_id"] in busy_devices:
                    continue
                if "device_id" not in d and d["ip"] in busy_devices:
                    continue
                available.append(d)
            
            if required_count is not None and len(available) < required_count:
                print(f"Warning: Only {len(available)} devices available, but {required_count} required")
                return None
            
            return available
    
    def allocate_devices_for_task(self, device_count, task_id=None):
        """为任务分配设备"""
        with self.lock:
            available = self.get_available_devices()
            
            if len(available) < device_count:
                print(f"Not enough devices available. Required: {device_count}, Available: {len(available)}")
                return None
            
            # 分配设备
            allocated = []
            header_allocated = False
            
            # 优先分配header设备
            for device in available:
                if not header_allocated and device["role"] == "header":
                    allocated.append(device)
                    header_allocated = True
                    if len(allocated) == device_count:
                        break
            
            # 分配其他设备
            for device in available:
                if device not in allocated:
                    allocated.append(device)
                    if len(allocated) == device_count:
                        break
            
            # 如果没有指定task_id，创建新的
            if task_id is None:
                self.task_counter += 1
                task_id = f"task_{self.task_counter}"
            
            # 记录任务使用的设备
            self.active_devices[task_id] = allocated
            
            return task_id, allocated
    
    def release_task_devices(self, task_id):
        """释放任务使用的设备"""
        with self.lock:
            if task_id in self.active_devices:
                del self.active_devices[task_id]
                print(f"Released devices for task {task_id}")
                return True
            return False
    
    def get_task_devices(self, task_id):
        """获取指定任务的设备列表"""
        with self.lock:
            return self.active_devices.get(task_id, [])

# 创建设备池管理器实例
device_pool_manager = DevicePoolManager()

def heartbeat_check_thread():
    """心跳检查线程"""
    while True:
        device_pool_manager.check_device_heartbeats()
        time.sleep(device_pool_manager.heartbeat_check_interval)

# 设备注册线程函数
def device_registration_thread(context, port=23456, main_devices=None, ip_graph_requested_list=None):
    """持续监听并处理设备注册请求的线程"""
    global requested_model  # 添加全局声明以便可以修改主程序变量
    
    registration_socket = server.establish_connection(context, zmq.ROUTER, port)
    
    print(f"Device registration thread started. Listening for device registrations on port {port}...")
    
    while True:  # 持续运行
        if registration_socket.poll(1000):  # 1秒超时
            try:
                identifier, action, msg_content = registration_socket.recv_multipart()
                
                if action.decode() == "RegisterIP":
                    print("Device registration request received")
                    jsonObject = json.loads(msg_content.decode())
                    ip = jsonObject.get("ip")
                    role = jsonObject.get("role")
                    model_request = jsonObject.get("model", None)
                    device_id = jsonObject.get("device_id", None)
                    
                    device_info = {
                        "ip": ip, 
                        "role": role, 
                        "identifier": identifier,
                        "last_seen": time.time(),
                        "model_request": model_request
                    }
                    
                    # 如果提供了设备ID，则添加到设备信息中
                    if device_id:
                        device_info["device_id"] = device_id
                        print(f"Device registration with device_id: {device_id}")
                    
                    # 注册到设备池
                    device_pool_manager.register_device(device_info)
                    
                    # 更新设备心跳
                    device_pool_manager.update_device_heartbeat(device_id or ip)
                    
                    # 回复设备已注册成功
                    registration_socket.send_multipart([identifier, b"REGISTRATION_SUCCESSFUL", b"Device registered successfully"])
                    print(f"Device registration response sent to {ip}")
                    
                    # 同步到主线程的设备列表
                    if main_devices is not None:
                        device_entry = {"ip": ip, "role": role}
                        if device_id:
                            device_entry["device_id"] = device_id
                            
                        with devices_pool_lock:
                            # 避免重复添加 - 根据设备ID或IP检查
                            exists = False
                            for dev in main_devices:
                                if (device_id and dev.get("device_id") == device_id) or (not device_id and dev["ip"] == ip):
                                    exists = True
                                    break
                                    
                            if not exists:
                                if role == "header":
                                    main_devices.appendleft(device_entry)
                                    # 更新请求的模型
                                    if model_request:
                                        global requested_model
                                        requested_model = model_request
                                else:
                                    main_devices.append(device_entry)
                                    
                            print(f"Synchronized device to main thread: {device_id or ip} as {role}")
                            
                    # 添加到IP图请求列表
                    if ip_graph_requested_list is not None and identifier not in ip_graph_requested_list:
                        ip_graph_requested_list.append(identifier)
                        print(f"Added {device_id or ip} to IP graph request list")
                
                elif action.decode() == "HEARTBEAT":
                    # 处理设备心跳
                    jsonObject = json.loads(msg_content.decode())
                    device_id = jsonObject.get("device_id")
                    if device_id:
                        device_pool_manager.update_device_heartbeat(device_id)
                        registration_socket.send_multipart([identifier, b"HEARTBEAT_ACK", b"Heartbeat received"])
                
                elif action.decode() == "GET_STATUS":
                    # 设备查询状态
                    print("\n--- Processing status query request ---")
                    # 检查设备池完整性
                    is_pool_intact = device_pool_manager.check_device_pool_integrity()
                    
                    all_devices = device_pool_manager.get_all_devices()
                    total_devices = len(all_devices)
                    failed_devices = device_pool_manager.get_failed_devices()
                    
                    # 获取详细信息用于诊断
                    available_devices = len(device_pool_manager.get_available_devices() or [])
                    active_tasks_count = device_pool_manager.get_active_task_count()
                    main_thread_devices = len(main_devices) if main_devices is not None else 0
                    
                    print(f"设备池状态:")
                    print(f"  总设备数: {total_devices}")
                    print(f"  可用设备数: {available_devices}")
                    print(f"  活跃任务数: {active_tasks_count}")
                    print(f"  主线程设备数: {main_thread_devices}")
                    print(f"  故障设备数: {len(failed_devices)}")
                    print(f"  设备池完整性: {'正常' if is_pool_intact else '存在重复项'}")
                    
                    # 打印所有设备的详情
                    print("设备池中的设备:")
                    for i, device in enumerate(all_devices):
                        device_id = device.get("device_id", "N/A")
                        ip = device.get("ip", "N/A")
                        role = device.get("role", "N/A")
                        status = device_pool_manager.get_device_status(device_id)
                        print(f"  {i+1}. ID: {device_id}, IP: {ip}, 角色: {role}, 状态: {status['status']}")
                    
                    # 打印故障设备详情
                    if failed_devices:
                        print("\n故障设备列表:")
                        for i, device in enumerate(failed_devices):
                            device_id = device.get("device_id", "N/A")
                            ip = device.get("ip", "N/A")
                            failure_time = device.get("failure_time", "N/A")
                            failure_reason = device.get("failure_reason", "N/A")
                            print(f"  {i+1}. ID: {device_id}, IP: {ip}, 故障时间: {failure_time}, 原因: {failure_reason}")
                    
                    status_info = {
                        "total_devices": total_devices,
                        "available_devices": available_devices,
                        "active_tasks": active_tasks_count,
                        "main_thread_devices": main_thread_devices,
                        "failed_devices": len(failed_devices),
                        "pool_integrity": "ok" if is_pool_intact else "duplicates_found",
                        "devices": [
                            {
                                "id": d.get("device_id", "N/A"),
                                "ip": d.get("ip", "N/A"),
                                "role": d.get("role", "N/A"),
                                "status": device_pool_manager.get_device_status(d.get("device_id") or d.get("ip"))
                            } for d in all_devices
                        ],
                        "failed_devices": [
                            {
                                "id": d.get("device_id", "N/A"),
                                "ip": d.get("ip", "N/A"),
                                "failure_time": d.get("failure_time"),
                                "failure_reason": d.get("failure_reason")
                            } for d in failed_devices
                        ]
                    }
                    
                    registration_socket.send_multipart([
                        identifier, 
                        b"STATUS_INFO", 
                        json.dumps(status_info).encode('utf-8')
                    ])
                    print("Status information sent to client")
            except Exception as e:
                print(f"Error in device registration thread: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 注意：这里永远不会执行到，因为线程持续运行
    registration_socket.close()

def process_status_query(data, conn, addr):
    """处理状态查询请求"""
    try:
        # 获取设备池管理器实例
        device_manager = DevicePoolManager()
        
        # 检查设备池完整性
        device_manager.check_device_pool_integrity()
        
        # 获取设备池状态
        total_devices = device_manager.get_device_count()
        available_devices = len(device_manager.get_available_devices())
        active_tasks = device_manager.get_active_task_count()
        main_thread_devices = device_manager.get_main_thread_device_count()
        failed_devices = device_manager.get_failed_devices()
        
        # 构建状态信息
        status_info = {
            "total_devices": total_devices,
            "available_devices": available_devices,
            "active_tasks": active_tasks,
            "main_thread_devices": main_thread_devices,
            "failed_devices": len(failed_devices),
            "failed_devices_list": [
                {
                    "device_id": device.get("device_id", "Unknown"),
                    "ip": device.get("ip", "Unknown"),
                    "role": device.get("role", "Unknown"),
                    "failure_time": device.get("failure_time", "Unknown"),
                    "failure_reason": device.get("failure_reason", "Unknown")
                }
                for device in failed_devices
            ]
        }
        
        # 发送状态信息
        conn.send_multipart([b"STATUS", json.dumps(status_info).encode('utf-8')])
        
        # 打印状态信息到服务器控制台
        print("\n设备池状态:")
        print(f"  总设备数: {total_devices}")
        print(f"  可用设备数: {available_devices}")
        print(f"  活跃任务数: {active_tasks}")
        print(f"  主线程设备数: {main_thread_devices}")
        print(f"  故障设备数: {len(failed_devices)}")
        
        if failed_devices:
            print("\n故障设备列表:")
            for device in failed_devices:
                device_id = device.get("device_id", "Unknown")
                ip = device.get("ip", "Unknown")
                role = device.get("role", "Unknown")
                failure_time = device.get("failure_time", "Unknown")
                failure_reason = device.get("failure_reason", "Unknown")
                print(f"  - 设备ID: {device_id}")
                print(f"    角色: {role}")
                print(f"    IP: {ip}")
                print(f"    故障时间: {failure_time}")
                print(f"    故障原因: {failure_reason}")
        
    except Exception as e:
        print(f"处理状态查询时出错: {e}")
        conn.send_multipart([b"ERROR", str(e).encode('utf-8')])

def main_thread():
    """主线程，处理任务分配和模型推理请求"""
    global running
    
    # 创建服务器套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    server_socket.settimeout(1)
    
    print(f"Server started on {HOST}:{PORT}")
    
    try:
        while running:
            try:
                conn, addr = server_socket.accept()
                print(f"Connected by {addr}")
                
                data = receive_data(conn)
                if not data:
                    continue
                
                request_type = data.get("type")
                
                if request_type == "inference":
                    # 处理推理请求
                    process_inference_request(data, conn, addr)
                elif request_type == "GET_STATUS":
                    # 处理状态查询请求
                    process_status_query(data, conn, addr)
                else:
                    print(f"Unknown request type: {request_type}")
                    response = json.dumps({"status": "error", "message": "Unknown request type"}).encode()
                    conn.sendall(response)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error in main thread: {str(e)}")
                traceback.print_exc()
    finally:
        server_socket.close()
        print("Main thread stopped")

if __name__ == "__main__":
    start = time.time()
    context = zmq.Context()
    send = server.establish_connection(context, zmq.ROUTER, 23456)
    
    # 设置默认模型，防止未定义错误
    requested_model = "bloom560m-int8"  # 默认模型
    
    # 定义常量
    HOST = '0.0.0.0'  # 监听所有网络接口
    PORT = 5000  # HTTP服务器端口
    running = True  # 控制主线程运行的标志
    
    # 启动心跳检查线程
    heartbeat_thread = threading.Thread(
        target=heartbeat_check_thread,
        daemon=True
    )
    heartbeat_thread.start()
    
    # 提前定义用于处理HTTP请求的函数
    def receive_data(conn):
        """从连接中接收HTTP数据"""
        try:
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\r\n\r\n" in data:
                    break
            
            if not data:
                return None
                
            # 打印收到的完整请求，用于调试
            print("Received HTTP request:")
            print(data.decode('utf-8', errors='ignore'))
            
            # 解析HTTP请求
            try:
                # 先解析HTTP请求行和头部
                headers_part, body_part = data.split(b"\r\n\r\n", 1) if b"\r\n\r\n" in data else (data, b"")
                request_line = headers_part.split(b"\r\n")[0].decode()
                
                # 提取请求方法和路径
                method, path, _ = request_line.split(" ", 2)
                print(f"HTTP Method: {method}, Path: {path}")
                
                # 处理GET请求，通常状态查询是通过GET请求发送的
                if method == "GET":
                    if "status" in path.lower() or "get_status" in path.lower():
                        return {"type": "GET_STATUS"}
                
                # 处理POST请求，通常推理请求是通过POST发送的
                elif method == "POST" and body_part:
                    try:
                        # 尝试解析JSON数据
                        body_json = json.loads(body_part.decode())
                        return body_json
                    except json.JSONDecodeError:
                        print("Warning: Failed to parse JSON in POST body")
                        
                # 兼容非HTTP协议请求（直接发送JSON字符串）
                if body_part:
                    try:
                        return json.loads(body_part.decode())
                    except:
                        pass
                
                # 最后，尝试处理可能的简单文本命令
                if b"GET_STATUS" in data:
                    return {"type": "GET_STATUS"}
                elif b"inference" in data:
                    return {"type": "inference"}
                
            except Exception as e:
                print(f"Error parsing HTTP request: {e}")
                traceback.print_exc()
                
            return None
        except Exception as e:
            print(f"Error receiving data: {e}")
            traceback.print_exc()
            return None
    
    # 添加函数用于处理推理请求
    def process_inference_request(data, conn, addr):
        """处理推理请求"""
        print(f"Processing inference request from {addr}")
        response = json.dumps({"status": "error", "message": "Inference not implemented yet"}).encode()
        
        # 构建 HTTP 响应
        http_response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response)}\r\n\r\n"
        conn.sendall(http_response.encode() + response)
    
    # 原有的设备注册和连接过程（仍然保留以兼容现有代码）
    # 但现在可以有限的时间窗口收集初始设备集合
    devices = deque()
    add_devices = deque()
    lost_devices = deque()
    ip_graph_requested = []  # Buffer to store addresses from devices
    last_received_time = time.time()
    continue_listening = True
    last_print_time = time.time()
    
    # 启动设备注册线程
    registration_thread = threading.Thread(
        target=device_registration_thread,
        args=(context, 23457, devices, ip_graph_requested),  # 传递设备队列和IP请求列表
        daemon=True  # 设为守护线程，主线程结束时自动终止
    )
    registration_thread.start()
    
    # 启动HTTP服务器主线程
    http_server_thread = threading.Thread(
        target=main_thread,
        daemon=True
    )
    http_server_thread.start()
    
    # 等待设备连接的时间窗口
    print(f"Waiting for initial devices to connect (timeout: {TIMEOUT} seconds)...")
    
    # 收集初始设备集合
    while continue_listening:
        current_time = time.time()
        # 每隔3秒打印一次当前时间
        if current_time - last_print_time >= 3:
            print(f"Current time: {current_time:.2f} seconds since epoch")
            last_print_time = current_time  # 更新打印时间
        if send.poll(1000):
            print("start listening")
            identifier, action, msg_content = send.recv_multipart()

            # 计算等待时间（从上次接收到消息到现在的时间差）
            wait_time = current_time - last_received_time
            print(f"Wait time: {wait_time:.2f} seconds")  # 打印等待时间，保留两位小数

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
                
                # 创建设备信息
                device_info = {
                    "ip": ip, 
                    "role": role, 
                    "identifier": identifier,
                    "last_seen": time.time(),
                    "model_request": model_request
                }
                
                # 同时注册到传统设备集合和新的设备池
                if role == "header":
                    devices.appendleft({"ip": ip, "role": role})
                    if model_request:
                        requested_model = model_request
                else:
                    devices.append({"ip": ip, "role": role})
                
                # 注册到设备池
                device_pool_manager.register_device(device_info)
                
                last_received_time = time.time()

        if time.time() - last_received_time > TIMEOUT:
            print("No new devices connected in the last", TIMEOUT, "seconds. Broadcasting message.")
            continue_listening = False
                                                    
    print(f'devices in root.py: {devices}')
    # requested_model = "opt125m"
    print(f'request model = {requested_model}')

    # 创建第一个任务并分配设备
    # 注意：任务只有在实际需要使用设备时才应该被分配
    # 初始阶段只是记录设备，但不要立即将它们标记为活跃使用中
    print(f"Found {len(devices)} initial devices. These will be used for the first task when needed.")
    
    # 暂时不预先分配设备给任务，而是在实际需要时再分配
    # task_id = "task_1" 
    # device_pool_manager.active_devices[task_id] = list(devices)

    # 仅当有推理需求时再创建任务并分配设备
    # 以下保持原代码不变，但设备只在实际使用时才会被分配
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
        if requested_model == "bloom560m":
            Quntization_Option = False
        if requested_model == "bloom560m-int8":
            Quntization_Option = True

        requested_model='bloom560m'
        to_send_path = retrieve_sending_dir(root_dir, requested_model, quantization_option=Quntization_Option,
                                            residual_connection=residual_connection_option)

        if  os.path.isdir(to_send_path):
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
            pathList = [str(ip), "/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_unquantized_res/device{}/module{}/module.zip".format(index,index)]

        else:
              pathList = [str(ip),
                    "/workspace/ams-LinguaLinked-Inference/onnx_model__/to_send/bloom560m_quantized_int8_res/device{}/module{}/module.zip".format(
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

    # 当任务完成时，释放设备
    # 这应该在所有处理完成后添加
    # device_pool_manager.release_task_devices(task_id)
    
    # 任务结束后，可以开始新的任务
    # 例如: task_id_2, new_devices = device_pool_manager.allocate_devices_for_task(required_device_count)
