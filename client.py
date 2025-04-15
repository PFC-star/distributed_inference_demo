#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设备动态注册示例客户端
用于演示如何在任何时刻将设备注册到服务器的设备池中
"""

import zmq
import json
import time
import argparse
import socket
import sys
import threading

def get_local_ip():
    """获取本地IP地址"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 不需要真正连接
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def send_heartbeat(socket, device_id):
    """发送心跳到服务器"""
    try:
        heartbeat_data = {
            "device_id": device_id,
            "timestamp": time.time()
        }
        socket.send_multipart([b"HEARTBEAT", json.dumps(heartbeat_data).encode('utf-8')])
        response = socket.recv_multipart()
        print(f"收到心跳响应: {response[0].decode()} - {response[1].decode()}")
        return True
    except zmq.Again:
        print(f"心跳发送超时，设备 {device_id} 可能已与服务器断开连接")
        return False
    except zmq.ZMQError as e:
        print(f"ZMQ错误: {e}")
        return False
    except Exception as e:
        print(f"发送心跳失败: {e}")
        return False

def heartbeat_thread(socket, device_id, interval=5):
    """心跳发送线程"""
    retry_count = 0
    max_retries = 3
    
    while True:
        success = send_heartbeat(socket, device_id)
        
        if not success:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"设备 {device_id} 心跳连续失败 {max_retries} 次，尝试重新连接...")
                try:
                    # 重新连接
                    socket.close()
                    context = zmq.Context()
                    socket = context.socket(zmq.DEALER)
                    socket.identity = device_id.encode('utf-8')
                    socket.setsockopt(zmq.RCVTIMEO, 5000)
                    socket.connect(f"tcp://{server_address}:{server_port}")
                    print(f"设备 {device_id} 重新连接成功")
                    retry_count = 0
                except Exception as e:
                    print(f"重新连接失败: {e}")
                    time.sleep(interval)
                    continue
            else:
                print(f"心跳失败，将在 {interval} 秒后重试 (第 {retry_count} 次)")
        else:
            retry_count = 0
            
        time.sleep(interval)

def register_device(server_address, server_port, device_role, model_request=None, device_id=None, virtual_ip=None):
    """注册设备到服务器设备池"""
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    
    # 生成一个唯一标识符
    if device_id is None:
        device_id = f"device_{int(time.time())}"
    socket.identity = device_id.encode('utf-8')
    
    # 设置接收超时，单位是毫秒
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    
    # 连接到服务器 - 注册端口使用专用的设备注册端口
    socket.connect(f"tcp://{server_address}:{server_port}")
    
    # 准备设备信息
    device_info = {
        "ip": virtual_ip if virtual_ip else get_local_ip(),
        "role": device_role,
        "device_id": device_id  # 添加设备ID作为唯一标识
    }
    
    if model_request:
        device_info["model"] = model_request
    
    # 发送注册请求
    print(f"正在注册设备 {device_id} 到服务器 {server_address}:{server_port}...")
    print(f"使用IP: {device_info['ip']}")
    socket.send_multipart([b"RegisterIP", json.dumps(device_info).encode('utf-8')])
    
    # 等待注册确认
    try:
        response = socket.recv_multipart()  # 不再使用timeout参数
        print(f"收到服务器响应: {response[0].decode()} - {response[1].decode()}")
        # 成功注册，返回socket和context以便继续使用
        return True, socket, context, device_id
    except zmq.Again:
        print("注册超时，服务器没有响应")
        socket.close()
        context.term()
        return False, None, None, None
    except Exception as e:
        print(f"注册过程出错: {e}")
        socket.close()
        context.term()
        return False, None, None, None

def query_device_pool_status(server_address, server_port):
    """查询设备池状态"""
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    
    # 生成一个唯一标识符
    query_id = f"query_{int(time.time())}"
    socket.identity = query_id.encode('utf-8')
    
    # 设置接收超时，单位是毫秒
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    
    # 连接到服务器
    socket.connect(f"tcp://{server_address}:{server_port}")
    
    # 发送状态查询请求
    print(f"正在查询设备池状态...")
    socket.send_multipart([b"GET_STATUS", b""])
    
    # 等待响应
    try:
        response = socket.recv_multipart()  # 不再使用timeout参数
        status_info = json.loads(response[1].decode())
        print("\n设备池状态:")
        print(f"  总设备数: {status_info['total_devices']}")
        print(f"  可用设备数: {status_info['available_devices']}")
        print(f"  活跃任务数: {status_info['active_tasks']}")
        
        # 显示主线程设备数量（如果服务器返回了这个信息）
        if 'main_thread_devices' in status_info:
            print(f"  主线程设备数: {status_info['main_thread_devices']}")
            if status_info['main_thread_devices'] == 0:
                print("\n警告: 设备已注册到设备池，但未同步到主线程！")
                print("这可能导致推理任务无法正常启动。请检查服务器设置。")
        
        return status_info
    except zmq.Again:
        print("查询超时，服务器没有响应")
        return None
    except Exception as e:
        print(f"查询出错: {e}")
        return None
    finally:
        socket.close()
        context.term()

def main():
    parser = argparse.ArgumentParser(description='设备动态注册客户端')
    parser.add_argument('--server', default='localhost', help='服务器地址')
    parser.add_argument('--port', type=int, default=23457, help='服务器设备注册端口')
    parser.add_argument('--role', choices=['header', 'worker', 'tail'], default='worker', 
                        help='设备角色 (header: 头节点, worker: 工作节点, tail: 尾节点)')
    parser.add_argument('--model',default='bloom560m-int8', help='请求的模型名称 (例如: bloom560m, bloom560m-int8)')
    parser.add_argument('--query', action='store_true', help='只查询设备池状态')
    parser.add_argument('--device-id', help='指定唯一设备ID（用于测试多设备场景）')
    parser.add_argument('--virtual-ip', help='指定虚拟IP地址（用于测试多设备场景）')
    parser.add_argument('--heartbeat-interval', type=int, default=5, help='心跳发送间隔（秒）')
    
    args = parser.parse_args()
    
    # 将服务器地址和端口设为全局变量，以便心跳线程使用
    global server_address, server_port
    server_address = args.server
    server_port = args.port
    
    if args.query:
        query_device_pool_status(args.server, args.port)
    else:
        success, socket, context, device_id = register_device(
            args.server, args.port, args.role, args.model, args.device_id, args.virtual_ip
        )
        
        if success:
            print(f"\n设备已成功注册为 '{args.role}'")
            if args.model:
                print(f"已请求模型: {args.model}")
            
            # 注册后，查询一下当前设备池状态
            time.sleep(1)
            status_info = query_device_pool_status(args.server, args.port)
            
            try:
                print("\n开始持续发送心跳...")
                print("按 Ctrl+C 终止程序")
                
                # 启动心跳线程
                heartbeat_t = threading.Thread(
                    target=heartbeat_thread,
                    args=(socket, device_id, args.heartbeat_interval),
                    daemon=True
                )
                heartbeat_t.start()
                
                # 主线程保持运行，直到用户中断
                while True:
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n用户中断，正在关闭连接...")
            finally:
                if socket:
                    socket.close()
                if context:
                    context.term()
                print("客户端已关闭")
        else:
            print("设备注册失败")
            sys.exit(1)

if __name__ == "__main__":
    main() 