package com.example.SecureConnection;

import org.zeromq.ZContext;
import org.zeromq.ZMQ;
import org.json.JSONObject;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import com.example.distribute_ui.service.MonitorServiceKt;
import com.example.SecureConnection.Config;
public class HeartbeatSender {

    private static final String SERVER_HOST = "tcp://10.214.149.209:34567";  // 服务器的 IP 地址和端口
    private static final long HEARTBEAT_INTERVAL = 5000;  // 5秒发送一次心跳包

    private static ZMQ.Socket monitorSocket;
    // 创建 Config 实例
    static Config config = new Config("127.0.0.1", 8080);

    // 调用 getCurrentDeviceIP() 方法
    static String deviceIP = config.getCurrentDeviceIP();



    public static void main(String[] args) {
        // 启动心跳发送功能
        heartrun();  // 调用 heartrun 方法启动心跳发送
    }

    public static void heartrun() {
        // 创建 ZeroMQ 上下文
        ZContext context = new ZContext();

        // 创建 DEALER 套接字
        monitorSocket = context.createSocket(ZMQ.DEALER);
        monitorSocket.connect(SERVER_HOST);  // 连接到指定的服务器

        // 创建一个定时任务调度器
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

        // 定时发送心跳包
        scheduler.scheduleWithFixedDelay(new Runnable() {
            @Override
            public void run() {
                sendHeartbeat();
            }
        }, 0, HEARTBEAT_INTERVAL, TimeUnit.MILLISECONDS);  // 每5秒发送一次
    }

    public static void sendHeartbeat() {
        try {
            // 创建一个 JSON 对象来存储心跳包信息
            JSONObject jsonObject = new JSONObject();
            String currentIP =  deviceIP;  // 获取本地 IP 地址 (根据需要替换)
            jsonObject.put("ip", currentIP);
            jsonObject.put("role", "client");  // 设置角色

            // 将 JSON 转换为字符串并发送
            String heartbeatMessage = jsonObject.toString();
            monitorSocket.send(heartbeatMessage.getBytes(StandardCharsets.UTF_8));

            System.out.println("Sent heartbeat: " + heartbeatMessage);
        } catch (org.json.JSONException e) {
            // 捕获并处理 JSON 异常
            System.err.println("Error creating JSON object: " + e.getMessage());
        } catch (Exception e) {
            // 捕获其他异常
            System.err.println("Error sending heartbeat: " + e.getMessage());
        }
    }
}