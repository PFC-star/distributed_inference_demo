package com.example.SecureConnection;
import static com.example.distribute_ui.BackgroundService.TAG;

import android.annotation.SuppressLint;
import android.util.Log;
import org.json.JSONArray;
import org.json.JSONException;
import org.junit.Test;
import org.zeromq.ZMQ;
import java.io.IOException;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.*;
import org.zeromq.ZMQ.Socket;
import org.zeromq.ZContext;
import org.zeromq.SocketType;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.IntStream;
import org.json.JSONObject;
import com.example.SecureConnection.Utils.LBPause;
import com.example.distribute_ui.DataRepository;
import java.util.ArrayList;
import com.example.SecureConnection.HeartbeatSender;

import static com.example.distribute_ui.BackgroundService.TAG;

import android.content.Intent;
import android.util.Log;

import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.TimeUnit;
import java.io.File;
import java.io.FileOutputStream;
import java.util.zip.ZipOutputStream;

import org.greenrobot.eventbus.EventBus;
import org.json.JSONException;
import org.json.JSONObject;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;
import org.zeromq.ZMQ.Socket;

import java.io.IOException;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipEntry;
import com.example.SecureConnection.Utils;
import com.example.distribute_ui.BackgroundService;
import com.example.distribute_ui.Events;

public class Communication {

    public class Params {
        public String modelPath;

        public String sourcePath;
        public int max_length;
        public String task_type;
        public boolean skip_special_token;
        public int corePoolSize;
        public int num_device;
        public String[] classes;

        public volatile String status = "Ready";
        public int numSample;

    }
    public static volatile Long tokenizer;

    public static Long session;
    public static ArrayList<Long> sessions;
    public static String[] sessionIndex;
    public static LoadBalance loadBalance;
    public static LBPause LB_Pause;
    public ExecutorService executor;

    public Params param;
    public Config cfg;
    public double commutime,infertime;
    public ArrayList<Byte> byteArraySum = new ArrayList<>();
    public ArrayList<Double> commutimeArraySum = new ArrayList<>();
    public ArrayList<Double> infertimeArraySum = new ArrayList<>();
    public String[] InputString;
    public volatile Map<Integer, ArrayList<Integer>> InputIds;
    public volatile Map<Integer, byte[]> InputData;
    public volatile Map<Integer, byte[]> OutputData; //放着header的输出结果，并发送给了worker，worker放在了inputdata里，然后推理，推理结果写在了outputdata里，然后返回到inputIDs
    public volatile Map<Integer, Map<String, ArrayList<byte[]>>> ResidualDataFromDevice;
    public volatile Map<Integer, Map<String , ArrayList<byte[]>>> ResidualDataToDevice;
    public volatile Map<Integer, byte[]> logits;
    public int sampleId;

    public boolean valid;

    public Map<String, Socket> sockets;

    public ZContext context;
    public Socket rootSocket;
    public Client beClient;
    public Server beServer;
    public LinkedBlockingQueue<ArrayList<Map<Integer, Socket>>> allSockets;
    //    public LinkedBlockingQueue<Map<Integer, Socket>> serverSockets;
//    public LinkedBlockingQueue<Map<Integer, Socket>> clientSockets;
    public double[] timeUsage;
    public Set<Integer> receiveDeviceIndex;
    public Set<Integer> sendDeviceIndex;
    public TreeMap<String, ArrayList<JSONObject>> sendIndex;
    public TreeMap<String, ArrayList<JSONObject>> receiveIndex;
    public TreeMap<Integer, ArrayList<String>> sendD2D;
    public TreeMap<Integer, ArrayList<String>> receiveD2D;
    public TreeMap<String, Integer> module_on_devices;

    public Communication(Config cfg) {
        Communication.sessions = new ArrayList<>();
        Communication.LB_Pause = new LBPause();

        this.cfg = cfg;
        param = new Params();
        param.skip_special_token = false;

        InputIds = new HashMap<>();
        InputData = new ConcurrentHashMap<>();
        OutputData = new ConcurrentHashMap<>();
        ResidualDataFromDevice = new ConcurrentHashMap<>();
        ResidualDataToDevice = new ConcurrentHashMap<>();
        logits = new ConcurrentHashMap<>();

        sampleId = 0;
        sockets = new HashMap<>();
        context = new ZContext();
        beClient = new Client();
        beServer = new Server();
//        HeartbeatSender heartbeatSender = new HeartbeatSender();
        allSockets = new LinkedBlockingQueue<>();
//        serverSockets = new LinkedBlockingQueue<>();
//        clientSockets = new LinkedBlockingQueue<>();
        sendIndex = new TreeMap<>();
        receiveIndex = new TreeMap<>();

        sendDeviceIndex = new TreeSet<>();
        receiveDeviceIndex = new TreeSet<>();
        timeUsage = new double[2];

        module_on_devices = new TreeMap<>();

    }

    public boolean sendIPToServer(String role, String modelRequest) throws JSONException {
        rootSocket = beClient.establish_connection(context, SocketType.DEALER, cfg.rootPort, cfg.root);
        Log.d(TAG, "socket establish connection");
        String currentIP = Config.local;
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("ip", currentIP);
        jsonObject.put("role", role);
        if ("header".equals(role)) {
            jsonObject.put("model", modelRequest); // Include the requested model specified by user
        }
        rootSocket.sendMore("RegisterIP");
        rootSocket.send(jsonObject.toString());
        Log.d(TAG, "IP message sent");
        String need_monitor = new String(rootSocket.recv(0));
        return need_monitor.equals("True");
    }

    public void runPrepareThread(){
        executor = Executors.newFixedThreadPool(2);
        executor.submit(()-> {
            try {
                this.prepare();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    public void runRunningThread(int corePoolSize, int maximumPoolSize, int keepAliveTime, ArrayList<String> input_data){
        executor.submit(()-> {
            try {
                this.running(corePoolSize, maximumPoolSize, keepAliveTime, input_data);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    public void shutDownPrepare(){
        executor.shutdown();
    }

    public void prepare() throws Exception {
        long startTime = System.nanoTime();
        // Communicate with Root Root Server  应该算在准备时间里
        System.out.println(cfg.root);
        System.out.println(cfg.rootPort);
        Log.d(TAG, "root IP: " + cfg.root +  " ,root port: " + cfg.rootPort);
        beClient.communicationOpenClose(cfg, this, rootSocket);
        long prepareTime = System.nanoTime();
        System.out.println("Prepare Time in seconds: " + (prepareTime - startTime) / 1000000000.0);
        Log.d(TAG, "Prepare Time in seconds: " + (prepareTime - startTime) / 1000000000.0);
        timeUsage[0] = (prepareTime - startTime) / 1000000000.0;
    }

    public void cleanUpBuffer(int id) {
//        ResidualDataToDevice.remove(id);
//        ResidualDataFromDevice.remove(id);

        InputData.remove(id);
        OutputData.remove(id);
    }

    public int[][] getResIndices(int module_idx) throws JSONException {
        if (!sendIndex.containsKey(sessionIndex[module_idx]) || sendIndex.get(sessionIndex[module_idx]).size() <= 1 )
            return new int[0][];

        JSONObject resIndex = sendIndex.get(sessionIndex[module_idx]).get(1);
        resIndex.keys();
        int[][] ResIndex = new int[resIndex.length()][];
        Iterator<String> keys = resIndex.keys();
        // Need to sorted before computing
        List<String> tmp = new ArrayList<>();
        while (keys.hasNext())
            tmp.add(keys.next());
        Collections.sort(tmp);

        for (int i=0; i< tmp.size(); i++){
            ResIndex[i] = Utils.JsonArray2IntArray(resIndex.getJSONArray(tmp.get(i)));
        }
        return ResIndex;
    }


    public ArrayList<byte[]> mergeResFromAndToDevice(int id, String module_idx){
        ArrayList<byte[]> data1 = new ArrayList<>();
        ArrayList<byte[]> data2 = new ArrayList<>();
        if (ResidualDataFromDevice.containsKey(id) && ResidualDataFromDevice.get(id).containsKey(module_idx))
            data1 = ResidualDataFromDevice.get(id).get(module_idx); // Comes from previous module on the different device
        if (ResidualDataToDevice.containsKey(id) && ResidualDataToDevice.get(id).containsKey(module_idx))
            data2 = ResidualDataToDevice.get(id).get(module_idx);   // Comes from previous module on the same device
//        System.out.println("Merge ResidualDataToDevice Size: " +  data1.size());
        data1.addAll(data2);  // if sorted, then from previous device first, then from local device next.
//        System.out.println("Merge ResidualDataToDevice Size: " +  data2.size());
        System.out.println("Merge Receive Res Size: " +  data1.size());
        return data1;
    }

    public void addBytes(byte[] bytes) {
        for (byte b : bytes) {
            byteArraySum.add(b);
        }
    }
    public void convertOutput(int id, int module_idx, Object[] result){

        OutputData.put(id, (byte[]) result[0]);
        long sTime = System.nanoTime();
//        addBytes((byte[]) result[0]);
        long eTime = System.nanoTime();
        System.out.println("通信OutputData: " + result[0]);
        System.out.println("No." + id + " addBytes Time in seconds: " + (eTime - sTime) / 1000000000.0);

        JSONObject resIndex = null;
        if (sendIndex.get(sessionIndex[module_idx]).size() > 1)
            resIndex = sendIndex.get(sessionIndex[module_idx]).get(1);

        if (result.length > 1 && resIndex != null) {
            Iterator<String> keys = resIndex.keys();
            int i = 0;
            while (keys.hasNext()) {
                String k = keys.next();
                if (!ResidualDataToDevice.get(id).containsKey(k))
                    ResidualDataToDevice.get(id).put(k, new ArrayList<>());
                byte[][] val = (byte[][])result[1];
                ResidualDataToDevice.get(id).get(k).add(val[i]);    // Careful about the order added in
                i++;
            }
            if (ResidualDataToDevice.get(id).size() > 0)
                for (Map.Entry<String, ArrayList<byte[]>> e: ResidualDataToDevice.get(id).entrySet())
                    if (e.getValue().size() > 1)
                        System.out.println("To Module "+ e.getKey() +" receive byte: "+  e.getValue().size());
        }
    }

    public void inferenceProcedure(int id) throws JSONException {
        long sTime = System.nanoTime();
        if (((InputData.containsKey(id) && InputData.get(id) != null)) || (this.InputIds.get(id)) != null) {
            System.out.println("Obtain data");
            if (sessions.size() != 0) {
                byte[] res;
                Object[] result = null;
                ResidualDataToDevice.put(id, new TreeMap<>());
//                System.out.println("ipgraph: " + cfg.ipGraph[0] + "," + cfg.ipGraph[1] + "," + cfg.ipGraph[2]);
                if (cfg.isHeader()) {
                    System.out.println("Inference on Master");
                    for (int i = 0;  i< sessions.size(); i++){
                        System.out.println("sessions.size() header"+sessions.size());
                        int[] to_send_seq_indices = Utils.JsonArray2IntArray(sendIndex.get(sessionIndex[i]).get(0).getJSONArray(String.valueOf(Integer.parseInt(sessionIndex[i]) + 1)));
                        if (i == 0) {
//                             只使用最后一个 token
                            ArrayList<Integer> inputIds = InputIds.get(id);
                            int[] currentToken = new int[]{inputIds.get(inputIds.size() - 1)};
                            result = ((Object[]) runInferenceMasterResidual(sessions.get(i), currentToken, to_send_seq_indices, getResIndices(i)));
//                            result = ((Object[]) runInferenceMasterResidual(sessions.get(i), Utils.convertArrayListToIntArray(InputIds.get(id)), to_send_seq_indices, getResIndices(i)));
                        } else {
                            result = ((Object[]) runInferenceWorkerResidual(sessions.get(i), OutputData.get(id), mergeResFromAndToDevice(id, sessionIndex[i]), to_send_seq_indices, getResIndices(i)));
                        }
                        convertOutput(id, i, result);
                    }
                }else if (cfg.isTailer()) {
                    System.out.println("Inference on Tail");
                    for (int i = 0;  i< sessions.size(); i++){
                        if (i == sessions.size() - 1) {
                            if (i == 0)

                                res = runInferenceWorkerResidualLastGeneration(sessions.get(i),
                                        InputData.get(id),
                                        mergeResFromAndToDevice(id, sessionIndex[i]),
                                        cfg.k,
                                        cfg.initial_temp);
                            else
                                res = runInferenceWorkerResidualLastGeneration(sessions.get(i),
                                        OutputData.get(id),
                                        mergeResFromAndToDevice(id, sessionIndex[i]),
                                        cfg.k,
                                        cfg.initial_temp);
                            OutputData.put(id, res);
                            break;
                        }else if (i == 0) {
                            result = ((Object[]) runInferenceWorkerResidual(sessions.get(i),InputData.get(id), mergeResFromAndToDevice(id, sessionIndex[i]), Utils.JsonArray2IntArray(sendIndex.get(sessionIndex[i]).get(0).getJSONArray(sessionIndex[i + 1])), getResIndices(i)));
                        } else {
                            result = ((Object[]) runInferenceWorkerResidual(sessions.get(i), OutputData.get(id), mergeResFromAndToDevice(id, sessionIndex[i]), Utils.JsonArray2IntArray(sendIndex.get(sessionIndex[i]).get(0).getJSONArray(sessionIndex[i + 1])), getResIndices(i)));
                        }
                        convertOutput(id, i, result);
                    }
                }else {
                    System.out.println("Inference on Worker");
                    for (int i = 0;  i< sessions.size(); i++){
                        int[] to_send_seq_indices = Utils.JsonArray2IntArray(sendIndex.get(sessionIndex[i]).get(0).getJSONArray(String.valueOf(Integer.parseInt(sessionIndex[i]) + 1)));
                        if (i == 0) {
                            result = ((Object[]) runInferenceWorkerResidual(sessions.get(i), InputData.get(id), mergeResFromAndToDevice(id, sessionIndex[i]), to_send_seq_indices, getResIndices(i)));
                        } else {
                            result = ((Object[]) runInferenceWorkerResidual(sessions.get(i), OutputData.get(id), mergeResFromAndToDevice(id, sessionIndex[i]), to_send_seq_indices, getResIndices(i)));
                        }

                        System.out.println("No." + id + " : "+result);
                        System.out.println("InputData." + id + " : "+ InputData.get(id));

                        convertOutput(id, i, result);
                    }
                }
                System.out.println("No." + id + " Inference is complete!");
                // add

                long eTime = System.nanoTime();

                System.out.println("No." + id + " Inference Time in seconds: " + (eTime - sTime) / 1000000000.0);
                Log.d(TAG, "No." + id + " Inference Time in seconds: " + (eTime - sTime) / 1000000000.0);

            }
        } else {
            System.out.println("Data missing");
        }
    }

    public void running(int corePoolSize, int maximumPoolSize, int keepAliveTime, ArrayList<String> input_data) throws Exception {
        while(!param.status.equals("Running")) {
//            Log.d(TAG, "param.status: " + param.status);
            Thread.sleep(1000);
        }

//        if (cfg.isHeader()) {
//            while(input_data.length == 0) {
////            Log.d(TAG, "param.status: " + param.status);
//                Thread.sleep(1000);
//            }
//            for (int i = 0; i < input_data.length; i++) {
//                System.out.println(input_data[i]);
//                int[] data = encodeString(input_data[i], tokenizer);
//                System.out.println(Arrays.toString(data));
//                this.InputIds.put(i, Utils.convertIntegerArrayToArrayList(data));
//            }
//        } else {
            // ignore all input_data
//            input_data = null;
//        }

        if (!cfg.isHeader())
            input_data = null;


//        System.out.println("Work here");

        // Design for testing datasets
        if (param.corePoolSize > 0) {
            corePoolSize = param.corePoolSize;
            maximumPoolSize = param.corePoolSize;
        }

//        CountDownLatch latch = new CountDownLatch(param.numSample);

        Semaphore latch = new Semaphore(param.corePoolSize);

        Lock socketLock = new ReentrantLock();

        LinkedBlockingQueue<Runnable> waitingQueue = new LinkedBlockingQueue<Runnable>();

        ThreadPoolExecutor pool = new ThreadPoolExecutor(corePoolSize,
                maximumPoolSize,
                keepAliveTime,
                TimeUnit.MILLISECONDS,
                waitingQueue,
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.AbortPolicy());

        updateSockets(corePoolSize);

        System.out.println("Load Balance On Running");
        Log.d(TAG, "Load Balance On Running");

        long startTime = System.nanoTime();

        while (true) {
            if (sampleId >= param.numSample) {
                break;
            }else{
                if (cfg.isHeader) {
                    while(sampleId >= input_data.size())
                        Thread.sleep(1000);
                    int[] data = encodeString(input_data.get(sampleId), tokenizer);
                    System.out.println(Arrays.toString(data));
                    this.InputIds.put(sampleId, Utils.convertIntegerArrayToArrayList(data));
                }
            }
            if (pool.getActiveCount() + waitingQueue.size() < corePoolSize) {
                System.out.println(!LB_Pause.condition);
                System.out.println(loadBalance.reSampleId == -1);
                if ((!LB_Pause.condition && loadBalance.reSampleId == -1 ) || sampleId < loadBalance.reSampleId) {
                    latch.acquire();
                    pool.execute(new multiSteps(sampleId, latch));
                    sampleId += 1;
                }else if (LB_Pause.condition){
                    System.out.println("resampleId " + loadBalance.reSampleId);
                    System.out.println("wait the Process to Finish");
                    System.out.println("Active Thread Count: " + (pool.getActiveCount() + waitingQueue.size()));
                    if ((pool.getActiveCount() + waitingQueue.size()) == 0 && (loadBalance.reSampleId != -1 && sampleId >= loadBalance.reSampleId)) {
                        // Launch re-load when no active process in the pool
                        System.out.println("===================== Load Balance =====================");
                        loadBalance.ModifySession();
                        loadBalance.reLoadBalance();
                        Communication.loadBalance.setReSampleId(-1);
                        LB_Pause.setConditionFalse();
                    }
                }
//                Thread.sleep(1000);
            }
//            Log.d(TAG, "****InputId array:" + InputIds.get(0));

//            if (sampleId >= param.numSample) {
////                int current_permit = latch.availablePermits();
//////                System.out.println("available permits " + latch.availablePermits());
////                System.out.println("Num of sample left" + (param.corePoolSize - latch.availablePermits()));
////                Log.d(TAG, "Num of sample left" + (param.corePoolSize - latch.availablePermits()));
////                while (true) {
////                    if (latch.availablePermits() < param.corePoolSize) {
////                        System.out.println("available permits " + latch.availablePermits());
////                        Log.d(TAG, "available permits " + latch.availablePermits());
////                        if (pool.getActiveCount() < param.corePoolSize) {
//////                            pool.execute(new multiSteps(param.numSample-1, latch));
////                            pool.execute(new EndProcess(cfg,this, latch));
////                        } else {
////                            Thread.sleep(1000);
////                        }
////                    } else {
////                        break;
////                    }
////                }
//                break;
//            }
        }

        Utils.await(latch, param.corePoolSize);


        long runningTime = System.nanoTime();
        System.out.println("Running Time in seconds: " + (runningTime - startTime) / 1000000000.0);
        Log.d(TAG, "Running Time in seconds: " + (runningTime - startTime) / 1000000000.0);
        timeUsage[1] = (runningTime - startTime) / 1000000000.0;

        // Do after processing
        param.status = "Finish";
        // Communicate with Root Server
//        beClient.communicationOpenClose(cfg, this, rootSocket, param, sockets);

        pool.shutdown();
        shutDownPrepare();

        // Print out results
        System.out.println("Prepare time is: " + timeUsage[0] + "seconds");
        System.out.println("Running time is: " + timeUsage[1] + "seconds");

        Log.d(TAG, "Prepare time is: " + timeUsage[0] + "seconds");
        Log.d(TAG, "Running time is: " + timeUsage[1] + "seconds");

        if (cfg.isHeader()) {
            assert Objects.requireNonNull(input_data).size() >= logits.size();
            assert Objects.requireNonNull(input_data).size() >= param.numSample;
            for (int i = 0; i < param.numSample; i++) {
                if ((param.max_length == 0) && (param.task_type.equals("classification"))) {
                    System.out.println("The result of sample " + i + ":" + this.param.classes[binaryClassify(logits.get(i))]);
                    Log.d(TAG, "The result of sample " + i + ":" + this.param.classes[binaryClassify(logits.get(i))]);
                } else {
                    System.out.println(InputIds.get(i));
                    String decoding_String = decodeID(Utils.convertArrayListToIntArray(Objects.requireNonNull(InputIds.get(i))), tokenizer);
                    System.out.println("Generated sequence:" + decoding_String);
                    Log.d(TAG, "Generated sequence:" + decoding_String);
                }
            }
        }
    }
    class loadModel implements Runnable{
        public void run(){
            //  设置当前线程的系统优先级为后台（可运行在安卓的小核）
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND);
            // 加载模型权重和tokenizer
            long startTime = System.nanoTime();
            Log.d(TAG, "load while inference");
            createSession("/data/user/0/com.example.distribute_ui/files" + "/device/module.onnx");

            System.out.println("load while inference Time: " + (System.nanoTime() - startTime) / 1000000000.0);

//                Communication.tokenizer = createHuggingFaceTokenizer("/data/user/0/com.example.distribute_ui/files" + "/device/tokenizer.json");
            // OR SENTENCEPIECE LATER





        }
    }
    class multiSteps implements Runnable {
        private Map<Integer, Socket> serverSocket;
        private Map<Integer, Socket> clientSocket;
        private final int sample_id;
        private final Semaphore latch;

        public multiSteps(int sample_id, Semaphore latch) {
            this.sample_id = sample_id;
            ArrayList<Map<Integer, Socket>> sockets = null;
            try {
                sockets = allSockets.take();
            } catch (InterruptedException e) {
                System.out.println("Waiting for an element from the sockets queue...");
                e.printStackTrace();
            }
//
            this.clientSocket = sockets.get(0);
            this.serverSocket = sockets.get(1);

            this.latch = latch;
        }
        @SuppressLint("SdCardPath")



        @Override
        public void run() {
            DataRepository.INSTANCE.updateSampleId(this.sample_id);
            Log.d(TAG, "Sample ID: "+this.sample_id);
            if (param.max_length < 0) {
                System.out.println("ERROR: Set up max_length");
            } else if (param.max_length == 0) {
                // classification
                System.out.println("++++++++++++SampleID: " + sample_id);
                int receivedId = 0;
                try {
                    receivedId = new OneStep(this.sample_id, serverSocket, clientSocket).run();
                } catch (InterruptedException | JSONException e) {
                    throw new RuntimeException(e);
                }
                cleanUpBuffer(receivedId);
            } else {
                // generation
                int receivedId = sampleId;
//                param.max_length=100;
                int input_size = param.max_length;
//                loadModel
//

                loadModel loadthread = new loadModel();
                Thread thread = new Thread(loadthread);

// 设置线程优先级为后台（小核）
                thread.setPriority(Thread.MIN_PRIORITY);  // 这个是标准Java线程的优先级
                thread.start();


                for (int m = 0; m < param.max_length; m++) {
                    long startTime = System.nanoTime();
                    System.out.println("++++++++++++SampleID: " + sample_id + "++++++++++TokenID:" + m);
                    try {


                        receivedId = new OneStep(this.sample_id, serverSocket, clientSocket).run();

                        if (cfg.isHeader()) {
                            // Synchronize the decoded string in data-repo for UI - Junchen
                            input_size = Math.min(input_size, InputIds.get(receivedId).size());
                            System.out.println(input_size);
                            System.out.println(InputIds.get(receivedId).size());
                            ArrayList<Integer> decodeList = new ArrayList(InputIds.get(receivedId).subList(input_size-1, InputIds.get(receivedId).size()));
                            String decodedString = decodeID(Utils.convertArrayListToIntArray(
                                    Objects.requireNonNull(decodeList)), tokenizer);
                            DataRepository.INSTANCE.updateDecodingString(decodedString);
                            System.out.println("No." + receivedId + " Results Obtained");
                        }

                    } catch (InterruptedException | JSONException e) {
                        throw new RuntimeException(e);
                    }
//
////                    // Break for </s> token
//                    if (receivedId == -1){
//                        break;
//                    }

                    System.out.println("Token Process Time: " + (System.nanoTime() - startTime) / 1000000000.0);
                }
                System.out.println("Sum"  + " Infer Time: " +infertimeArraySum);
                System.out.println("Sum"  + " Commu Time: " +commutimeArraySum.stream().mapToDouble(Double::doubleValue).sum());
                System.out.println("Sum"  + " Infer Time: " +infertimeArraySum.stream().mapToDouble(Double::doubleValue).sum());
//                System.out.println("Sum."  + " Commu msg: " +byteArraySum);
                System.out.println("Sum"  + " Commu msg size: " +byteArraySum.size());


                byteArraySum.clear();
                commutimeArraySum.clear();
                infertimeArraySum.clear();
//                System.out.println("Sum"  + " Commu Time: " +commutimeArraySum);


                cleanUpBuffer(this.sample_id);
            }

            try {
//                serverSockets.put(serverSocket);
//                clientSockets.put(clientSocket);
                allSockets.put(new ArrayList<Map<Integer, Socket>>(){{
                    add(clientSocket);
                    add(serverSocket);
                }});
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            latch.release();
        }
    }

    public class OneStep {
        // Todo, use thread for entire receive-process-send or use thread for each procedure
        //  One step includes, receiving data, compute data and send data
        private final Map<Integer, Socket>  serverSocketMap;
        private final Map<Integer, Socket>  clientSocketMap;
        private final Socket  serverSocket;
        private final Socket  clientSocket;
        private final int sample_id;

        private int current_token_index;
        byte[] msgAll ;



        public OneStep(int sample_id, Map<Integer, Socket> serverSide, Map<Integer, Socket>  clientSide) {
            this.sample_id = sample_id;
            this.serverSocketMap = serverSide;
            this.clientSocketMap = clientSide;
            this.serverSocket = serverSide.get(cfg.prevDeviceId());
            this.clientSocket = clientSide.get(cfg.nextDeviceId());
        }



        public int procssingAsClient(int receivedId) throws InterruptedException {
            if (!cfg.isHeader()) {
                System.out.println("Start to be a Client");
//                Communication.LB_Pause.waitForCondition();
                // worker 作为client 先发送请求然后等待 header作为server返回outputdata

                serverSocket.send("Request Data");


                // 通信信息 1


                byte[] msgFrom1 = serverSocket.recv(0);

//                addBytes(msgFrom1);
                System.out.println("通信 1: " + msgFrom1);

                receivedId = Utils.convertByteArrayToInt(msgFrom1);
                System.out.println("ReceiveID: " + receivedId);

//                if (receivedId == -1)
//                    return -1;

                Thread workerThread = new Thread(new ReceiveResidualConnection(receivedId, serverSocketMap));
                workerThread.start();

                if (receivedId != this.sample_id) {
                    System.out.println("Client: Data out of the order, sampleId: " + this.sample_id + ", receivedId: " + receivedId);
                }
                // 通信信息 2
                byte[] msgFrom = serverSocket.recv(0);

//                addBytes(msgFrom);
                System.out.println("通信 2: " + msgFrom);
                InputData.put(receivedId, msgFrom);
                workerThread.join();
                System.out.println("Received Data");
                System.out.println(InputData.get(receivedId));
                System.out.println("InputData:"+ Arrays.toString(InputData.get(receivedId)));

            } else {
                // load data from the local
                if (logits.get(receivedId) == null) {
                    System.out.println("Load Data");
                }
//                System.out.println("last token:");
//                System.out.println(InputIds.get(receivedId).get(InputIds.get(receivedId).size()-1));
//                if (InputIds.get(receivedId).get(InputIds.get(receivedId).size()-1) == 2){
//                    return -1;
//                }
            }
            return receivedId; // Either comes from received id or direct sample id
        }

        public void processAsServer(int receivedId) throws InterruptedException {
            // return data to the header machine;
            if (clientSocket == null)
                System.out.println("ProcessAsServer Error");

            System.out.println("Start to be a Server");
            // 这是什么？貌似只是传输标号

            // 通信信息 3

            byte[] comefrom_id = clientSocket.recv(0);
//            addBytes(comefrom_id);
            System.out.println("通信 3: " + comefrom_id);
            System.out.println(new String(comefrom_id));
            // 通信信息 4
            byte[] msgTo = clientSocket.recv(0);
//            addBytes(msgTo);

            System.out.println("通信 4: " + msgTo);
            System.out.println(new String(msgTo));
            System.out.println("Thread id " + this.sample_id);
            System.out.println(receivedId);
//            System.out.println(OutputData);
            System.out.println("No. " + receivedId + " " + new String(msgTo));
            if (new String(msgTo).contains("Request Data")) {
                if (OutputData.containsKey(receivedId)) {
                    byte[] id = "from".getBytes();
                    System.out.println("No. " + receivedId + "thread开始建立" );
                    Thread workerThread = new Thread(new SendResidualConnection(receivedId, clientSocketMap));
                    workerThread.start();
                    System.out.println("No. " + receivedId + "thread建立成功" );
                    id = Utils.convertIntToByteArray(receivedId);
                    clientSocket.sendMore(comefrom_id);
                    clientSocket.sendMore(id);
//                    System.out.println("No. " + receivedId + " " + new String(msgTo));
                    if (cfg.isTailer() && (param.task_type.equals("generation"))) {
                        byte[] decode_id = OutputData.get(receivedId);
                        System.out.println("No. " + receivedId + "worker开始发送" );
                        clientSocket.send(decode_id, 0);
                        System.out.println("No. " + receivedId + "worker发送结束" );

                    }
                    else {
                        System.out.println("No. " + receivedId + "header开始发送数据" );
                        clientSocket.send(OutputData.get(receivedId), 0);
                        System.out.println("No. " + receivedId + "header结束发送数据" );
                        System.out.println("OutputData:"+OutputData.get(receivedId));
//                        System.out.println("OutputData:"+ Arrays.toString(OutputData.get(receivedId)));
//                        System.out.println("OutputData:"+ new String(OutputData.get(receivedId)));
                    }
//                    workerThread.join();
                } else {
//                    if (receivedId == -1){
//                        byte[] id = Utils.convertIntToByteArray(receivedId);
//                        clientSocket.sendMore(comefrom_id);
//                        clientSocket.send(id);
//                        return;
//                    }
                    System.out.println(receivedId + " is not in the OutputData");
                }
            }
        }

        public int obtainResultsFromTailer(int receivedId) {
            // Special for header to obtain results from tailer
            if (cfg.isHeader()) {
                // Handle the case header to request tailer results
                serverSocket.send("Request Data");
                System.out.println("No. " + receivedId + "header发送了" );
                receivedId = Utils.convertByteArrayToInt(serverSocket.recv(0));

//                if (receivedId == -1)
//                    return receivedId;

                if (receivedId != this.sample_id) {
                    System.out.println("Server: Data out of the order, sampleId: " + this.sample_id + ", receivedId: " + receivedId);
                }
                byte[] res = serverSocket.recv(0);
                System.out.println("No. " + receivedId + "header接收了" );
                if (param.task_type.equals("generation")) {
//                    int decode_id = Utils.convertByteArrayToInt(res);
//                    InputIds.get(receivedId).add(decode_id);
                    System.out.println("No. " + receivedId + "开始解析" );
                    int decode_id = deserializeInt(res);
                    System.out.println("No. " + receivedId + "解析完成" );
//                    int decode_id = 0;
                    InputIds.get(receivedId).add(decode_id);
                } else {
                    logits.put(receivedId, res);
                }
            }
            return receivedId;
        }


        public int run() throws RuntimeException, InterruptedException, JSONException {
            int receivedId = this.sample_id;

            try {
                // 通信时间 1
                long startTime = System.nanoTime();
                System.out.println("No." + receivedId + "commu 1 start");
                receivedId = procssingAsClient(receivedId);
                System.out.println("No." + receivedId + "commu 1 end");
                double commutime1 = (System.nanoTime() - startTime) / 1000000000.0;
                System.out.println("No." + receivedId + " commutime1: " + commutime1);

                commutimeArraySum.add(commutime1);

                // 推理时间
                startTime = System.nanoTime();
                System.out.println("No." + receivedId + "infer  start");
                inferenceProcedure(receivedId);
                System.out.println("No." + receivedId + "infer  end");
                double infertime = (System.nanoTime() - startTime) / 1000000000.0;
                infertimeArraySum.add(infertime);

                // 通信时间 2
                startTime = System.nanoTime();
                System.out.println("No." + receivedId + "commu 2 start");
                processAsServer(receivedId);
                System.out.println("No." + receivedId + "commu 2 end");
                double commutime2 = (System.nanoTime() - startTime) / 1000000000.0;
                System.out.println("No." + receivedId + " commutime2: " + commutime2);
                commutimeArraySum.add(commutime2);

                // 通信时间 3
                startTime = System.nanoTime();
                System.out.println("No." + receivedId + "commu 3 start");
                receivedId = obtainResultsFromTailer(receivedId);
                System.out.println("No." + receivedId + "commu 3 end");
                double commutime3 = (System.nanoTime() - startTime) / 1000000000.0;
                System.out.println("No." + receivedId + " commutime3: " + commutime3);
                commutimeArraySum.add(commutime3);

                System.out.println("No." + receivedId + " Commu Time: " + (commutime1 + commutime2 + commutime3));
                System.out.println("No." + receivedId + " Infer Time: " + infertime);

                return receivedId;

            } catch (JSONException e) {
                System.err.println("JSON parsing error occurred: " + e.getMessage());
                e.printStackTrace();
                // Handle specific JSONException recovery
            } catch (InterruptedException e) {
                System.err.println("Operation was interrupted: " + e.getMessage());
                e.printStackTrace();
                // Handle specific InterruptedException recovery
            } catch (RuntimeException e) {
                System.err.println("Runtime exception occurred: " + e.getMessage());
                e.printStackTrace();
                // Handle unexpected runtime exceptions
            } catch (Exception e) {
                System.err.println("Unexpected exception occurred: " + e.getMessage());
                e.printStackTrace();
                // Handle general exceptions
            } catch (Error e) {
                System.err.println("Serious error occurred: " + e.getMessage());
                e.printStackTrace();
                // Handle critical errors
            } finally {
                // Ensure any necessary cleanup happens here
                System.out.println("Completed processing for receivedId: " + receivedId);
            }

            return receivedId;
        }

    }

    public void updateSockets(int corePoolSize) throws InterruptedException {
        int j = cfg.ipGraph.length;
        System.out.println("Graph length: " + j);
        for (int i = 0; i < corePoolSize; i++) {
            ArrayList<Map<Integer, Socket>> socketContainer = new ArrayList<>();
            Map<Integer, Socket> SendSocket = new HashMap<>();
            for (Integer idx : sendDeviceIndex) {
                Socket temp = beServer.establish_connection(context, SocketType.ROUTER, Config.port + j * i + (idx - cfg.deviceId));
                temp.setIdentity(("ROUTER Send From " + cfg.deviceId + " to " + idx + "." + (Config.port + j * i + (idx - cfg.deviceId))).getBytes());
                SendSocket.put(idx, temp);
            }

            if (cfg.isTailer()){
                Socket temp = beServer.establish_connection(context, SocketType.ROUTER, Config.port + j*i + 1);
                temp.setIdentity(("ROUTER Send From " + cfg.deviceId + " to " + cfg.nextDeviceId() + "." + (Config.port + j*i + 1)).getBytes());
                SendSocket.put(cfg.nextDeviceId(), temp);
            }
//            clientSockets.put(SendSocket);
            socketContainer.add(SendSocket);

            Map<Integer, Socket> receiveSocket = new HashMap<>();
            for (Integer idx : receiveDeviceIndex) {
                Socket temp = beClient.establish_connection(context, SocketType.DEALER, Config.port + j*i + (cfg.deviceId-idx), cfg.ipGraph[idx]);
                temp.setIdentity(("DEALER Receive From: " + cfg.deviceId + " to " + idx + "." + (Config.port + j*i + (cfg.deviceId-idx))).getBytes());
                receiveSocket.put(idx, temp);
            }

            if (cfg.isHeader()){
                Socket temp = beClient.establish_connection(context, SocketType.DEALER, Config.port + j*i + 1, cfg.prevNodes.get(0));
                temp.setIdentity(("DEALER Receive From: " + cfg.deviceId + " to " + cfg.nextDeviceId() + "." + (Config.port + j*i +1)).getBytes());
                receiveSocket.put(cfg.prevDeviceId(), temp);
            }

//            serverSockets.put(receiveSocket);
            socketContainer.add(receiveSocket);

            allSockets.put(socketContainer);
        }

        System.out.println("Sockets are build successfully");
    }

    public void getSendResDevice2Device(){
        sendD2D =  new TreeMap<>();
        for (ArrayList<JSONObject> sendIndexList : sendIndex.values()) {
            if (sendIndexList.size() > 1) {
                JSONObject sendResIndex = sendIndexList.get(1); // Obtain res Index
                Iterator<String> keys = sendResIndex.keys();
                while (keys.hasNext()) {
                    String k = keys.next();
                    int device = module_on_devices.get(k);
                    if (device != cfg.deviceId) {
                        if (!sendD2D.containsKey(device))
                            sendD2D.put(device, new ArrayList<>());
                        sendD2D.get(device).add(k);
                    }
                }
            }
        }
        // Sort in case disorder
        for (List<String> i : sendD2D.values())
            Collections.sort(i);
    }

    public void getReceiveResDevice2Device(){
        receiveD2D =  new TreeMap<>();
        for (Map.Entry<String, ArrayList<JSONObject>> receiveIndexList : receiveIndex.entrySet()) {
            if (receiveIndexList.getValue().size() > 1) {
                JSONObject receiveResIndex = receiveIndexList.getValue().get(1); // Obtain res Index
                Iterator<String> keys = receiveResIndex.keys();
                while (keys.hasNext()) {
                    String k = keys.next();
                    int device = module_on_devices.get(k);
                    if (device != cfg.deviceId) {
                        if (!receiveD2D.containsKey(device))
                            receiveD2D.put(device, new ArrayList<>());
                        receiveD2D.get(device).add(receiveIndexList.getKey());
                    }
                }
            }
        }
        // Sort if disorder
        for (List<String> i : receiveD2D.values())
            Collections.sort(i);
    }


    class SendResidualConnection implements Runnable {
        int receiveId;
        Map<Integer, Socket> clientSide;
        public SendResidualConnection(int receiveId, Map<Integer, Socket> clientSide) {
            this.receiveId = receiveId;
            this.clientSide = clientSide;
        }

        @Override
        public void run() {

            for (Map.Entry<Integer, ArrayList<String>> entry : sendD2D.entrySet()) {
                int target_device_id = entry.getKey();
                System.out.println("Send to device "+ target_device_id);
                Socket sendSocket = this.clientSide.get(target_device_id);
                System.out.println(new String(sendSocket.getIdentity()));
                byte[] comefrom_id = sendSocket.recv(0);
                int target_id = Utils.convertByteArrayToInt(sendSocket.recv(0));
                assert target_id == target_device_id;
                byte[] msgTo = sendSocket.recv(0);
                System.out.println(new String(msgTo));
                if (new String(msgTo).contains("Request Res Data")) {
                    sendSocket.sendMore(comefrom_id);
                    sendSocket.sendMore(Utils.convertIntToByteArray(cfg.deviceId));
                    System.out.println("Target Device ID: " + target_id);
                    List<String> sendByte = entry.getValue();
                    for (String k : sendByte) {
                        ArrayList<byte[]> data = ResidualDataToDevice.get(receiveId).get(k);
                        for (byte[] i : data)
                            sendSocket.sendMore(i);
                        sendSocket.sendMore(";");
                    }
                    sendSocket.send("Over");
                }
                System.out.println("Send the Residual Data to Device " + entry.getKey());
            }
        }
    }


    class ReceiveResidualConnection implements Runnable {
        int receiveId;
        Map<Integer, Socket> serverSide;

        public ReceiveResidualConnection(int receiveId, Map<Integer, Socket> serverSide) {
            this.receiveId = receiveId;
            this.serverSide = serverSide;
        }

        @Override
        public void run() {
//            Map<Integer, List<String>> tmp = new TreeMap<>();
//            for (Map.Entry<String, ArrayList<JSONObject>> receiveIndexList : receiveIndex.entrySet()) {
//                if (receiveIndexList.getValue().size() > 1) {
//                    JSONObject receiveResIndex = receiveIndexList.getValue().get(1); // Obtain res Index
//                    Iterator<String> keys = receiveResIndex.keys();
//                    while (keys.hasNext()) {
//                        String k = keys.next();
//                        int device = module_on_devices.get(k);
//                        if (device != cfg.deviceId) {
//                            if (!tmp.containsKey(device))
//                                tmp.put(device, new ArrayList<>());
//                            tmp.get(device).add(receiveIndexList.getKey());
//                        }
//                    }
//                }
//            }
//            // Sort if disorder
//            for (List<String> i : tmp.values())
//                Collections.sort(i);
//
//            // Empty ResidualDataFromDevice receiveID and initial res receive each round
//            if (!ResidualDataFromDevice.containsKey(receiveId))
//                ResidualDataFromDevice.put(receiveId, new TreeMap<>());
//
//            for (List<String> ks : tmp.values()){
//                for (String k: ks) {
//                    if (ResidualDataFromDevice.get(receiveId).containsKey(k))
//                        ResidualDataFromDevice.get(receiveId).put(k, new ArrayList<>());
//                }
//            }
            receiveIndex.keySet();
            for (Map.Entry<Integer, ArrayList<String>> entry : receiveD2D.entrySet()) {
                Socket receiveSocket = serverSide.get(entry.getKey());
                System.out.println(new String(receiveSocket.getIdentity()));
                receiveSocket.sendMore(Utils.convertIntToByteArray(cfg.deviceId));
                receiveSocket.send("Request Res Data");
                int send_device_id = Utils.convertByteArrayToInt(receiveSocket.recv(0));
                System.out.println("Actual Receive the Residual Data from Device " + send_device_id);

                int i = 0;
                List<String> keyOnDevices = entry.getValue();
                Map<String, ArrayList<byte[]>> tmpReceiver = ResidualDataFromDevice.get(receiveId);
//                if (!tmpReceiver.containsKey(keyOnDevices.get(i)))
//                    tmpReceiver.put(keyOnDevices.get(i), new ArrayList<>());
                if (tmpReceiver == null){
                    tmpReceiver = new TreeMap<>();
                    ResidualDataFromDevice.put(receiveId, tmpReceiver);
                }
                tmpReceiver.put(keyOnDevices.get(i), new ArrayList<>());

                while (true) {
                    byte[] data = receiveSocket.recv(0);
                    if (new String(data).equals("Over")) {
                        break;
                    }else if (new String(data).equals(";")) {
                        i += 1;
                        if (keyOnDevices.size() > i && !tmpReceiver.containsKey(keyOnDevices.get(i)))
                            tmpReceiver.put(keyOnDevices.get(i), new ArrayList<>());
                    }else {
                        tmpReceiver.get(keyOnDevices.get(i)).add(data);
                    }
                }

                System.out.println("Receive the Residual Data from Device " + entry.getKey());
                System.out.println(this.receiveId + " With the idx and size ");
            }
        }
    }


    public Socket getSocketsInQueue(LinkedBlockingQueue<Socket> queue, String identity) {
//        Byte[] identity = ("server: " + Config.local + "."+ 1).getBytes();
        Iterator<Socket> iterator = queue.iterator();
        while (iterator.hasNext()) {
            Socket item = iterator.next();
            if (Arrays.toString(item.getIdentity()).equals(identity)) {
                iterator.remove();  // Remove the current item
                return item;
            }
        }
        return null;
    }

    //    public native byte[] performInferenceMaster(long session, String input_string, long tokenizer);

    public native int tensorSizeDebug(byte[] logits);
    public native byte[] performInferenceMaster(long session, int[] input_ids);
    public native byte[] performInferenceWorker(long session, byte[] data);
    public native int binaryClassify(byte[] data);
    public native int[] encodeString(String input_string, long tokenizer);
    public native int greedyDecoding(byte[] data);
    public native String decodeID(int[] data, long tokenizer);

    public native Double modelFlopsPerSecond(int modelFlops, long session, int[] input_ids_j);
    public native Object runInferenceMasterResidual(long session, int[] input_ids_j, int[] to_send_seq_indices, int[][] to_send_res_indices);
    public native Object runInferenceWorkerResidual(long session,  byte[] sequential_input, ArrayList<byte[]> residual_input, int[] to_send_seq_indices, int[][] to_send_res_indices);
    public native byte[] runInferenceWorkerResidualLast(long session, byte[] sequential_input, ArrayList<byte[]>  residual_input);

    public native byte[] runInferenceWorkerResidualLastGeneration(long session,
                                                                  byte[] sequential_input,
                                                                  ArrayList<byte[]>  residual_input,
                                                                  int k,
                                                                  float init_temp);

    public native byte[] runInferenceWorkerResidualLastClassification(long session, byte[] sequential_input, ArrayList<byte[]>  residual_input);

    public native int deserializeInt(byte[] decode_id);

    public native int TokenToID(String token, long tokenizer);

    public native boolean EosCheck(byte[] output, long tokenizer); // TODO: adding EOS string check for generation early stopping - Junchen 02/28/2024

//    public native OnnxValue[] DeserializeTensor(byte[] data);
    public static native long createSession(String inference_model_path);
    public native long createHuggingFaceTokenizer(String tokenizer_path);
}

