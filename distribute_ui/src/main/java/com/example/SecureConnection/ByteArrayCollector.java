package com.example.SecureConnection;

import java.util.ArrayList;

public class ByteArrayCollector {
    private ArrayList<Byte> byteArray = new ArrayList<>();

    public void addBytes(byte[] bytes) {
        for (byte b : bytes) {
            byteArray.add(b);
        }
    }
    
}
