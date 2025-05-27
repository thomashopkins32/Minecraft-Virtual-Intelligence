package com.mvi.mvimod;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import net.minecraft.world.level.gameevent.GameEvent;

public class DataBridge {
    private static DataBridge instance;
    private NetworkHandler networkHandler;
    private Queue<GameEvent> eventQueue = new ConcurrentLinkedQueue<>();
    private byte[] latestFrame;
    
    public static DataBridge getInstance() {
        if (instance == null) {
            instance = new DataBridge();
        }
        return instance;
    }
    
    public void setNetworkHandler(NetworkHandler handler) {
        this.networkHandler = handler;
    }
    
    public void sendFrame(byte[] frameData) {
        this.latestFrame = frameData;
        if (networkHandler != null) {
            networkHandler.sendFrameToClient(frameData);
        }
    }
    
    public void sendEvent(String eventType, String data) {
        GameEvent event = new GameEvent(eventType, data);
        eventQueue.offer(event);
        if (networkHandler != null) {
            networkHandler.sendEventToClient(event);
        }
    } 
}
