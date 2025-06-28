package com.mvi.mvimod;

public class DataBridge {
  private static DataBridge instance;
  private NetworkHandler networkHandler;

  public static DataBridge getInstance() {
    if (instance == null) {
      instance = new DataBridge();
    }
    return instance;
  }

  public void setNetworkHandler(NetworkHandler handler) {
    this.networkHandler = handler;
  }

  public void sendEvent(String eventType, String data) {
    if (networkHandler != null) {
      // TODO: Implement this
    }
  }

  public void sendFrame(byte[] frameData) {
    if (networkHandler != null) {
      networkHandler.sendResponse(frameData, 0);
    }
  }
}
