package com.mvi.mvimod;

import com.mojang.logging.LogUtils;
import org.slf4j.Logger;

public class DataBridge {
  private static final Logger LOGGER = LogUtils.getLogger();
  private static DataBridge instance;
  private NetworkHandler networkHandler;

  public static DataBridge getInstance() {
    if (instance == null) {
      instance = new DataBridge();
      LOGGER.info("DataBridge instance created");
    }
    return instance;
  }

  public void setNetworkHandler(NetworkHandler handler) {
    this.networkHandler = handler;
    LOGGER.info("NetworkHandler connected to DataBridge");
  }

  public void sendObservation(Observation obs) {
    if (networkHandler != null) {
      LOGGER.info("DataBridge sending frame data (size: {} bytes)", obs.frame().length);
      networkHandler.setLatest(obs.frame(), obs.reward());
    } else {
      LOGGER.warn("Cannot send frame - NetworkHandler is null");
    }
  }
}
