package com.mvi.mvimod;

import com.mojang.logging.LogUtils;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import org.slf4j.Logger;

public class NetworkHandler implements Runnable {
  private static final Logger LOGGER = LogUtils.getLogger();
  private ServerSocket serverSocket;
  private Socket clientSocket;
  private boolean running = true;
  private int port;

  public NetworkHandler(int port) {
    this.port = port;
  }

  @Override
  public void run() {
    try {
      this.serverSocket = new ServerSocket(this.port);
      LOGGER.info("Network server started on port " + this.port);

      while (this.running && !Thread.currentThread().isInterrupted()) {
        this.clientSocket = this.serverSocket.accept();
        LOGGER.info("Client connected");
        this.handleClient();
      }
    } catch (IOException e) {
      LOGGER.error("Error starting network server", e);
    } finally {
      this.cleanup();
    }
  }

  private void handleClient() {
    try {
      BufferedReader in =
          new BufferedReader(new InputStreamReader(this.clientSocket.getInputStream()));
      PrintWriter out = new PrintWriter(this.clientSocket.getOutputStream(), true);

      String inputLine;
      while ((inputLine = in.readLine()) != null && this.running) {
        this.processCommand(inputLine, out);
      }
    } catch (IOException e) {
      LOGGER.error("Error handling client", e);
    }
  }

  public void sendResponse(byte[] frameData, int reward) {
    try {
      PrintWriter out = new PrintWriter(this.clientSocket.getOutputStream(), true);
      out.println(frameData.length);
      out.println(reward);
      out.println(frameData);
      out.flush();
    } catch (IOException e) {
      LOGGER.error("Error sending response", e);
    }
  }

  private void processCommand(String command, PrintWriter out) {
    LOGGER.info("Received command: " + command);
  }

  private void cleanup() {
    try {
      if (this.clientSocket != null) this.clientSocket.close();
      if (this.serverSocket != null) this.serverSocket.close();
    } catch (IOException e) {
      LOGGER.error("Cleanup error: " + e.getMessage());
    }
  }
}
