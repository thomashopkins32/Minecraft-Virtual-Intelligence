package com.mvi.mvimod;

import com.mojang.logging.LogUtils;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;

public class NetworkHandler implements Runnable {
  private static final Logger LOGGER = LogUtils.getLogger();
  private ServerSocket readServerSocket;
  private DatagramSocket writeSocket;
  private final AtomicBoolean running = new AtomicBoolean(true);
  private int writePort;
  private int readPort;
  private InetAddress clientAddress;
  private int clientPort;

  // Async frame sending
  private final BlockingQueue<FrameData> frameQueue = new LinkedBlockingQueue<>();
  private final ExecutorService frameExecutor =
      Executors.newSingleThreadExecutor(
          r -> {
            Thread t = new Thread(r, "FrameSender");
            t.setDaemon(true);
            return t;
          });

  // Async command handling
  private final ExecutorService commandExecutor =
      Executors.newCachedThreadPool(
          r -> {
            Thread t = new Thread(r, "CommandHandler");
            t.setDaemon(true);
            return t;
          });

  // Frame data container
  private static class FrameData {
    final byte[] data;
    final int reward;
    final long timestamp;

    FrameData(byte[] data, int reward) {
      this.data = data;
      this.reward = reward;
      this.timestamp = System.currentTimeMillis();
    }
  }

  public NetworkHandler(int readPort, int writePort) {
    this.readPort = readPort;
    this.writePort = writePort;
  }

  @Override
  public void run() {
    try {
      this.readServerSocket = new ServerSocket(this.readPort);
      this.writeSocket = new DatagramSocket(); // Don't bind to specific port when only sending
      LOGGER.info("TCP server started on port " + this.readPort);
      LOGGER.info("UDP sender socket created (will send to client on port " + this.writePort + ")");

      // Start async frame sender
      startFrameSender();

      // Accept client connections asynchronously
      while (this.running.get() && !Thread.currentThread().isInterrupted()) {
        Socket clientSocket = this.readServerSocket.accept();
        LOGGER.info("Client connected: " + clientSocket.getInetAddress());
        this.clientAddress = clientSocket.getInetAddress();
        this.clientPort = this.writePort;

        // Handle each client asynchronously
        handleClientAsync(clientSocket);
      }
    } catch (IOException e) {
      LOGGER.error("Error starting network server", e);
    } finally {
      this.cleanup();
    }
  }

  private void handleClientAsync(Socket clientSocket) {
    commandExecutor.submit(
        () -> {
          try (BufferedReader in =
                  new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
              PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true)) {

            String inputLine;
            while ((inputLine = in.readLine()) != null && this.running.get()) {
              final String command = inputLine;
              // Process each command asynchronously to avoid blocking
              CompletableFuture.runAsync(() -> processCommand(command, out), commandExecutor)
                  .exceptionally(
                      throwable -> {
                        LOGGER.error("Error processing command: " + command, throwable);
                        return null;
                      });
            }
          } catch (IOException e) {
            LOGGER.error("Error handling client", e);
          } finally {
            try {
              clientSocket.close();
            } catch (IOException e) {
              LOGGER.error("Error closing client socket", e);
            }
          }
        });
  }

  private void startFrameSender() {
    frameExecutor.submit(
        () -> {
          while (this.running.get()) {
            try {
              FrameData frameData = frameQueue.take(); // Blocks until frame available
              sendFrameImmediate(frameData);
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
              break;
            }
          }
        });
  }

  public void sendResponse(byte[] frameData, int reward) {
    // Non-blocking: just add to queue
    if (this.clientAddress != null) {
      FrameData frame = new FrameData(frameData, reward);

      // Drop oldest frames if queue is getting too full (backpressure handling)
      while (frameQueue.size() > 10) {
        FrameData dropped = frameQueue.poll();
        if (dropped != null) {
          LOGGER.debug(
              "Dropped frame due to backpressure (age: {}ms)",
              System.currentTimeMillis() - dropped.timestamp);
        }
      }

      LOGGER.info("Adding frame to queue");
      frameQueue.offer(frame);
    }
  }

  private void sendFrameImmediate(FrameData frameData) {
    LOGGER.info("Sending frame");
    try {
      ByteBuffer buffer = ByteBuffer.allocate(8 + frameData.data.length);
      buffer.putInt(frameData.reward);
      buffer.putInt(frameData.data.length);
      buffer.put(frameData.data);
      byte[] packet = buffer.array();

      DatagramPacket udpPacket =
          new DatagramPacket(packet, packet.length, this.clientAddress, this.clientPort);
      this.writeSocket.send(udpPacket);
    } catch (IOException e) {
      LOGGER.error("Error sending frame", e);
    }
  }

  private void processCommand(String command, PrintWriter out) {
    LOGGER.info("Received command: " + command);
  }

  private void cleanup() {
    LOGGER.info("Shutting down NetworkHandler...");
    this.running.set(false);

    // Shutdown executors
    frameExecutor.shutdown();
    commandExecutor.shutdown();

    try {
      if (this.readServerSocket != null) this.readServerSocket.close();
      if (this.writeSocket != null) this.writeSocket.close();
    } catch (IOException e) {
      LOGGER.error("Cleanup error: " + e.getMessage());
    }

    LOGGER.info("NetworkHandler shutdown complete");
  }
}
