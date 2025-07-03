package com.mvi.mvimod;

import com.mojang.logging.LogUtils;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.channels.SocketChannel;
import java.nio.ByteBuffer;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import java.net.UnixDomainSocketAddress;
import java.nio.channels.ServerSocketChannel;
import java.net.StandardProtocolFamily;

public class NetworkHandler implements Runnable {
  private static final Logger LOGGER = LogUtils.getLogger();
  private static final String SEND_SOCKET_PATH = "/tmp/mvi_send.sock";
  private static final String RECEIVE_SOCKET_PATH = "/tmp/mvi_receive.sock";
  private static final ExecutorService senderExecutor = Executors.newCachedThreadPool();
  private static final ExecutorService receiverExecutor = Executors.newCachedThreadPool();
  private Thread sendThread;
  private Thread receiveThread;
  private ServerSocketChannel sendSocketChannel;
  private ServerSocketChannel receiveSocketChannel;
  private final AtomicBoolean running = new AtomicBoolean(true);

  // Async frame sending
  private final BlockingQueue<FrameData> frameQueue = new LinkedBlockingQueue<>();

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

  @Override
  public void run() {
    try {
      Files.deleteIfExists(Path.of(SEND_SOCKET_PATH));
      Files.deleteIfExists(Path.of(RECEIVE_SOCKET_PATH));

      sendSocketChannel = ServerSocketChannel.open(StandardProtocolFamily.UNIX);
      sendSocketChannel.bind(UnixDomainSocketAddress.of(SEND_SOCKET_PATH));
      sendSocketChannel.configureBlocking(true);

      receiveSocketChannel = ServerSocketChannel.open(StandardProtocolFamily.UNIX);
      receiveSocketChannel.bind(UnixDomainSocketAddress.of(RECEIVE_SOCKET_PATH));
      receiveSocketChannel.configureBlocking(true);

      LOGGER.info("Socket files created: {} and {}", SEND_SOCKET_PATH, RECEIVE_SOCKET_PATH);

      // Verify socket files were actually created
      if (!Files.exists(Path.of(SEND_SOCKET_PATH))) {
        throw new IOException("Failed to create send socket file: " + SEND_SOCKET_PATH);
      }
      if (!Files.exists(Path.of(RECEIVE_SOCKET_PATH))) {
        throw new IOException("Failed to create receive socket file: " + RECEIVE_SOCKET_PATH);
      }
      
      LOGGER.info("Socket files verified - Send: {}, Receive: {}", 
                  Files.exists(Path.of(SEND_SOCKET_PATH)), 
                  Files.exists(Path.of(RECEIVE_SOCKET_PATH)));

      sendThread = new Thread(this::acceptSendClients, "SendClients");
      receiveThread = new Thread(this::acceptReceiveClients, "ReceiveClients");
      
      sendThread.start();
      receiveThread.start();

      // Keep main thread alive while server is running
      while (this.running.get() && !Thread.currentThread().isInterrupted()) {
        try {
          Thread.sleep(1000); // Check every second
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          break;
        }
      }

    } catch (IOException e) {
      LOGGER.error("Error starting network server", e);
    } finally {
      this.cleanup();
    }
  }

  private void acceptSendClients() {
    LOGGER.info("Send clients acceptor thread started");
    while (this.running.get() && !Thread.currentThread().isInterrupted()) {
      try {
        SocketChannel clientSocket = sendSocketChannel.accept();
        clientSocket.socket().setSendBufferSize(1024 * 1024); // 1MB send buffer
        clientSocket.socket().setTcpNoDelay(true); // Disable Nagle's algorithm for lower latency
        LOGGER.info("Send client connected: " + clientSocket.getRemoteAddress());
        handleSendClient(clientSocket);
      } catch (IOException e) {
        if (this.running.get()) {
          LOGGER.error("Error accepting send client", e);
        } else {
          LOGGER.info("Send socket channel closed, stopping accept loop");
          break;
        }
      }
    }
    LOGGER.info("Send clients acceptor thread stopped");
  }

  private void acceptReceiveClients() {
    LOGGER.info("Receive clients acceptor thread started");
    while (this.running.get() && !Thread.currentThread().isInterrupted()) {
      try {
        SocketChannel clientSocket = receiveSocketChannel.accept();
        LOGGER.info("Receive client connected: " + clientSocket.getRemoteAddress());
        handleReceiveClient(clientSocket);
      } catch (IOException e) {
        if (this.running.get()) {
          LOGGER.error("Error accepting receive client", e);
        } else {
          LOGGER.info("Receive socket channel closed, stopping accept loop");
          break;
        }
      }
    }
    LOGGER.info("Receive clients acceptor thread stopped");
  }

  private void handleReceiveClient(SocketChannel clientSocket) {
    receiverExecutor.submit(
        () -> {
          try (BufferedReader in =
                  new BufferedReader(new InputStreamReader(clientSocket.socket().getInputStream()));
              PrintWriter out = new PrintWriter(clientSocket.socket().getOutputStream(), true)) {

            String inputLine;
            while ((inputLine = in.readLine()) != null && this.running.get()) {
              final String command = inputLine;
              // Process each command asynchronously to avoid blocking
              CompletableFuture.runAsync(() -> processCommand(command, out), receiverExecutor)
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

  private void handleSendClient(SocketChannel clientSocket) {
    senderExecutor.submit(
        () -> {
          while (this.running.get()) {
            try {
              FrameData frameData = frameQueue.take(); // Blocks until frame available
              sendFrameImmediate(frameData, clientSocket);
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
              break;
            }
          }
        });
  }

  public void sendResponse(byte[] frameData, int reward) {
    // Non-blocking: just add to queue
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

  private void sendFrameImmediate(FrameData frameData, SocketChannel clientSocket) {
    LOGGER.info("Sending frame");
    try {
      ByteBuffer buffer = ByteBuffer.allocate(8 + frameData.data.length);
      buffer.putInt(frameData.reward);
      buffer.putInt(frameData.data.length);
      buffer.put(frameData.data);
      buffer.flip();
      int bytesWritten = clientSocket.write(buffer);
      LOGGER.info("Frame sent: {} bytes", bytesWritten);
      if (bytesWritten != buffer.remaining()) {
        LOGGER.error("Data size mismatch! Expected: {}, Actual: {}", buffer.remaining(), bytesWritten);
      }
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
    // Wait for threads to finish
    if (this.sendThread != null) {
      try {
        this.sendThread.interrupt();
        this.sendThread.join(5000); // Wait up to 5 seconds
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }
    
    if (this.receiveThread != null) {
      try {
        this.receiveThread.interrupt();
        this.receiveThread.join(5000); // Wait up to 5 seconds
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }

    // Shutdown executors
    senderExecutor.shutdown();
    receiverExecutor.shutdown();

    try {
      if (this.sendSocketChannel != null) {
        this.sendSocketChannel.close();
      }
      if (this.receiveSocketChannel != null) {
        this.receiveSocketChannel.close();
      }
    } catch (IOException e) {
      LOGGER.error("Cleanup error: " + e.getMessage());
    }

    try {
      Files.deleteIfExists(Path.of(SEND_SOCKET_PATH));
      Files.deleteIfExists(Path.of(RECEIVE_SOCKET_PATH));
    } catch (IOException e) {
      LOGGER.error("Cleanup error: " + e.getMessage());
    }

    LOGGER.info("NetworkHandler shutdown complete");
  }
}
