package com.mvi.mvimod;

import com.mojang.blaze3d.platform.Window;
import com.mojang.logging.LogUtils;
import java.nio.ByteBuffer;
import net.minecraft.client.Minecraft;
import net.minecraft.world.entity.player.Player;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.event.entity.living.LivingDeathEvent;
import net.minecraftforge.event.entity.living.LivingHurtEvent;
import net.minecraftforge.event.server.ServerStartingEvent;
import net.minecraftforge.event.server.ServerStoppingEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import org.lwjgl.opengl.GL11;
import org.slf4j.Logger;

public class ClientEventHandler {
  private static final Logger LOGGER = LogUtils.getLogger();
  private static DataBridge dataBridge = DataBridge.getInstance();

  @SubscribeEvent
  public static void onServerStarting(ServerStartingEvent event) {
    LOGGER.info("MVI Mod Server Starting - Network handler is managed on client side");
  }

  @SubscribeEvent
  public static void onServerStopping(ServerStoppingEvent event) {
    LOGGER.info("MVI Mod Server Stopping");
  }

  @SubscribeEvent
  public static void onClientTick(TickEvent.ClientTickEvent event) {
    ArrayList<String> commands = dataBridge.emptyCommandQueue();
    for (String command : commands) {
      processCommand(command);

    }

    if (event.phase == TickEvent.Phase.END) {
      // TODO: Move to data bridge?
      int reward = packageReward();
      ActionState actionState = captureActionState();
      byte[] frame = captureFrame();
      dataBridge.sendObservation(new Observation(reward, actionState, frame));
    }
  }

  @SubscribeEvent
  public static void onPlayerHurt(LivingHurtEvent event) {
    if (event.getEntity() instanceof Player) {
      dataBridge.sendEvent("PLAYER_HURT", String.valueOf(event.getAmount()));
    }
  }

  @SubscribeEvent
  public static void onPlayerDeath(LivingDeathEvent event) {
    if (event.getEntity() instanceof Player) {
      dataBridge.sendEvent("PLAYER_DEATH", "-100.0");
    }
  }

  private static int packageReward() {
    // TODO: Get rewards from the reward queue
    throw new UnsupportedOperationException("Not implemented");
  }

  private static ActionState captureActionState() {
    // TODO: Get the state of every action the user can take
    throw new UnsupportedOperationException("Not implemented");
  }

  private static byte[] captureFrame() {
    Minecraft mc = Minecraft.getInstance();
    if (mc.level != null && mc.player != null) {
      return captureScreenshot(mc.getWindow());
    }
    return null;
  }

  private static byte[] captureScreenshot(Window window) {
    int width = window.getWidth();
    int height = window.getHeight();

    ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 3);
    GL11.glReadPixels(0, 0, width, height, GL11.GL_RGB, GL11.GL_UNSIGNED_BYTE, buffer);

    byte[] bytes = new byte[buffer.capacity()];
    buffer.get(bytes);
    return bytes;
  }
}
