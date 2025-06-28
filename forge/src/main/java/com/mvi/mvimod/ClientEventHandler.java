package com.mvi.mvimod;

import net.minecraft.client.Minecraft;
import com.mojang.blaze3d.platform.Window;
import net.minecraft.world.entity.player.Player;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.event.entity.living.LivingDeathEvent;
import net.minecraftforge.event.entity.living.LivingHurtEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventBusSubscriber.Bus;

@Mod.EventBusSubscriber(modid = "mvimod", bus = Bus.FORGE, value = Dist.CLIENT)
public class ClientEventHandler {
    private DataBridge dataBridge = DataBridge.getInstance();

    @SubscribeEvent
    public void onClientTick(TickEvent.ClientTickEvent event) {
        if (event.phase == TickEvent.Phase.END) {
            captureAndSendFrame();
        }
    }
    
    @SubscribeEvent
    public void onPlayerHurt(LivingHurtEvent event) {
        if (event.getEntity() instanceof Player) {
            dataBridge.sendEvent("PLAYER_HURT", 
                String.valueOf(event.getAmount()));
        }
    }
    
    @SubscribeEvent
    public void onPlayerDeath(LivingDeathEvent event) {
        if (event.getEntity() instanceof Player) {
            dataBridge.sendEvent("PLAYER_DEATH", "");
        }
    }
    
    private void captureAndSendFrame() {
        Minecraft mc = Minecraft.getInstance();
        if (mc.level != null && mc.player != null) {
            // Capture frame (you'll need to implement this)
            byte[] frameData = captureScreenshot(mc.getWindow());
            dataBridge.sendFrame(frameData);
        }
    }
    
    private byte[] captureScreenshot(Window window) {
        int width = window.getWidth();
        int height = window.getHeight();

        ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 4);
        GL11.glReadPixels(0, 0, width, height, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, buffer);

        return buffer.array();
    }
}
