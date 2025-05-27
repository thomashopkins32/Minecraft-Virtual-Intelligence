package com.mvi.mvimod;

import net.minecraft.client.Minecraft;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.api.distmarker.OnlyIn;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.event.entity.living.LivingDeathEvent;
import net.minecraftforge.event.entity.living.LivingHurtEvent;
import net.minecraft.world.entity.player.Player;
import net.minecraftforge.eventbus.api.SubscribeEvent;

@OnlyIn(Dist.CLIENT)
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
            byte[] frameData = captureScreenshot();
            dataBridge.sendFrame(frameData);
        }
    }
    
    private byte[] captureScreenshot() {
        // Implementation for capturing frame as byte array
        // This is complex - you'll need to read from the framebuffer
        return new byte[0]; // Placeholder
    }
}
