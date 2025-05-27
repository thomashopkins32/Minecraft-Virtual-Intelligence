package com.mvi.mvimod;

import com.mojang.logging.LogUtils;
import net.minecraft.client.Minecraft;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.server.ServerStartingEvent;
import net.minecraftforge.event.server.ServerStoppingEvent;
import net.minecraftforge.eventbus.api.IEventBus;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.config.ModConfig;
import net.minecraftforge.fml.event.lifecycle.FMLClientSetupEvent;
import net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext;
import org.slf4j.Logger;

// The value here should match an entry in the META-INF/mods.toml file
@Mod(MviMod.MODID)
public class MviMod {
    // Define mod id in a common place for everything to reference
    public static final String MODID = "mvi";
    // Directly reference a slf4j logger
    private static final Logger LOGGER = LogUtils.getLogger();
    private Thread networkThread;

    public MviMod(FMLJavaModLoadingContext context) {
        // Register ourselves for server and other game events we are interested in
        MinecraftForge.EVENT_BUS.register(this);

        // Register our mod's ForgeConfigSpec so that Forge can create and load the config file for us
        context.registerConfig(ModConfig.Type.COMMON, Config.SPEC);
    }

    @SubscribeEvent
    public void onServerStarting(ServerStartingEvent event) {
        LOGGER.info("Starting MVI Mod Server");
        startNetworkServer();
    }

    @SubscribeEvent
    public void onServerStopping(ServerStoppingEvent event) {
        LOGGER.info("Stopping MVI Mod Server...");
        stopNetworkServer();
    }

    private void startNetworkServer() {
        if (networkThread != null || !networkThread.isAlive()) {
            networkThread = new Thread(new NetworkHandler(Config.PORT.get()));
            networkThread.start();
            LOGGER.info("Network server started on port " + Config.PORT.get());
        }
    }

    private void stopNetworkServer() {
        if (networkThread != null && networkThread.isAlive()) {
            networkThread.interrupt();
            networkThread = null;
            LOGGER.info("Network server stopped.");
        }
    }
}
