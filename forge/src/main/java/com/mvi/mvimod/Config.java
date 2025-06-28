package com.mvi.mvimod;

import net.minecraftforge.common.ForgeConfigSpec;
import net.minecraftforge.fml.common.Mod;

@Mod.EventBusSubscriber(modid = MviMod.MODID, bus = Mod.EventBusSubscriber.Bus.MOD)
public class Config {

  // Configuration Builder
  private static final ForgeConfigSpec.Builder BUILDER = new ForgeConfigSpec.Builder();

  // Configuration Values
  public static final ForgeConfigSpec.ConfigValue<Integer> PORT;

  // Built Configuration Specification
  public static final ForgeConfigSpec SPEC;

  // Static initialization block ensures proper ordering
  static {
    // Define all configuration values here
    BUILDER.comment("Network Configuration");
    BUILDER.push("network");

    PORT = BUILDER.comment("Port for MVI communication").defineInRange("port", 12345, 1024, 65535);

    BUILDER.pop();

    // Build the specification after all values are defined
    SPEC = BUILDER.build();
  }
}
