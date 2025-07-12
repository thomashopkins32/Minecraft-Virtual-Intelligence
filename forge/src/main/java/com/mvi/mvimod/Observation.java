package com.mvi.mvimod;

public record Observation(int reward, ActionState actionState, byte[] frame) {
    public byte[] serialize() {
        // TODO: Implement this
        throw new UnsupportedOperationException("Not implemented");
    }

    public static Observation deserialize(byte[] data) {
        // TODO: Implement this
        throw new UnsupportedOperationException("Not implemented");
    }
}
