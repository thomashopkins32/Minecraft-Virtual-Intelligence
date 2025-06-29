"""
Minecraft Forge Client for Gymnasium API

This module provides a client implementation that connects to the Minecraft Forge mod
and exposes a Gymnasium-compatible interface for reinforcement learning experiments.
The current implementation focuses on reading frame data only.
"""

import socket
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any
import time
import logging
from dataclasses import dataclass
from PIL import Image

from .config import Config


@dataclass
class ConnectionConfig:
    """Configuration for Minecraft Forge mod connection"""
    host: str = "localhost"
    command_port: int = 12345  # TCP port for commands
    data_port: int = 12346     # UDP port for frame data
    width: int = 1024
    height: int = 768
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class MinecraftForgeClient:
    """
    Low-level client for communicating with the Minecraft Forge mod
    Uses TCP for commands and UDP for frame data (asynchronous)
    """
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.command_socket: socket.socket | None = None
        self.data_socket: socket.socket | None = None
        self.connected: bool = False
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        # Delta encoding state
        self.current_frame: np.ndarray | None = None
        self.frame_width: int = 0
        self.frame_height: int = 0
    
    def connect(self) -> bool:
        """
        Establish connection to the Minecraft Forge mod
        
        Returns
        -------
        bool
            True if connection successful, False otherwise
        """
        for attempt in range(self.config.max_retries):
            try:
                # Connect TCP socket for commands
                self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.command_socket.settimeout(self.config.timeout)
                self.command_socket.connect((self.config.host, self.config.command_port))
                
                # Create UDP socket for frame data
                self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.data_socket.settimeout(self.config.timeout)
                self.data_socket.bind(('', self.config.data_port))  # Bind to receive data
                
                self.connected = True
                self.logger.info(f"Connected to Minecraft Forge mod - TCP: {self.config.host}:{self.config.command_port}, UDP: {self.config.data_port}")
                return True
            except (socket.error, ConnectionRefusedError) as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                self._cleanup_sockets()
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error("Failed to connect after all retries")
                    return False
        
        return False
    
    def disconnect(self):
        """Disconnect from the Minecraft Forge mod"""
        self._cleanup_sockets()
        self.connected = False
        self.logger.info("Disconnected from Minecraft Forge mod")
    
    def _cleanup_sockets(self):
        """Clean up both sockets"""
        if self.command_socket:
            try:
                self.command_socket.close()
            except:
                pass
            finally:
                self.command_socket = None
        
        if self.data_socket:
            try:
                self.data_socket.close()
            except:
                pass
            finally:
                self.data_socket = None
    
    def send_command(self, command: str) -> bool:
        """
        Send a command to the Minecraft Forge mod via TCP
        
        Parameters
        ----------
        command : str
            Command to send to the mod
            
        Returns
        -------
        bool
            True if command sent successfully, False otherwise
        """
        if not self.connected or not self.command_socket:
            self.logger.error("Not connected to Minecraft Forge mod")
            return False
        
        try:
            self.command_socket.send((command + "\n").encode())
            return True
        except socket.error as e:
            self.logger.error(f"Failed to send command: {e}")
            self.connected = False
            return False
    
    def receive_frame_data(self) -> np.ndarray | None:
        """
        Receive frame data from the Minecraft Forge mod via UDP
        
        Returns
        -------
        np.ndarray | None
            Frame data as numpy array if successful, None otherwise
        """
        if not self.connected or not self.data_socket:
            self.logger.error("Not connected to Minecraft Forge mod")
            return None
        
        try:
            # Receive UDP packet (max 65507 bytes for UDP)
            data, addr = self.data_socket.recvfrom(65507)
            
            if len(data) < 8:
                self.logger.error("Received packet too small for protocol")
                return None
            
            # Parse protocol: [reward:4bytes][data_length:4bytes][data:N bytes]
            reward = int.from_bytes(data[0:4], byteorder='big', signed=True)
            data_length = int.from_bytes(data[4:8], byteorder='big', signed=False)
            
            if len(data) != 8 + data_length:
                self.logger.error(f"Packet size mismatch: expected {8 + data_length}, got {len(data)}")
                return None
            
            frame_data = data[8:8+data_length]
            
            if len(frame_data) == 0:
                return None

            # Parse frame data using delta encoding protocol
            return self._parse_frame_data(frame_data)
            
        except socket.timeout:
            # Timeout is expected if no frames are being sent
            self.logger.warning("Timeout waiting for frame data")
            return None
        except socket.error as e:
            self.logger.error(f"Failed to receive frame data: {e}")
            self.connected = False
            return None
    
    def _parse_frame_data(self, frame_data: bytes) -> np.ndarray | None:
        """
        Parse frame data using delta encoding protocol
        
        Parameters
        ----------
        frame_data : bytes
            Raw frame data from UDP packet
            
        Returns
        -------
        np.ndarray | None
            Parsed frame as numpy array if successful, None otherwise
        """
        try:
            if len(frame_data) < 1:
                return None
            
            packet_type = frame_data[0]
            
            if packet_type == 0:
                # Full frame packet: [type:1][width:4][height:4][data_length:4][data...]
                if len(frame_data) < 17:  # 1 + 4 + 4 + 4 + 4 minimum
                    self.logger.error("Full frame packet too small")
                    return None
                
                width = int.from_bytes(frame_data[1:5], byteorder='big')
                height = int.from_bytes(frame_data[5:9], byteorder='big')
                data_length = int.from_bytes(frame_data[9:13], byteorder='big')
                
                if len(frame_data) != 13 + data_length:
                    self.logger.error("Full frame data length mismatch")
                    return None
                
                raw_data = frame_data[13:13+data_length]
                
                # Convert RGBA to RGB (skip alpha channel)
                if len(raw_data) == width * height * 4:
                    frame_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 4)
                    # Convert RGBA to RGB and flip Y axis (OpenGL to standard image coordinates)
                    frame_array = np.flipud(frame_array[:, :, :3])
                    
                    # Store current frame for delta updates
                    self.current_frame = frame_array.copy()
                    self.frame_width = width
                    self.frame_height = height
                    
                    return frame_array
                else:
                    self.logger.error(f"Invalid full frame data size: {len(raw_data)}, expected: {width * height * 4}")
                    return None
                    
            elif packet_type == 1:
                # Delta packet: [type:1][width:4][height:4][num_changes:4][changes...]
                if len(frame_data) < 17:  # 1 + 4 + 4 + 4 + 4 minimum
                    self.logger.error("Delta packet too small")
                    return None
                
                width = int.from_bytes(frame_data[1:5], byteorder='big')
                height = int.from_bytes(frame_data[5:9], byteorder='big')
                num_changes = int.from_bytes(frame_data[9:13], byteorder='big')
                
                if self.current_frame is None or self.frame_width != width or self.frame_height != height:
                    self.logger.warning("Delta packet received but no valid base frame")
                    return None
                
                # Apply delta changes
                changes_data = frame_data[13:]
                expected_size = num_changes * 7  # 4 bytes index + 3 bytes RGB
                
                if len(changes_data) != expected_size:
                    self.logger.error(f"Delta changes data size mismatch: {len(changes_data)}, expected: {expected_size}")
                    return None
                
                # Create a copy of current frame to modify
                updated_frame = self.current_frame.copy()
                
                for i in range(num_changes):
                    offset = i * 7
                    pixel_index = int.from_bytes(changes_data[offset:offset+4], byteorder='big')
                    r = changes_data[offset+4]
                    g = changes_data[offset+5]
                    b = changes_data[offset+6]
                    
                    # Convert linear pixel index to y, x coordinates
                    y = pixel_index // width
                    x = pixel_index % width
                    
                    if 0 <= y < height and 0 <= x < width:
                        # Apply Y-flip (OpenGL coordinates)
                        flipped_y = height - 1 - y
                        updated_frame[flipped_y, x] = [r, g, b]
                
                # Update stored frame
                self.current_frame = updated_frame.copy()
                
                self.logger.debug(f"Applied {num_changes} delta changes")
                return updated_frame
                
            else:
                self.logger.error(f"Unknown packet type: {packet_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to parse frame data: {e}")
            return None


class MinecraftEnv(gym.Env[np.ndarray, np.int64]):
    """
    Gymnasium environment for Minecraft using the Forge mod
    
    This environment provides a Gymnasium-compatible interface for interacting
    with Minecraft through the custom Forge mod. Currently supports reading
    frame data only.
    """
    
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(self, config: Config | None = None, connection_config: ConnectionConfig | None = None):
        super().__init__()
        
        self.config = config or Config()
        self.connection_config = connection_config or ConnectionConfig()
        self.client = MinecraftForgeClient(self.connection_config)
        
        # Set up observation space (RGB image)
        height, width = self.config.engine.image_size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
        
        # For now, we don't support actions, but we need to define the space
        # This will be expanded when action support is added
        self.action_space = spaces.Discrete(1)  # No-op action
        
        self.current_frame = None
        self.step_count = 0
        self.logger = logging.getLogger(__name__)
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment
        
        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility
        options : dict[str, Any] | None
            Additional options for reset
            
        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Initial observation and info dictionary
        """
        super().reset(seed=seed, options=options)
        
        # Connect to Minecraft if not already connected
        if not self.client.connected:
            if not self.client.connect():
                raise RuntimeError("Failed to connect to Minecraft Forge mod")
        
        # Send reset command to mod (for future use)
        self.client.send_command("RESET")
        
        # Get initial frame
        self.current_frame = self.client.receive_frame_data()
        if self.current_frame is None:
            self.logger.warning("No frame data received")
            # Return a black frame if we can't get data
            height, width = self.config.engine.image_size
            self.current_frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            self.logger.info(f"Received frame data: {self.current_frame.shape}")
        
        self.step_count = 0
        
        return self.current_frame, {"step_count": self.step_count}
    
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment
        
        Parameters
        ----------
        action
            Action to take (currently ignored)
            
        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
            Observation, reward, terminated, truncated, info
        """
        # For now, we ignore the action since we're only reading frame data
        
        # Get new frame data
        new_frame = self.client.receive_frame_data()
        if new_frame is not None:
            self.current_frame = self._resize_frame(new_frame)
        
        # Ensure current_frame is never None
        if self.current_frame is None:
            height, width = self.config.engine.image_size
            self.current_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        self.step_count += 1
        
        # Placeholder values for reward and termination
        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.config.engine.max_steps
        
        info = {
            "step_count": self.step_count,
            "frame_received": new_frame is not None
        }
        
        return self.current_frame, reward, terminated, truncated, info
    
    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        """
        Render the environment
        
        Parameters
        ----------
        mode : str
            Render mode (only "rgb_array" supported)
            
        Returns
        -------
        np.ndarray | None
            Rendered frame as numpy array
        """
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode: {mode}")
        
        return self.current_frame
    
    def close(self):
        """Close the environment and disconnect from Minecraft"""
        self.client.disconnect()
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to match expected dimensions
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame
            
        Returns
        -------
        np.ndarray
            Resized frame
        """
        target_height, target_width = self.config.engine.image_size
        
        if frame.shape[:2] != (target_height, target_width):
            # Use PIL for high-quality resizing
            pil_image = Image.fromarray(frame)
            try:
                # Try new PIL API first
                from PIL.Image import Resampling
                pil_image = pil_image.resize((target_width, target_height), Resampling.LANCZOS)
            except (ImportError, AttributeError):
                # Fall back to old PIL API using numeric constant (1 = LANCZOS)
                pil_image = pil_image.resize((target_width, target_height), 1)
            frame = np.array(pil_image)
        
        return frame


def create_minecraft_env(config: Config | None = None, connection_config: ConnectionConfig | None = None) -> MinecraftEnv:
    """
    Factory function to create a Minecraft environment
    
    Parameters
    ----------
    config : Config | None
        MVI configuration object
    connection_config : ConnectionConfig | None
        Connection configuration for the Minecraft mod
        
    Returns
    -------
    MinecraftEnv
        Configured Minecraft environment
    """
    return MinecraftEnv(config=config, connection_config=connection_config) 