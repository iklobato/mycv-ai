"""
WebSocket Manager for real-time communication.

Handles WebSocket connections, audio streaming, and real-time
communication between clients and the AI services.
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages individual WebSocket connections."""
    
    def __init__(self, websocket: WebSocket, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.connected_at = datetime.now()
        self.is_active = True
        self.user_id: Optional[str] = None
        self.session_data: Dict[str, Any] = {}
        
    async def send_message(self, message: Dict[str, Any]):
        """Send a message to the connected client."""
        try:
            if self.is_active:
                await self.websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message to {self.connection_id}: {e}")
            self.is_active = False
            
    async def send_text(self, text: str):
        """Send text message to the connected client."""
        try:
            if self.is_active:
                await self.websocket.send_text(text)
        except Exception as e:
            logger.error(f"Failed to send text to {self.connection_id}: {e}")
            self.is_active = False
            
    async def send_bytes(self, data: bytes):
        """Send binary data to the connected client."""
        try:
            if self.is_active:
                await self.websocket.send_bytes(data)
        except Exception as e:
            logger.error(f"Failed to send bytes to {self.connection_id}: {e}")
            self.is_active = False
            
    def disconnect(self):
        """Mark connection as disconnected."""
        self.is_active = False


class WebSocketManager:
    """
    Manages multiple WebSocket connections and handles real-time communication.
    
    Features:
    - Connection management
    - Broadcasting to multiple clients
    - Audio streaming
    - Session management
    - Error handling and reconnection
    """
    
    def __init__(self):
        self.connections: Dict[str, ConnectionManager] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.audio_buffers: Dict[str, List[bytes]] = {}
        self.conversation_contexts: Dict[str, List[Dict[str, str]]] = {}
        self._lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None) -> str:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            user_id: Optional user identifier
            
        Returns:
            str: Connection ID
        """
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        connection = ConnectionManager(websocket, connection_id)
        connection.user_id = user_id
        
        async with self._lock:
            self.connections[connection_id] = connection
            
            # Track user connections
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
                
            # Initialize audio buffer and conversation context
            self.audio_buffers[connection_id] = []
            self.conversation_contexts[connection_id] = []
            
        logger.info(f"New WebSocket connection: {connection_id} (user: {user_id})")
        
        # Send connection confirmation
        await connection.send_message({
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat()
        })
        
        return connection_id
        
    def disconnect(self, websocket: WebSocket):
        """
        Disconnect a WebSocket connection.
        
        Args:
            websocket: WebSocket connection to disconnect
        """
        connection_id = self._find_connection_id(websocket)
        if connection_id:
            self._disconnect_by_id(connection_id)
            
    def _find_connection_id(self, websocket: WebSocket) -> Optional[str]:
        """Find connection ID by WebSocket instance."""
        for connection_id, connection in self.connections.items():
            if connection.websocket == websocket:
                return connection_id
        return None
        
    def _disconnect_by_id(self, connection_id: str):
        """Disconnect a connection by ID."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            user_id = connection.user_id
            
            # Clean up connection
            connection.disconnect()
            del self.connections[connection_id]
            
            # Clean up user tracking
            if user_id and user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
                    
            # Clean up buffers and contexts
            self.audio_buffers.pop(connection_id, None)
            self.conversation_contexts.pop(connection_id, None)
            
            logger.info(f"WebSocket disconnected: {connection_id}")
            
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """
        Send message to a specific connection.
        
        Args:
            connection_id: Target connection ID
            message: Message to send
            
        Returns:
            bool: Success status
        """
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            if connection.is_active:
                await connection.send_message(message)
                return True
            else:
                # Clean up inactive connection
                self._disconnect_by_id(connection_id)
        return False
        
    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """
        Send message to all connections of a user.
        
        Args:
            user_id: Target user ID
            message: Message to send
            
        Returns:
            int: Number of successful sends
        """
        if user_id not in self.user_connections:
            return 0
            
        success_count = 0
        connection_ids = list(self.user_connections[user_id])
        
        for connection_id in connection_ids:
            if await self.send_to_connection(connection_id, message):
                success_count += 1
                
        return success_count
        
    async def broadcast(self, message: Dict[str, Any], exclude_connections: Optional[List[str]] = None):
        """
        Broadcast message to all active connections.
        
        Args:
            message: Message to broadcast
            exclude_connections: Connection IDs to exclude
        """
        exclude_set = set(exclude_connections or [])
        
        for connection_id in list(self.connections.keys()):
            if connection_id not in exclude_set:
                await self.send_to_connection(connection_id, message)
                
    async def stream_audio(self, connection_id: str, audio_data: bytes):
        """
        Handle incoming audio stream data.
        
        Args:
            connection_id: Source connection ID
            audio_data: Raw audio bytes
        """
        if connection_id in self.audio_buffers:
            self.audio_buffers[connection_id].append(audio_data)
            
            # Notify about new audio data
            await self.send_to_connection(connection_id, {
                "type": "audio_received",
                "buffer_size": len(self.audio_buffers[connection_id]),
                "timestamp": datetime.now().isoformat()
            })
            
    async def get_audio_buffer(self, connection_id: str, clear: bool = False) -> List[bytes]:
        """
        Get accumulated audio buffer for a connection.
        
        Args:
            connection_id: Connection ID
            clear: Whether to clear the buffer after reading
            
        Returns:
            List of audio data chunks
        """
        if connection_id not in self.audio_buffers:
            return []
            
        buffer = self.audio_buffers[connection_id].copy()
        
        if clear:
            self.audio_buffers[connection_id].clear()
            
        return buffer
        
    async def combine_audio_buffer(self, connection_id: str, clear: bool = True) -> Optional[bytes]:
        """
        Combine all audio chunks in buffer into single bytes object.
        
        Args:
            connection_id: Connection ID
            clear: Whether to clear the buffer after combining
            
        Returns:
            Combined audio data or None if empty
        """
        buffer = await self.get_audio_buffer(connection_id, clear)
        
        if not buffer:
            return None
            
        return b''.join(buffer)
        
    def add_conversation_message(self, connection_id: str, role: str, content: str):
        """
        Add a message to conversation context.
        
        Args:
            connection_id: Connection ID
            role: Message role (user/assistant)
            content: Message content
        """
        if connection_id in self.conversation_contexts:
            self.conversation_contexts[connection_id].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 20 messages to prevent memory issues
            if len(self.conversation_contexts[connection_id]) > 20:
                self.conversation_contexts[connection_id] = \
                    self.conversation_contexts[connection_id][-20:]
                    
    def get_conversation_context(self, connection_id: str) -> List[Dict[str, str]]:
        """
        Get conversation context for a connection.
        
        Args:
            connection_id: Connection ID
            
        Returns:
            List of conversation messages
        """
        return self.conversation_contexts.get(connection_id, [])
        
    def clear_conversation_context(self, connection_id: str):
        """Clear conversation context for a connection."""
        if connection_id in self.conversation_contexts:
            self.conversation_contexts[connection_id].clear()
            
    async def send_heartbeat(self):
        """Send heartbeat to all connections to check status."""
        heartbeat_message = {
            "type": "heartbeat",
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all connections and clean up inactive ones
        inactive_connections = []
        
        for connection_id, connection in self.connections.items():
            try:
                if connection.is_active:
                    await connection.send_message(heartbeat_message)
                else:
                    inactive_connections.append(connection_id)
            except Exception as e:
                logger.warning(f"Heartbeat failed for {connection_id}: {e}")
                inactive_connections.append(connection_id)
                
        # Clean up inactive connections
        for connection_id in inactive_connections:
            self._disconnect_by_id(connection_id)
            
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        active_connections = sum(1 for conn in self.connections.values() if conn.is_active)
        
        return {
            "total_connections": len(self.connections),
            "active_connections": active_connections,
            "unique_users": len(self.user_connections),
            "audio_buffers": len(self.audio_buffers),
            "conversation_contexts": len(self.conversation_contexts)
        }
        
    async def cleanup_inactive_connections(self):
        """Clean up inactive connections periodically."""
        inactive_connections = []
        
        for connection_id, connection in self.connections.items():
            if not connection.is_active:
                inactive_connections.append(connection_id)
                
        for connection_id in inactive_connections:
            self._disconnect_by_id(connection_id)
            
        if inactive_connections:
            logger.info(f"Cleaned up {len(inactive_connections)} inactive connections")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


async def start_heartbeat_task():
    """Start periodic heartbeat task."""
    while True:
        try:
            await websocket_manager.send_heartbeat()
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
        except Exception as e:
            logger.error(f"Heartbeat task error: {e}")
            await asyncio.sleep(60)  # Wait longer on error


async def start_cleanup_task():
    """Start periodic cleanup task."""
    while True:
        try:
            await websocket_manager.cleanup_inactive_connections()
            await asyncio.sleep(300)  # Clean up every 5 minutes
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            await asyncio.sleep(300) 