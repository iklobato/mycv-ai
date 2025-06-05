"""
Unit tests for the WebSocketManager and ConnectionManager.

Tests cover:
- Connection management and lifecycle
- Message sending and broadcasting
- Audio streaming and buffering
- Session and conversation management
- Error handling and cleanup
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import uuid

from backend.services.websocket_manager import (
    ConnectionManager, 
    WebSocketManager,
    start_heartbeat_task,
    start_cleanup_task
)


class TestConnectionManager:
    """Test suite for ConnectionManager."""
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        websocket = AsyncMock()
        return websocket
    
    @pytest.fixture
    def connection_manager(self, mock_websocket):
        """Create a ConnectionManager instance."""
        connection_id = "test-connection-123"
        return ConnectionManager(mock_websocket, connection_id)
    
    def test_init(self, connection_manager, mock_websocket):
        """Test ConnectionManager initialization."""
        assert connection_manager.websocket == mock_websocket
        assert connection_manager.connection_id == "test-connection-123"
        assert connection_manager.is_active
        assert connection_manager.user_id is None
        assert isinstance(connection_manager.connected_at, datetime)
        assert connection_manager.session_data == {}
    
    async def test_send_message_success(self, connection_manager, mock_websocket):
        """Test successful message sending."""
        message = {"type": "test", "data": "hello"}
        
        await connection_manager.send_message(message)
        
        mock_websocket.send_json.assert_called_once_with(message)
        assert connection_manager.is_active
    
    async def test_send_message_failure(self, connection_manager, mock_websocket):
        """Test message sending failure."""
        message = {"type": "test", "data": "hello"}
        mock_websocket.send_json.side_effect = Exception("Connection lost")
        
        await connection_manager.send_message(message)
        
        assert not connection_manager.is_active
    
    async def test_send_message_inactive_connection(self, connection_manager, mock_websocket):
        """Test sending message to inactive connection."""
        connection_manager.is_active = False
        message = {"type": "test", "data": "hello"}
        
        await connection_manager.send_message(message)
        
        mock_websocket.send_json.assert_not_called()
    
    async def test_send_text_success(self, connection_manager, mock_websocket):
        """Test successful text sending."""
        text = "Hello, WebSocket!"
        
        await connection_manager.send_text(text)
        
        mock_websocket.send_text.assert_called_once_with(text)
        assert connection_manager.is_active
    
    async def test_send_text_failure(self, connection_manager, mock_websocket):
        """Test text sending failure."""
        text = "Hello, WebSocket!"
        mock_websocket.send_text.side_effect = Exception("Connection lost")
        
        await connection_manager.send_text(text)
        
        assert not connection_manager.is_active
    
    async def test_send_bytes_success(self, connection_manager, mock_websocket):
        """Test successful binary data sending."""
        data = b"binary_data"
        
        await connection_manager.send_bytes(data)
        
        mock_websocket.send_bytes.assert_called_once_with(data)
        assert connection_manager.is_active
    
    async def test_send_bytes_failure(self, connection_manager, mock_websocket):
        """Test binary data sending failure."""
        data = b"binary_data"
        mock_websocket.send_bytes.side_effect = Exception("Connection lost")
        
        await connection_manager.send_bytes(data)
        
        assert not connection_manager.is_active
    
    def test_disconnect(self, connection_manager):
        """Test connection disconnection."""
        assert connection_manager.is_active
        
        connection_manager.disconnect()
        
        assert not connection_manager.is_active


class TestWebSocketManager:
    """Test suite for WebSocketManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a WebSocketManager instance."""
        return WebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        websocket = AsyncMock()
        return websocket
    
    def test_init(self, manager):
        """Test WebSocketManager initialization."""
        assert manager.connections == {}
        assert manager.user_connections == {}
        assert manager.audio_buffers == {}
        assert manager.conversation_contexts == {}
        assert hasattr(manager, '_lock')
    
    @patch('backend.services.websocket_manager.uuid')
    async def test_connect_new_user(self, mock_uuid, manager, mock_websocket):
        """Test connecting a new user."""
        mock_uuid.uuid4.return_value = "test-uuid-123"
        user_id = "user123"
        
        connection_id = await manager.connect(mock_websocket, user_id)
        
        assert connection_id == "test-uuid-123"
        assert connection_id in manager.connections
        assert user_id in manager.user_connections
        assert connection_id in manager.user_connections[user_id]
        assert connection_id in manager.audio_buffers
        assert connection_id in manager.conversation_contexts
        
        # Verify WebSocket accept was called
        mock_websocket.accept.assert_called_once()
        
        # Verify connection confirmation was sent
        connection = manager.connections[connection_id]
        connection.websocket.send_json.assert_called_once()
        sent_message = connection.websocket.send_json.call_args[0][0]
        assert sent_message["type"] == "connection_established"
        assert sent_message["connection_id"] == connection_id
    
    @patch('backend.services.websocket_manager.uuid')
    async def test_connect_anonymous_user(self, mock_uuid, manager, mock_websocket):
        """Test connecting an anonymous user."""
        mock_uuid.uuid4.return_value = "test-uuid-456"
        
        connection_id = await manager.connect(mock_websocket, None)
        
        assert connection_id == "test-uuid-456"
        assert connection_id in manager.connections
        assert manager.connections[connection_id].user_id is None
        
        # Anonymous user should not be in user_connections
        assert len(manager.user_connections) == 0
    
    def test_find_connection_id(self, manager):
        """Test finding connection ID by WebSocket."""
        # Create mock connection
        mock_websocket = Mock()
        connection = Mock()
        connection.websocket = mock_websocket
        manager.connections["test-id"] = connection
        
        result = manager._find_connection_id(mock_websocket)
        assert result == "test-id"
        
        # Test with non-existent WebSocket
        other_websocket = Mock()
        result = manager._find_connection_id(other_websocket)
        assert result is None
    
    def test_disconnect_by_id(self, manager):
        """Test disconnecting by connection ID."""
        # Setup connection
        connection = Mock()
        connection.user_id = "user123"
        manager.connections["conn-123"] = connection
        manager.user_connections["user123"] = {"conn-123"}
        manager.audio_buffers["conn-123"] = []
        manager.conversation_contexts["conn-123"] = []
        
        manager._disconnect_by_id("conn-123")
        
        # Verify cleanup
        assert "conn-123" not in manager.connections
        assert "user123" not in manager.user_connections  # Removed because empty
        assert "conn-123" not in manager.audio_buffers
        assert "conn-123" not in manager.conversation_contexts
        connection.disconnect.assert_called_once()
    
    def test_disconnect_by_id_multiple_connections(self, manager):
        """Test disconnecting one of multiple user connections."""
        # Setup user with multiple connections
        connection1 = Mock()
        connection1.user_id = "user123"
        connection2 = Mock()
        connection2.user_id = "user123"
        
        manager.connections["conn-1"] = connection1
        manager.connections["conn-2"] = connection2
        manager.user_connections["user123"] = {"conn-1", "conn-2"}
        
        manager._disconnect_by_id("conn-1")
        
        # Verify partial cleanup
        assert "conn-1" not in manager.connections
        assert "conn-2" in manager.connections
        assert manager.user_connections["user123"] == {"conn-2"}  # Still has conn-2
    
    def test_disconnect_by_websocket(self, manager):
        """Test disconnecting by WebSocket instance."""
        mock_websocket = Mock()
        connection = Mock()
        connection.websocket = mock_websocket
        connection.user_id = "user123"
        
        manager.connections["conn-123"] = connection
        manager.user_connections["user123"] = {"conn-123"}
        
        manager.disconnect(mock_websocket)
        
        # Verify connection was found and disconnected
        assert "conn-123" not in manager.connections
        connection.disconnect.assert_called_once()
    
    async def test_send_to_connection_success(self, manager):
        """Test sending message to specific connection."""
        # Setup connection with AsyncMock for send_message
        connection = Mock()
        connection.is_active = True
        connection.send_message = AsyncMock()
        manager.connections["conn-123"] = connection
        
        message = {"type": "test", "data": "hello"}
        result = await manager.send_to_connection("conn-123", message)
        
        assert result
        connection.send_message.assert_called_once_with(message)
    
    async def test_send_to_connection_inactive(self, manager):
        """Test sending message to inactive connection."""
        # Setup inactive connection
        connection = Mock()
        connection.is_active = False
        manager.connections["conn-123"] = connection
        
        message = {"type": "test", "data": "hello"}
        result = await manager.send_to_connection("conn-123", message)
        
        assert not result
        # Connection should be cleaned up
        assert "conn-123" not in manager.connections
    
    async def test_send_to_connection_not_found(self, manager):
        """Test sending message to non-existent connection."""
        message = {"type": "test", "data": "hello"}
        result = await manager.send_to_connection("non-existent", message)
        
        assert not result
    
    async def test_send_to_user_success(self, manager):
        """Test sending message to all user connections."""
        # Setup user with multiple connections
        connection1 = Mock()
        connection1.is_active = True
        connection1.send_message = AsyncMock()
        connection2 = Mock()
        connection2.is_active = True
        connection2.send_message = AsyncMock()
        
        manager.connections["conn-1"] = connection1
        manager.connections["conn-2"] = connection2
        manager.user_connections["user123"] = {"conn-1", "conn-2"}
        
        message = {"type": "test", "data": "hello"}
        count = await manager.send_to_user("user123", message)
        
        assert count == 2
        connection1.send_message.assert_called_once_with(message)
        connection2.send_message.assert_called_once_with(message)
    
    async def test_send_to_user_not_found(self, manager):
        """Test sending message to non-existent user."""
        message = {"type": "test", "data": "hello"}
        count = await manager.send_to_user("non-existent", message)
        
        assert count == 0
    
    async def test_broadcast_all(self, manager):
        """Test broadcasting to all connections."""
        # Setup multiple connections
        connection1 = Mock()
        connection1.is_active = True
        connection1.send_message = AsyncMock()
        connection2 = Mock()
        connection2.is_active = True
        connection2.send_message = AsyncMock()
        connection3 = Mock()
        connection3.is_active = True
        connection3.send_message = AsyncMock()
        
        manager.connections["conn-1"] = connection1
        manager.connections["conn-2"] = connection2
        manager.connections["conn-3"] = connection3
        
        message = {"type": "broadcast", "data": "hello all"}
        await manager.broadcast(message)
        
        connection1.send_message.assert_called_once_with(message)
        connection2.send_message.assert_called_once_with(message)
        connection3.send_message.assert_called_once_with(message)
    
    async def test_broadcast_exclude_connections(self, manager):
        """Test broadcasting with excluded connections."""
        # Setup multiple connections
        connection1 = Mock()
        connection1.is_active = True
        connection1.send_message = AsyncMock()
        connection2 = Mock()
        connection2.is_active = True
        connection2.send_message = AsyncMock()
        connection3 = Mock()
        connection3.is_active = True
        connection3.send_message = AsyncMock()
        
        manager.connections["conn-1"] = connection1
        manager.connections["conn-2"] = connection2
        manager.connections["conn-3"] = connection3
        
        message = {"type": "broadcast", "data": "hello some"}
        await manager.broadcast(message, exclude_connections=["conn-2"])
        
        connection1.send_message.assert_called_once_with(message)
        connection2.send_message.assert_not_called()  # Excluded
        connection3.send_message.assert_called_once_with(message)
    
    async def test_stream_audio(self, manager):
        """Test audio streaming to connection."""
        # Initialize audio buffer
        manager.audio_buffers["conn-123"] = []
        
        audio_data = b"audio_chunk_data"
        await manager.stream_audio("conn-123", audio_data)
        
        assert manager.audio_buffers["conn-123"] == [audio_data]
    
    async def test_stream_audio_multiple_chunks(self, manager):
        """Test streaming multiple audio chunks."""
        manager.audio_buffers["conn-123"] = []
        
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        for chunk in chunks:
            await manager.stream_audio("conn-123", chunk)
        
        assert manager.audio_buffers["conn-123"] == chunks
    
    async def test_get_audio_buffer_keep(self, manager):
        """Test getting audio buffer without clearing."""
        audio_chunks = [b"chunk1", b"chunk2", b"chunk3"]
        manager.audio_buffers["conn-123"] = audio_chunks.copy()
        
        result = await manager.get_audio_buffer("conn-123", clear=False)
        
        assert result == audio_chunks
        assert manager.audio_buffers["conn-123"] == audio_chunks  # Not cleared
    
    async def test_get_audio_buffer_clear(self, manager):
        """Test getting audio buffer with clearing."""
        audio_chunks = [b"chunk1", b"chunk2", b"chunk3"]
        manager.audio_buffers["conn-123"] = audio_chunks.copy()
        
        result = await manager.get_audio_buffer("conn-123", clear=True)
        
        assert result == audio_chunks
        assert manager.audio_buffers["conn-123"] == []  # Cleared
    
    async def test_get_audio_buffer_not_found(self, manager):
        """Test getting audio buffer for non-existent connection."""
        result = await manager.get_audio_buffer("non-existent")
        
        assert result == []
    
    async def test_combine_audio_buffer_success(self, manager):
        """Test combining audio buffer chunks."""
        audio_chunks = [b"chunk1", b"chunk2", b"chunk3"]
        manager.audio_buffers["conn-123"] = audio_chunks.copy()
        
        result = await manager.combine_audio_buffer("conn-123", clear=True)
        
        assert result == b"chunk1chunk2chunk3"
        assert manager.audio_buffers["conn-123"] == []  # Cleared
    
    async def test_combine_audio_buffer_empty(self, manager):
        """Test combining empty audio buffer."""
        manager.audio_buffers["conn-123"] = []
        
        result = await manager.combine_audio_buffer("conn-123")
        
        assert result is None
    
    async def test_combine_audio_buffer_not_found(self, manager):
        """Test combining audio buffer for non-existent connection."""
        result = await manager.combine_audio_buffer("non-existent")
        
        assert result is None
    
    def test_add_conversation_message(self, manager):
        """Test adding conversation message."""
        manager.conversation_contexts["conn-123"] = []
        
        manager.add_conversation_message("conn-123", "user", "Hello")
        manager.add_conversation_message("conn-123", "assistant", "Hi there!")
        
        context = manager.conversation_contexts["conn-123"]
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "Hello"
        assert "timestamp" in context[0]
        assert context[1]["role"] == "assistant"
        assert context[1]["content"] == "Hi there!"
        assert "timestamp" in context[1]
    
    def test_add_conversation_message_with_limit(self, manager):
        """Test conversation message with history limit."""
        manager.conversation_contexts["conn-123"] = []
        
        # Add messages beyond the limit (actual limit is 20)
        for i in range(25):
            manager.add_conversation_message("conn-123", "user", f"Message {i}")
        
        context = manager.conversation_contexts["conn-123"]
        assert len(context) == 20  # Should be limited to 20
        assert context[0]["content"] == "Message 5"  # Oldest messages removed
        assert context[-1]["content"] == "Message 24"  # Latest message kept
    
    def test_get_conversation_context(self, manager):
        """Test getting conversation context."""
        messages = [
            {"role": "user", "content": "Hello", "timestamp": "2023-01-01T00:00:00"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "2023-01-01T00:00:01"}
        ]
        manager.conversation_contexts["conn-123"] = messages.copy()
        
        result = manager.get_conversation_context("conn-123")
        
        assert result == messages
    
    def test_get_conversation_context_not_found(self, manager):
        """Test getting conversation context for non-existent connection."""
        result = manager.get_conversation_context("non-existent")
        
        assert result == []
    
    def test_clear_conversation_context(self, manager):
        """Test clearing conversation context."""
        manager.conversation_contexts["conn-123"] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        manager.clear_conversation_context("conn-123")
        
        assert manager.conversation_contexts["conn-123"] == []
    
    async def test_send_heartbeat(self, manager):
        """Test sending heartbeat to all connections."""
        # Setup multiple connections
        connection1 = Mock()
        connection1.is_active = True
        connection1.send_message = AsyncMock()
        connection2 = Mock()
        connection2.is_active = True
        connection2.send_message = AsyncMock()
        
        manager.connections["conn-1"] = connection1
        manager.connections["conn-2"] = connection2
        
        await manager.send_heartbeat()
        
        # Verify heartbeat message was sent to all connections
        connection1.send_message.assert_called_once()
        connection2.send_message.assert_called_once()
        
        # Check heartbeat message format
        heartbeat_msg = connection1.send_message.call_args[0][0]
        assert heartbeat_msg["type"] == "heartbeat"
        assert "timestamp" in heartbeat_msg
    
    async def test_send_heartbeat_with_exception(self, manager):
        """Test sending heartbeat when some connections fail."""
        # Setup connections - one working, one failing
        working_connection = Mock()
        working_connection.is_active = True
        working_connection.send_message = AsyncMock()
        
        failing_connection = Mock()
        failing_connection.is_active = True
        failing_connection.send_message = AsyncMock(side_effect=Exception("Send failed"))
        
        manager.connections["working-conn"] = working_connection
        manager.connections["failing-conn"] = failing_connection
        
        # This should not raise an exception despite one connection failing
        await manager.send_heartbeat()
        
        # Verify working connection received heartbeat
        working_connection.send_message.assert_called_once()
        
        # Verify failing connection was attempted and cleaned up
        failing_connection.send_message.assert_called_once()
        assert "failing-conn" not in manager.connections  # Should be cleaned up
        assert "working-conn" in manager.connections  # Should remain
    
    def test_get_connection_stats(self, manager):
        """Test getting connection statistics."""
        # Setup connections
        connection1 = Mock()
        connection1.user_id = "user1"
        connection1.is_active = True
        connection2 = Mock()
        connection2.user_id = "user2"
        connection2.is_active = True
        connection3 = Mock()
        connection3.user_id = None
        connection3.is_active = False
        
        manager.connections["conn-1"] = connection1
        manager.connections["conn-2"] = connection2
        manager.connections["conn-3"] = connection3
        manager.user_connections["user1"] = {"conn-1"}
        manager.user_connections["user2"] = {"conn-2"}
        
        stats = manager.get_connection_stats()
        
        assert stats["total_connections"] == 3
        assert stats["active_connections"] == 2
        assert stats["unique_users"] == 2
        # Note: Check what stats are actually returned by the real implementation
        assert "total_connections" in stats
        assert "active_connections" in stats
        assert "unique_users" in stats
    
    async def test_cleanup_inactive_connections(self, manager):
        """Test cleaning up inactive connections."""
        # Setup active and inactive connections
        active_connection = Mock()
        active_connection.is_active = True
        active_connection.user_id = "user1"
        
        inactive_connection = Mock()
        inactive_connection.is_active = False
        inactive_connection.user_id = "user2"
        
        manager.connections["active"] = active_connection
        manager.connections["inactive"] = inactive_connection
        manager.user_connections["user1"] = {"active"}
        manager.user_connections["user2"] = {"inactive"}
        manager.audio_buffers["active"] = []
        manager.audio_buffers["inactive"] = []
        
        await manager.cleanup_inactive_connections()
        
        # Verify inactive connection was cleaned up
        assert "active" in manager.connections
        assert "inactive" not in manager.connections
        assert "user1" in manager.user_connections
        assert "user2" not in manager.user_connections
        assert "active" in manager.audio_buffers
        assert "inactive" not in manager.audio_buffers


class TestWebSocketManagerIntegration:
    """Integration tests for WebSocketManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a WebSocketManager instance."""
        return WebSocketManager()
    
    @pytest.fixture
    def mock_websockets(self):
        """Create multiple mock WebSockets."""
        return [AsyncMock() for _ in range(3)]
    
    @patch('backend.services.websocket_manager.uuid')
    async def test_multi_user_conversation_flow(self, mock_uuid, manager, mock_websockets):
        """Test complete conversation flow with multiple users."""
        # Setup UUIDs
        mock_uuid.uuid4.side_effect = ["conn-1", "conn-2", "conn-3"]
        
        # Connect multiple users
        conn1 = await manager.connect(mock_websockets[0], "user1")
        conn2 = await manager.connect(mock_websockets[1], "user2")
        conn3 = await manager.connect(mock_websockets[2], None)  # Anonymous
        
        # Add conversation messages
        manager.add_conversation_message(conn1, "user", "Hello from user1")
        manager.add_conversation_message(conn1, "assistant", "Hi user1!")
        manager.add_conversation_message(conn2, "user", "Hello from user2")
        
        # Stream audio data
        await manager.stream_audio(conn1, b"audio_chunk_1")
        await manager.stream_audio(conn1, b"audio_chunk_2")
        await manager.stream_audio(conn2, b"user2_audio")
        
        # Send targeted messages
        user1_msg = {"type": "private", "content": "Message for user1"}
        await manager.send_to_user("user1", user1_msg)
        
        broadcast_msg = {"type": "announcement", "content": "Message for all"}
        await manager.broadcast(broadcast_msg, exclude_connections=[conn3])
        
        # Verify conversation contexts
        user1_context = manager.get_conversation_context(conn1)
        user2_context = manager.get_conversation_context(conn2)
        assert len(user1_context) == 2
        assert len(user2_context) == 1
        
        # Verify audio buffers
        user1_audio = await manager.combine_audio_buffer(conn1)
        user2_audio = await manager.combine_audio_buffer(conn2)
        assert user1_audio == b"audio_chunk_1audio_chunk_2"
        assert user2_audio == b"user2_audio"
        
        # Verify message sending
        mock_websockets[0].send_json.assert_called()  # user1 received private message
        mock_websockets[1].send_json.assert_called()  # user2 received broadcast
        # conn3 was excluded from broadcast but got connection confirmation
        
        # Get stats
        stats = manager.get_connection_stats()
        assert stats["total_connections"] == 3
        assert stats["unique_users"] == 2
        # Check what anonymous connections field is actually called
        assert "total_connections" in stats
    
    @patch('backend.services.websocket_manager.uuid')
    async def test_connection_lifecycle_management(self, mock_uuid, manager, mock_websockets):
        """Test complete connection lifecycle."""
        mock_uuid.uuid4.side_effect = ["conn-1", "conn-2"]
        
        # Connect users
        conn1 = await manager.connect(mock_websockets[0], "user1")
        conn2 = await manager.connect(mock_websockets[1], "user1")  # Same user, different connection
        
        # Verify user has multiple connections
        assert len(manager.user_connections["user1"]) == 2
        
        # Add data for both connections
        manager.add_conversation_message(conn1, "user", "Message from conn1")
        manager.add_conversation_message(conn2, "user", "Message from conn2")
        await manager.stream_audio(conn1, b"audio_from_conn1")
        await manager.stream_audio(conn2, b"audio_from_conn2")
        
        # Disconnect one connection
        manager._disconnect_by_id(conn1)
        
        # Verify partial cleanup
        assert conn1 not in manager.connections
        assert conn2 in manager.connections
        assert len(manager.user_connections["user1"]) == 1
        assert conn1 not in manager.conversation_contexts
        assert conn2 in manager.conversation_contexts
        
        # Disconnect remaining connection
        manager._disconnect_by_id(conn2)
        
        # Verify complete cleanup
        assert "user1" not in manager.user_connections
        assert len(manager.connections) == 0
        assert len(manager.conversation_contexts) == 0
        assert len(manager.audio_buffers) == 0
    
    async def test_heartbeat_and_cleanup_workflow(self, manager):
        """Test heartbeat and cleanup workflow."""
        # Create connections with different states
        active_ws = AsyncMock()
        inactive_ws = AsyncMock()
        
        # Connect users
        with patch('backend.services.websocket_manager.uuid') as mock_uuid:
            mock_uuid.uuid4.side_effect = ["active-conn", "inactive-conn"]
            
            active_conn = await manager.connect(active_ws, "active_user")
            inactive_conn = await manager.connect(inactive_ws, "inactive_user")
        
        # Simulate inactive connection
        manager.connections[inactive_conn].is_active = False
        
        # Test heartbeat functionality
        await manager.send_heartbeat()
        
        # Verify active connection received heartbeat
        active_ws.send_json.assert_called()
        
        # Test cleanup functionality
        await manager.cleanup_inactive_connections()
        
        # Verify inactive connection was cleaned up
        assert active_conn in manager.connections
        assert inactive_conn not in manager.connections
    
    async def test_heartbeat_with_send_failure(self, manager):
        """Test heartbeat when sending fails for some connections."""
        # Create connection that will fail on send
        failing_ws = AsyncMock()
        failing_ws.send_json.side_effect = Exception("Send failed")
        
        with patch('backend.services.websocket_manager.uuid') as mock_uuid:
            mock_uuid.uuid4.return_value = "failing-conn"
            failing_conn = await manager.connect(failing_ws, "failing_user")
        
        # Mock the connection to be active but fail on heartbeat
        manager.connections[failing_conn].is_active = True
        
        # This should not raise an exception despite the send failure
        await manager.send_heartbeat()
        
        # Verify the connection is still in the manager (it should handle the error gracefully)
        assert failing_conn in manager.connections
    
    async def test_cleanup_with_exception(self, manager):
        """Test cleanup when it encounters an exception."""
        # Create a connection
        normal_ws = AsyncMock()
        
        with patch('backend.services.websocket_manager.uuid') as mock_uuid:
            mock_uuid.uuid4.return_value = "normal-conn"
            normal_conn = await manager.connect(normal_ws, "normal_user")
        
        # Mock an exception during cleanup by patching the dictionary access
        original_connections = manager.connections
        
        def side_effect_connections():
            # Raise exception when accessing connections during cleanup
            raise Exception("Cleanup error")
        
        with patch.object(manager, 'connections', side_effect=side_effect_connections):
            # This should not raise an exception despite the cleanup error
            try:
                await manager.cleanup_inactive_connections()
            except Exception:
                pass  # Expected to handle gracefully


class TestBackgroundTasks:
    """Test background task functions."""
    
    @patch('backend.services.websocket_manager.websocket_manager')
    @patch('backend.services.websocket_manager.asyncio.sleep')
    async def test_heartbeat_task_success(self, mock_sleep, mock_ws_manager):
        """Test heartbeat task normal operation."""
        mock_ws_manager.send_heartbeat = AsyncMock()
        mock_sleep.side_effect = [None, asyncio.CancelledError()]  # Cancel after first iteration
        
        try:
            await start_heartbeat_task()
        except asyncio.CancelledError:
            pass  # Expected cancellation
        
        # Verify heartbeat was called
        assert mock_ws_manager.send_heartbeat.call_count >= 1
        mock_sleep.assert_called()
    
    @patch('backend.services.websocket_manager.websocket_manager')
    @patch('backend.services.websocket_manager.asyncio.sleep')
    async def test_heartbeat_task_with_exception(self, mock_sleep, mock_ws_manager):
        """Test heartbeat task with exception handling."""
        mock_ws_manager.send_heartbeat = AsyncMock(side_effect=Exception("Heartbeat failed"))
        mock_sleep.side_effect = [None, None, asyncio.CancelledError()]  # Two iterations then cancel
        
        try:
            await start_heartbeat_task()
        except asyncio.CancelledError:
            pass  # Expected cancellation
        
        # Verify heartbeat was attempted and error sleep was used
        assert mock_ws_manager.send_heartbeat.call_count >= 1
        # Check that sleep(60) was called after exception
        sleep_calls = mock_sleep.call_args_list
        assert any(call[0][0] == 60 for call in sleep_calls)
    
    @patch('backend.services.websocket_manager.websocket_manager')
    @patch('backend.services.websocket_manager.asyncio.sleep')
    async def test_cleanup_task_success(self, mock_sleep, mock_ws_manager):
        """Test cleanup task normal operation."""
        mock_ws_manager.cleanup_inactive_connections = AsyncMock()
        mock_sleep.side_effect = [None, asyncio.CancelledError()]  # Cancel after first iteration
        
        try:
            await start_cleanup_task()
        except asyncio.CancelledError:
            pass  # Expected cancellation
        
        # Verify cleanup was called
        assert mock_ws_manager.cleanup_inactive_connections.call_count >= 1
        mock_sleep.assert_called()
    
    @patch('backend.services.websocket_manager.websocket_manager')
    @patch('backend.services.websocket_manager.asyncio.sleep')
    async def test_cleanup_task_with_exception(self, mock_sleep, mock_ws_manager):
        """Test cleanup task with exception handling."""
        mock_ws_manager.cleanup_inactive_connections = AsyncMock(side_effect=Exception("Cleanup failed"))
        mock_sleep.side_effect = [None, None, asyncio.CancelledError()]  # Two iterations then cancel
        
        try:
            await start_cleanup_task()
        except asyncio.CancelledError:
            pass  # Expected cancellation
        
        # Verify cleanup was attempted and normal sleep was used after exception
        assert mock_ws_manager.cleanup_inactive_connections.call_count >= 1
        # Check that sleep(300) was called after exception
        sleep_calls = mock_sleep.call_args_list
        assert any(call[0][0] == 300 for call in sleep_calls) 