/**
 * WebSocket Manager for AI Avatar Application
 * 
 * Handles real-time communication with the backend including:
 * - Connection management
 * - Audio streaming
 * - Message handling
 * - CV-enhanced AI responses
 */

class WebSocketManager {
    constructor() {
        this.websocket = null;
        this.connectionId = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.heartbeatInterval = null;
        this.messageQueue = [];
        
        // Bind methods
        this.connect = this.connect.bind(this);
        this.disconnect = this.disconnect.bind(this);
        this.sendMessage = this.sendMessage.bind(this);
        this.handleMessage = this.handleMessage.bind(this);
        this.handleError = this.handleError.bind(this);
        this.handleClose = this.handleClose.bind(this);
    }
    
    /**
     * Connect to WebSocket server
     */
    async connect() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            logger.info('Connecting to WebSocket...', wsUrl);
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = this.handleOpen.bind(this);
            this.websocket.onmessage = this.handleMessage.bind(this);
            this.websocket.onerror = this.handleError.bind(this);
            this.websocket.onclose = this.handleClose.bind(this);
            
            // Wait for connection
            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Connection timeout'));
                }, CONFIG.WS_CONNECT_TIMEOUT);
                
                this.websocket.onopen = (event) => {
                    clearTimeout(timeout);
                    this.handleOpen(event);
                    resolve();
                };
                
                this.websocket.onerror = (event) => {
                    clearTimeout(timeout);
                    reject(new Error('Connection failed'));
                };
            });
            
        } catch (error) {
            logger.error('WebSocket connection failed:', error);
            ErrorHandler.handle(error, 'WebSocket connection');
            throw error;
        }
    }
    
    /**
     * Handle WebSocket open event
     */
    handleOpen(event) {
        logger.info('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        // Update UI
        AppState.set('connectionStatus', 'connected');
        UI.updateConnectionStatus('connected');
        
        // Start heartbeat
        this.startHeartbeat();
        
        // Process queued messages
        this.processMessageQueue();
        
        // Notify application
        document.dispatchEvent(new CustomEvent('websocket:connected', {
            detail: { connectionId: this.connectionId }
        }));
    }
    
    /**
     * Handle incoming WebSocket messages
     */
    handleMessage(event) {
        try {
            let data;
            
            if (event.data instanceof Blob) {
                // Handle binary data (audio)
                this.handleBinaryMessage(event.data);
                return;
            } else {
                // Handle text messages
                data = JSON.parse(event.data);
            }
            
            logger.debug('WebSocket message received:', data.type);
            
            switch (data.type) {
                case 'connection_established':
                    this.connectionId = data.connection_id;
                    logger.info('Connection ID:', this.connectionId);
                    break;
                    
                case 'pong':
                    this.handlePong(data);
                    break;
                    
                case 'transcription':
                    this.handleTranscription(data);
                    break;
                    
                case 'llm_response':
                    this.handleLLMResponse(data);
                    break;
                    
                case 'complete_response':
                    this.handleCompleteResponse(data);
                    break;
                    
                case 'audio_received':
                    this.handleAudioReceived(data);
                    break;
                    
                case 'error':
                    this.handleServerError(data);
                    break;
                    
                case 'context_cleared':
                    this.handleContextCleared(data);
                    break;
                    
                case 'conversation_context':
                    this.handleConversationContext(data);
                    break;
                    
                case 'settings_updated':
                    this.handleSettingsUpdated(data);
                    break;
                    
                default:
                    logger.warn('Unknown message type:', data.type);
            }
            
        } catch (error) {
            logger.error('Error handling WebSocket message:', error);
            ErrorHandler.handle(error, 'WebSocket message handling');
        }
    }
    
    /**
     * Handle transcription results
     */
    handleTranscription(data) {
        logger.info('Transcription received:', data.text);
        
        // Update UI with transcription
        UI.addChatMessage('user', data.text);
        
        // Show confidence if available
        if (data.confidence && data.confidence < 0.8) {
            UI.showToast('Low confidence transcription', 'warning');
        }
        
        // Dispatch event
        document.dispatchEvent(new CustomEvent('transcription:received', {
            detail: data
        }));
    }
    
    /**
     * Handle LLM response (CV-enhanced Henrique personality)
     */
    handleLLMResponse(data) {
        logger.info('LLM response received:', data.text);
        
        // Update UI with AI response
        UI.addChatMessage('ai', data.text);
        
        // Update AI status
        UI.updateAIStatus('responding', 'Henrique is responding...');
        
        // Dispatch event
        document.dispatchEvent(new CustomEvent('llm:response', {
            detail: data
        }));
    }
    
    /**
     * Handle complete response (transcription + LLM + TTS + animation)
     */
    handleCompleteResponse(data) {
        logger.info('Complete response received');
        
        // Update chat with final messages
        if (data.transcription) {
            UI.addChatMessage('user', data.transcription);
        }
        
        if (data.llm_response) {
            UI.addChatMessage('ai', data.llm_response);
        }
        
        // Play avatar video if available
        if (data.video_url) {
            VideoManager.playAvatarVideo(data.video_url);
        }
        
        // Play audio if available
        if (data.audio_url) {
            AudioManager.playAvatarAudio(data.audio_url);
        }
        
        // Update AI status
        UI.updateAIStatus('ready', 'Ready');
        
        // Dispatch event
        document.dispatchEvent(new CustomEvent('response:complete', {
            detail: data
        }));
    }
    
    /**
     * Handle binary audio data
     */
    handleBinaryMessage(blob) {
        logger.debug('Binary message received:', blob.size, 'bytes');
        
        // Handle incoming audio data
        // This could be processed audio or avatar audio
        AudioManager.handleIncomingAudio(blob);
    }
    
    /**
     * Handle server errors
     */
    handleServerError(data) {
        logger.error('Server error:', data.message);
        
        UI.showToast(`Server Error: ${data.message}`, 'error');
        UI.updateAIStatus('error', 'Error occurred');
        
        // Dispatch event
        document.dispatchEvent(new CustomEvent('server:error', {
            detail: data
        }));
    }
    
    /**
     * Handle WebSocket errors
     */
    handleError(event) {
        logger.error('WebSocket error:', event);
        ErrorHandler.handle(new Error('WebSocket error'), 'WebSocket');
        
        UI.updateConnectionStatus('error');
        UI.showToast('Connection error occurred', 'error');
    }
    
    /**
     * Handle WebSocket close
     */
    handleClose(event) {
        logger.info('WebSocket closed:', event.code, event.reason);
        
        this.isConnected = false;
        this.connectionId = null;
        
        // Stop heartbeat
        this.stopHeartbeat();
        
        // Update UI
        UI.updateConnectionStatus('disconnected');
        AppState.set('connectionStatus', 'disconnected');
        
        // Attempt reconnection if not intentional
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
        }
        
        // Dispatch event
        document.dispatchEvent(new CustomEvent('websocket:disconnected', {
            detail: { code: event.code, reason: event.reason }
        }));
    }
    
    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        
        logger.info(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
        
        UI.showToast(`Reconnecting in ${delay/1000}s...`, 'warning');
        
        setTimeout(() => {
            if (!this.isConnected) {
                this.connect().catch(error => {
                    logger.error('Reconnection failed:', error);
                });
            }
        }, delay);
    }
    
    /**
     * Send message to server
     */
    sendMessage(message) {
        try {
            if (this.isConnected && this.websocket) {
                if (message instanceof ArrayBuffer || message instanceof Blob) {
                    // Send binary data
                    this.websocket.send(message);
                } else {
                    // Send JSON message
                    this.websocket.send(JSON.stringify(message));
                }
                
                logger.debug('Message sent:', message.type || 'binary');
                return true;
            } else {
                // Queue message for later
                logger.warn('WebSocket not connected, queueing message');
                this.messageQueue.push(message);
                return false;
            }
        } catch (error) {
            logger.error('Error sending message:', error);
            ErrorHandler.handle(error, 'WebSocket send');
            return false;
        }
    }
    
    /**
     * Send audio data to server
     */
    sendAudio(audioData) {
        return this.sendMessage(audioData);
    }
    
    /**
     * Send text input to server
     */
    sendTextInput(text) {
        return this.sendMessage({
            type: 'text_input',
            text: text,
            timestamp: new Date().toISOString()
        });
    }
    
    /**
     * Send ping to server
     */
    sendPing() {
        return this.sendMessage({
            type: 'ping',
            timestamp: new Date().toISOString()
        });
    }
    
    /**
     * Clear conversation context
     */
    clearContext() {
        return this.sendMessage({
            type: 'clear_context',
            timestamp: new Date().toISOString()
        });
    }
    
    /**
     * Request conversation context
     */
    getContext() {
        return this.sendMessage({
            type: 'get_context',
            timestamp: new Date().toISOString()
        });
    }
    
    /**
     * Update settings
     */
    updateSettings(settings) {
        return this.sendMessage({
            type: 'settings',
            settings: settings,
            timestamp: new Date().toISOString()
        });
    }
    
    /**
     * Process queued messages
     */
    processMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.sendMessage(message);
        }
    }
    
    /**
     * Start heartbeat to keep connection alive
     */
    startHeartbeat() {
        this.stopHeartbeat();
        
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected) {
                this.sendPing();
            }
        }, CONFIG.WS_HEARTBEAT_INTERVAL);
    }
    
    /**
     * Stop heartbeat
     */
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
    
    /**
     * Handle pong response
     */
    handlePong(data) {
        logger.debug('Pong received');
        
        // Calculate latency
        const now = new Date().toISOString();
        const latency = new Date(now) - new Date(data.timestamp);
        
        AppState.set('latency', latency);
        UI.updateLatencyDisplay(latency);
    }
    
    /**
     * Handle context cleared
     */
    handleContextCleared(data) {
        logger.info('Conversation context cleared');
        UI.showToast('Conversation cleared', 'success');
        UI.clearChat();
    }
    
    /**
     * Handle conversation context
     */
    handleConversationContext(data) {
        logger.info('Conversation context received:', data.context.length, 'messages');
        
        // Update chat with context
        UI.loadConversationContext(data.context);
    }
    
    /**
     * Handle settings updated
     */
    handleSettingsUpdated(data) {
        logger.info('Settings updated');
        UI.showToast('Settings saved', 'success');
    }
    
    /**
     * Handle audio received confirmation
     */
    handleAudioReceived(data) {
        logger.debug('Audio received confirmation, buffer size:', data.buffer_size);
        
        // Update audio status
        UI.updateAudioStatus('processing');
    }
    
    /**
     * Disconnect from server
     */
    disconnect() {
        if (this.websocket) {
            logger.info('Disconnecting WebSocket...');
            this.websocket.close(1000, 'User disconnected');
            this.websocket = null;
        }
        
        this.stopHeartbeat();
        this.isConnected = false;
        this.connectionId = null;
    }
    
    /**
     * Get connection status
     */
    getStatus() {
        return {
            connected: this.isConnected,
            connectionId: this.connectionId,
            reconnectAttempts: this.reconnectAttempts
        };
    }
}

// Create global WebSocket manager instance
const wsManager = new WebSocketManager(); 