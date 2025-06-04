/**
 * Utility functions for AI Avatar Meet application
 */

// Global app state
window.AppState = {
    connectionId: null,
    isConnected: false,
    isRecording: false,
    isCameraOn: true,
    isMicOn: true,
    currentStream: null,
    settings: {
        voiceSpeed: 1.0,
        responseLength: 'medium',
        autoSpeechDetection: true,
        noiseCancellation: true,
        videoQuality: '720p'
    }
};

// Configuration
const CONFIG = {
    WS_URL: window.location.protocol === 'https:' ? 'wss://' : 'ws://' + window.location.host + '/ws',
    API_BASE: window.location.origin,
    CHUNK_SIZE: 1024,
    SAMPLE_RATE: 16000,
    CHANNELS: 1,
    PING_INTERVAL: 30000,
    RECONNECT_DELAY: 3000,
    MAX_RECONNECT_ATTEMPTS: 5,
    TOAST_DURATION: 5000
};

/**
 * DOM Utility Functions
 */
const DOM = {
    /**
     * Get element by ID with error handling
     */
    get(id) {
        const element = document.getElementById(id);
        if (!element) {
            console.warn(`Element with id '${id}' not found`);
        }
        return element;
    },

    /**
     * Query selector with error handling
     */
    query(selector) {
        const element = document.querySelector(selector);
        if (!element) {
            console.warn(`Element with selector '${selector}' not found`);
        }
        return element;
    },

    /**
     * Query all with array return
     */
    queryAll(selector) {
        return Array.from(document.querySelectorAll(selector));
    },

    /**
     * Show element
     */
    show(element) {
        if (element) {
            element.style.display = '';
            element.classList.remove('hidden');
        }
    },

    /**
     * Hide element
     */
    hide(element) {
        if (element) {
            element.style.display = 'none';
            element.classList.add('hidden');
        }
    },

    /**
     * Toggle element visibility
     */
    toggle(element) {
        if (element) {
            element.classList.toggle('hidden');
        }
    },

    /**
     * Add event listener with error handling
     */
    on(element, event, handler) {
        if (element && typeof handler === 'function') {
            element.addEventListener(event, handler);
        }
    },

    /**
     * Remove event listener
     */
    off(element, event, handler) {
        if (element && typeof handler === 'function') {
            element.removeEventListener(event, handler);
        }
    },

    /**
     * Create element with attributes and content
     */
    create(tag, attributes = {}, content = '') {
        const element = document.createElement(tag);
        
        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'innerHTML') {
                element.innerHTML = value;
            } else {
                element.setAttribute(key, value);
            }
        });
        
        if (content) {
            element.textContent = content;
        }
        
        return element;
    }
};

/**
 * API Utility Functions
 */
const API = {
    /**
     * Make HTTP request with error handling
     */
    async request(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }
            
            return await response.text();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },

    /**
     * Upload file with progress
     */
    async uploadFile(url, file, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            const formData = new FormData();
            formData.append('file', file);

            xhr.upload.addEventListener('progress', (event) => {
                if (event.lengthComputable && onProgress) {
                    const percentComplete = (event.loaded / event.total) * 100;
                    onProgress(percentComplete);
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        resolve(response);
                    } catch (e) {
                        resolve(xhr.responseText);
                    }
                } else {
                    reject(new Error(`Upload failed: ${xhr.status}`));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('Upload failed'));
            });

            xhr.open('POST', url);
            xhr.send(formData);
        });
    },

    /**
     * Check system status
     */
    async checkStatus() {
        try {
            const health = await this.request(`${CONFIG.API_BASE}/health`);
            const models = await this.request(`${CONFIG.API_BASE}/models/status`);
            return { health, models };
        } catch (error) {
            console.error('Status check failed:', error);
            return null;
        }
    }
};

/**
 * Audio Utility Functions
 */
const AudioUtils = {
    /**
     * Get user media with constraints
     */
    async getUserMedia(constraints = {}) {
        const defaultConstraints = {
            audio: {
                sampleRate: CONFIG.SAMPLE_RATE,
                channelCount: CONFIG.CHANNELS,
                echoCancellation: true,
                noiseSuppression: AppState.settings.noiseCancellation,
                autoGainControl: true
            },
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 }
            }
        };

        const mergedConstraints = this.mergeDeep(defaultConstraints, constraints);

        try {
            return await navigator.mediaDevices.getUserMedia(mergedConstraints);
        } catch (error) {
            console.error('Failed to get user media:', error);
            throw error;
        }
    },

    /**
     * Get audio devices
     */
    async getAudioDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return {
                microphones: devices.filter(device => device.kind === 'audioinput'),
                speakers: devices.filter(device => device.kind === 'audiooutput')
            };
        } catch (error) {
            console.error('Failed to get audio devices:', error);
            return { microphones: [], speakers: [] };
        }
    },

    /**
     * Get video devices
     */
    async getVideoDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (error) {
            console.error('Failed to get video devices:', error);
            return [];
        }
    },

    /**
     * Convert audio buffer to WAV
     */
    audioBufferToWav(buffer, sampleRate = CONFIG.SAMPLE_RATE) {
        const length = buffer.length;
        const arrayBuffer = new ArrayBuffer(44 + length * 2);
        const view = new DataView(arrayBuffer);
        
        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, length * 2, true);
        
        // Convert samples
        let offset = 44;
        for (let i = 0; i < length; i++) {
            const sample = Math.max(-1, Math.min(1, buffer[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
        
        return arrayBuffer;
    },

    /**
     * Merge objects deeply
     */
    mergeDeep(target, source) {
        if (typeof target !== 'object' || typeof source !== 'object') {
            return source;
        }
        
        for (const key in source) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                target[key] = this.mergeDeep(target[key] || {}, source[key]);
            } else {
                target[key] = source[key];
            }
        }
        
        return target;
    }
};

/**
 * Error Handling Utilities
 */
const ErrorHandler = {
    /**
     * Handle and display errors
     */
    handle(error, context = '') {
        console.error(`Error in ${context}:`, error);
        
        let message = 'An unexpected error occurred';
        
        if (error.name === 'NotAllowedError') {
            message = 'Permission denied. Please allow microphone and camera access.';
        } else if (error.name === 'NotFoundError') {
            message = 'No microphone or camera found.';
        } else if (error.name === 'NotSupportedError') {
            message = 'Audio/video not supported in this browser.';
        } else if (error.message) {
            message = error.message;
        }
        
        this.showToast(message, 'error');
        return message;
    },

    /**
     * Show toast notification
     */
    showToast(message, type = 'info', duration = CONFIG.TOAST_DURATION) {
        const container = DOM.get('toast-container');
        if (!container) return;
        
        const toast = DOM.create('div', {
            className: `toast ${type}`
        }, message);
        
        container.appendChild(toast);
        
        // Auto-remove toast
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, duration);
        
        // Click to dismiss
        DOM.on(toast, 'click', () => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        });
    }
};

/**
 * Storage Utilities
 */
const Storage = {
    /**
     * Save data to localStorage
     */
    save(key, data) {
        try {
            localStorage.setItem(key, JSON.stringify(data));
        } catch (error) {
            console.warn('Failed to save to localStorage:', error);
        }
    },

    /**
     * Load data from localStorage
     */
    load(key, defaultValue = null) {
        try {
            const data = localStorage.getItem(key);
            return data ? JSON.parse(data) : defaultValue;
        } catch (error) {
            console.warn('Failed to load from localStorage:', error);
            return defaultValue;
        }
    },

    /**
     * Remove data from localStorage
     */
    remove(key) {
        try {
            localStorage.removeItem(key);
        } catch (error) {
            console.warn('Failed to remove from localStorage:', error);
        }
    }
};

/**
 * Validation Utilities
 */
const Validator = {
    /**
     * Validate email format
     */
    email(email) {
        const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return regex.test(email);
    },

    /**
     * Validate required field
     */
    required(value) {
        return value !== null && value !== undefined && value.toString().trim() !== '';
    },

    /**
     * Validate string length
     */
    length(value, min = 0, max = Infinity) {
        const length = value ? value.toString().length : 0;
        return length >= min && length <= max;
    },

    /**
     * Validate number range
     */
    range(value, min = -Infinity, max = Infinity) {
        const num = parseFloat(value);
        return !isNaN(num) && num >= min && num <= max;
    }
};

/**
 * Performance Utilities
 */
const Performance = {
    /**
     * Debounce function calls
     */
    debounce(func, wait, immediate = false) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                timeout = null;
                if (!immediate) func(...args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func(...args);
        };
    },

    /**
     * Throttle function calls
     */
    throttle(func, limit) {
        let inThrottle;
        return function executedFunction(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    /**
     * Measure execution time
     */
    measure(name, func) {
        return async function(...args) {
            const start = performance.now();
            const result = await func.apply(this, args);
            const end = performance.now();
            console.log(`${name} took ${end - start} milliseconds`);
            return result;
        };
    }
};

/**
 * Format Utilities
 */
const Format = {
    /**
     * Format time duration
     */
    duration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    },

    /**
     * Format file size
     */
    fileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Format timestamp
     */
    timestamp(date = new Date()) {
        return date.toLocaleTimeString(undefined, {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    },

    /**
     * Truncate text
     */
    truncate(text, length = 100, suffix = '...') {
        if (text.length <= length) return text;
        return text.substring(0, length) + suffix;
    }
};

/**
 * Animation Utilities
 */
const Animation = {
    /**
     * Animate element property
     */
    animate(element, property, from, to, duration = 300, easing = 'ease') {
        if (!element) return;
        
        element.style.transition = `${property} ${duration}ms ${easing}`;
        element.style[property] = from;
        
        requestAnimationFrame(() => {
            element.style[property] = to;
        });
        
        setTimeout(() => {
            element.style.transition = '';
        }, duration);
    },

    /**
     * Fade in element
     */
    fadeIn(element, duration = 300) {
        this.animate(element, 'opacity', '0', '1', duration);
        DOM.show(element);
    },

    /**
     * Fade out element
     */
    fadeOut(element, duration = 300) {
        this.animate(element, 'opacity', '1', '0', duration);
        setTimeout(() => DOM.hide(element), duration);
    },

    /**
     * Slide in element
     */
    slideIn(element, direction = 'down', duration = 300) {
        const property = direction === 'down' || direction === 'up' ? 'translateY' : 'translateX';
        const from = direction === 'down' || direction === 'right' ? '-100%' : '100%';
        
        element.style.transform = `${property}(${from})`;
        DOM.show(element);
        
        requestAnimationFrame(() => {
            this.animate(element, 'transform', `${property}(${from})`, `${property}(0)`, duration);
        });
    }
};

// Initialize utilities when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('AI Avatar Meet utilities loaded');
    
    // Load saved settings
    const savedSettings = Storage.load('aiAvatarSettings');
    if (savedSettings) {
        Object.assign(AppState.settings, savedSettings);
    }
    
    // Save settings when changed
    const saveSettings = Performance.debounce(() => {
        Storage.save('aiAvatarSettings', AppState.settings);
    }, 1000);
    
    // Monitor settings changes
    const originalSettings = JSON.stringify(AppState.settings);
    setInterval(() => {
        const currentSettings = JSON.stringify(AppState.settings);
        if (currentSettings !== originalSettings) {
            saveSettings();
        }
    }, 5000);
});

// Export utilities for global access
window.DOM = DOM;
window.API = API;
window.AudioUtils = AudioUtils;
window.ErrorHandler = ErrorHandler;
window.Storage = Storage;
window.Validator = Validator;
window.Performance = Performance;
window.Format = Format;
window.Animation = Animation;
window.CONFIG = CONFIG; 