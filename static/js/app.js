/**
 * HoloRay Motion-Tracked Annotation System
 * Click = set anchor, Shift+Drag = draw (matches demo_cli.py)
 */

class HoloRayApp {
    constructor() {
        this.socket = null;
        this.canvas = document.getElementById('videoCanvas');
        this.ctx = this.canvas.getContext('2d');

        // State
        this.isPlaying = false;
        this.isConnected = false;
        this.hasAnchor = false;
        this.isDrawing = false;

        // Video info
        this.videoLoaded = false;
        this.totalFrames = 0;
        this.currentFrame = 0;
        this.fps = 30;

        // DOM elements
        this.elements = {
            connectionStatus: document.getElementById('connectionStatus'),
            videoSelect: document.getElementById('videoSelect'),
            loadBtn: document.getElementById('loadBtn'),
            playBtn: document.getElementById('playBtn'),
            restartBtn: document.getElementById('restartBtn'),
            clearBtn: document.getElementById('clearBtn'),
            progressBar: document.getElementById('progressBar'),
            currentTime: document.getElementById('currentTime'),
            totalTime: document.getElementById('totalTime'),
            speedSelect: document.getElementById('speedSelect'),
            canvasOverlay: document.getElementById('canvasOverlay'),
            trackingBadge: document.getElementById('trackingBadge'),
            trackingStatus: document.getElementById('trackingStatus'),
            trackingPoints: document.getElementById('trackingPoints'),
            fpsValue: document.getElementById('fpsValue'),
            frameValue: document.getElementById('frameValue'),
            statusMessage: document.getElementById('statusMessage')
        };

        this.init();
    }

    init() {
        this.connectSocket();
        this.bindEvents();
    }

    connectSocket() {
        this.socket = io();

        this.socket.on('connect', () => {
            this.isConnected = true;
            this.updateConnectionStatus('connected');
            this.loadVideoList();
        });

        this.socket.on('disconnect', () => {
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
        });

        this.socket.on('status', (data) => {
            if (data.playing !== undefined) {
                this.isPlaying = data.playing;
                this.updatePlayButton();
            }
        });

        this.socket.on('frame', (data) => {
            this.renderFrame(data);
        });

        this.socket.on('first_frame', (data) => {
            this.renderFirstFrame(data.image);
        });

        this.socket.on('annotations_cleared', () => {
            this.hasAnchor = false;
            this.updateStatus();
        });
    }

    updateConnectionStatus(status) {
        const el = this.elements.connectionStatus;
        el.classList.remove('connected', 'error');
        if (status === 'connected') {
            el.classList.add('connected');
            el.querySelector('.status-text').textContent = 'Connected';
        } else {
            el.classList.add('error');
            el.querySelector('.status-text').textContent = 'Disconnected';
        }
    }

    bindEvents() {
        this.elements.loadBtn.addEventListener('click', () => this.loadVideo());
        this.elements.playBtn.addEventListener('click', () => this.togglePlay());
        this.elements.restartBtn.addEventListener('click', () => this.restart());
        this.elements.clearBtn.addEventListener('click', () => this.clearAnnotations());
        this.elements.progressBar.addEventListener('input', (e) => this.seek(e.target.value));
        this.elements.speedSelect.addEventListener('change', (e) => {
            this.socket.emit('set_speed', { speed: parseFloat(e.target.value) });
        });

        // Canvas events - Click = anchor, Shift+Drag = draw
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', () => this.handleMouseUp());
        this.canvas.addEventListener('mouseleave', () => this.handleMouseUp());

        // Keyboard
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            if (e.key === ' ') {
                e.preventDefault();
                this.togglePlay();
            } else if (e.key.toLowerCase() === 'c') {
                this.clearAnnotations();
            }
        });
    }

    async loadVideoList() {
        try {
            const response = await fetch('/api/videos');
            const videos = await response.json();
            this.elements.videoSelect.innerHTML = '<option value="">Select a video...</option>';
            videos.forEach(video => {
                const option = document.createElement('option');
                option.value = video;
                option.textContent = video;
                this.elements.videoSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load video list:', error);
        }
    }

    async loadVideo() {
        const videoName = this.elements.videoSelect.value;
        if (!videoName) return;

        try {
            const response = await fetch('/api/load_video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video: videoName })
            });

            const data = await response.json();
            if (data.success) {
                this.videoLoaded = true;
                this.totalFrames = data.total_frames;
                this.fps = data.fps;
                this.canvas.width = data.width;
                this.canvas.height = data.height;

                this.elements.canvasOverlay.classList.add('hidden');
                this.elements.progressBar.max = this.totalFrames;
                this.updateTimeDisplay();

                this.hasAnchor = false;
                this.updateStatus();
                this.socket.emit('get_first_frame');
            }
        } catch (error) {
            console.error('Failed to load video:', error);
        }
    }

    togglePlay() {
        if (!this.videoLoaded) return;
        if (this.isPlaying) {
            this.socket.emit('pause');
        } else {
            this.socket.emit('play');
        }
        this.isPlaying = !this.isPlaying;
        this.updatePlayButton();
    }

    updatePlayButton() {
        this.elements.playBtn.classList.toggle('playing', this.isPlaying);
    }

    restart() {
        if (!this.videoLoaded) return;
        this.socket.emit('seek', { frame: 0 });
        this.currentFrame = 0;
        this.updateProgress();
    }

    seek(frame) {
        if (!this.videoLoaded) return;
        this.socket.emit('seek', { frame: parseInt(frame) });
        this.currentFrame = parseInt(frame);
        this.updateTimeDisplay();
    }

    renderFrame(data) {
        const img = new Image();
        img.onload = () => this.ctx.drawImage(img, 0, 0);
        img.src = 'data:image/jpeg;base64,' + data.image;

        if (data.tracking) {
            this.updateTrackingInfo(data.tracking);
        }

        this.currentFrame = data.frame_num;
        this.totalFrames = data.total_frames;
        this.updateProgress();
    }

    renderFirstFrame(imageData) {
        const img = new Image();
        img.onload = () => this.ctx.drawImage(img, 0, 0);
        img.src = 'data:image/jpeg;base64,' + imageData;
    }

    updateProgress() {
        this.elements.progressBar.value = this.currentFrame;
        this.updateTimeDisplay();
    }

    updateTimeDisplay() {
        const currentSec = Math.floor(this.currentFrame / this.fps);
        const totalSec = Math.floor(this.totalFrames / this.fps);
        this.elements.currentTime.textContent = this.formatTime(currentSec);
        this.elements.totalTime.textContent = this.formatTime(totalSec);
        this.elements.frameValue.textContent = `${this.currentFrame}/${this.totalFrames}`;
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // Canvas interaction - matches demo_cli.py behavior
    getCanvasCoords(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: Math.round((e.clientX - rect.left) * scaleX),
            y: Math.round((e.clientY - rect.top) * scaleY)
        };
    }

    handleMouseDown(e) {
        if (!this.videoLoaded) return;
        const coords = this.getCanvasCoords(e);

        if (e.shiftKey && this.hasAnchor) {
            // Shift+Click = start drawing
            this.isDrawing = true;
            this.socket.emit('start_drawing');
            this.socket.emit('draw_point', { x: coords.x, y: coords.y });
        } else if (!e.shiftKey) {
            // Regular click = set anchor
            this.socket.emit('add_point', { x: coords.x, y: coords.y });
            this.hasAnchor = true;
            this.updateStatus();
        }
    }

    handleMouseMove(e) {
        if (!this.isDrawing || !this.hasAnchor) return;
        const coords = this.getCanvasCoords(e);
        this.socket.emit('draw_point', { x: coords.x, y: coords.y });
    }

    handleMouseUp() {
        if (!this.isDrawing) return;
        this.socket.emit('end_stroke');
        this.isDrawing = false;
    }

    updateTrackingInfo(data) {
        const badge = this.elements.trackingBadge;

        if (data.is_lost) {
            badge.classList.add('visible', 'lost');
            badge.classList.remove('tracking');
            badge.querySelector('.badge-text').textContent = 'SEARCHING';
            this.elements.trackingStatus.textContent = 'Lost';
            this.elements.trackingStatus.classList.add('lost');
        } else if (data.tracking_points > 0) {
            badge.classList.add('visible', 'tracking');
            badge.classList.remove('lost');
            badge.querySelector('.badge-text').textContent = 'TRACKING';
            this.elements.trackingStatus.textContent = 'Active';
            this.elements.trackingStatus.classList.remove('lost');
        } else {
            badge.classList.remove('visible');
            this.elements.trackingStatus.textContent = '-';
        }

        this.elements.trackingPoints.textContent = data.tracking_points || 0;
        this.elements.fpsValue.textContent = (data.fps || 0).toFixed(1);
    }

    updateStatus() {
        if (!this.videoLoaded) {
            this.elements.statusMessage.textContent = 'Load a video to begin';
        } else if (!this.hasAnchor) {
            this.elements.statusMessage.textContent = 'Click to set anchor point';
        } else {
            this.elements.statusMessage.textContent = 'Anchor set! Shift+Drag to draw';
        }
    }

    clearAnnotations() {
        this.socket.emit('clear_annotations');
        this.hasAnchor = false;
        this.updateStatus();
    }
}

const app = new HoloRayApp();
