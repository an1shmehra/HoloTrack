/**
 * HoloRay Motion-Tracked Annotation System - Frontend
 */

class HoloRayApp {
    constructor() {
        // Socket connection
        this.socket = io();

        // Canvas elements
        this.videoCanvas = document.getElementById('video-canvas');
        this.annotationCanvas = document.getElementById('annotation-canvas');
        this.videoCtx = this.videoCanvas.getContext('2d');
        this.annotationCtx = this.annotationCanvas.getContext('2d');

        // State
        this.currentTool = 'point';
        this.isPlaying = false;
        this.videoLoaded = false;
        this.annotations = {};
        this.trackingData = {};

        // Region drawing state
        this.isDrawingRegion = false;
        this.regionStart = null;

        // Performance tracking
        this.lastFrameTime = Date.now();
        this.frameCount = 0;

        this.init();
    }

    init() {
        this.setupSocketHandlers();
        this.setupEventListeners();
        this.loadVideoList();
    }

    setupSocketHandlers() {
        this.socket.on('connect', () => {
            this.updateConnectionStatus(true);
            this.setStatus('Connected to server');
        });

        this.socket.on('disconnect', () => {
            this.updateConnectionStatus(false);
            this.setStatus('Disconnected from server');
        });

        this.socket.on('status', (data) => {
            if (data.message) {
                this.setStatus(data.message);
            }
            if (data.playing !== undefined) {
                this.isPlaying = data.playing;
                this.updatePlayPauseButtons();
            }
        });

        this.socket.on('frame', (data) => {
            this.displayFrame(data.image);
            this.updateTrackingData(data.tracking);
            this.updateProgressBar(data.frame_num, data.total_frames);
            this.updateStats(data.tracking);
        });

        this.socket.on('first_frame', (data) => {
            this.displayFrame(data.image);
            document.getElementById('video-placeholder').style.display = 'none';
        });

        this.socket.on('annotation_added', (data) => {
            this.annotations[data.id] = data;
            this.updateAnnotationsList();
        });

        this.socket.on('annotation_removed', (data) => {
            delete this.annotations[data.id];
            this.updateAnnotationsList();
        });

        this.socket.on('annotations_cleared', () => {
            this.annotations = {};
            this.updateAnnotationsList();
        });
    }

    setupEventListeners() {
        // Video controls
        document.getElementById('btn-play').addEventListener('click', () => this.play());
        document.getElementById('btn-pause').addEventListener('click', () => this.pause());
        document.getElementById('btn-load').addEventListener('click', () => this.loadSelectedVideo());

        // Progress bar
        document.getElementById('progress-bar').addEventListener('input', (e) => {
            if (this.videoLoaded) {
                this.socket.emit('seek', { frame: parseInt(e.target.value) });
            }
        });

        // Tool selection
        document.getElementById('tool-point').addEventListener('click', () => this.selectTool('point'));
        document.getElementById('tool-region').addEventListener('click', () => this.selectTool('region'));

        // Clear annotations
        document.getElementById('btn-clear-all').addEventListener('click', () => {
            this.socket.emit('clear_annotations');
        });

        // Canvas click for annotations
        const container = document.getElementById('video-container');
        container.addEventListener('mousedown', (e) => this.handleCanvasMouseDown(e));
        container.addEventListener('mousemove', (e) => this.handleCanvasMouseMove(e));
        container.addEventListener('mouseup', (e) => this.handleCanvasMouseUp(e));
    }

    async loadVideoList() {
        try {
            const response = await fetch('/api/videos');
            const videos = await response.json();

            const select = document.getElementById('video-select');
            videos.forEach(video => {
                const option = document.createElement('option');
                option.value = video;
                option.textContent = video;
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load video list:', error);
        }
    }

    async loadSelectedVideo() {
        const select = document.getElementById('video-select');
        const video = select.value;

        if (!video) {
            this.setStatus('Please select a video');
            return;
        }

        this.setStatus('Loading video...');

        try {
            const response = await fetch('/api/load_video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video })
            });

            const data = await response.json();

            if (data.success) {
                this.videoLoaded = true;
                this.resizeCanvas(data.width, data.height);

                // Enable controls
                document.getElementById('btn-play').disabled = false;
                document.getElementById('btn-pause').disabled = false;
                document.getElementById('progress-bar').disabled = false;
                document.getElementById('progress-bar').max = data.total_frames;

                // Request first frame
                this.socket.emit('get_first_frame');

                this.setStatus(`Loaded: ${video} (${data.width}x${data.height} @ ${data.fps.toFixed(1)} FPS)`);
            } else {
                this.setStatus(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Failed to load video:', error);
            this.setStatus('Failed to load video');
        }
    }

    resizeCanvas(width, height) {
        // Calculate scaled dimensions to fit container
        const container = document.getElementById('video-container');
        const maxWidth = container.clientWidth - 20;
        const maxHeight = container.clientHeight - 20;

        let scale = Math.min(maxWidth / width, maxHeight / height);
        scale = Math.min(scale, 1); // Don't upscale

        const displayWidth = Math.floor(width * scale);
        const displayHeight = Math.floor(height * scale);

        this.videoCanvas.width = width;
        this.videoCanvas.height = height;
        this.videoCanvas.style.width = displayWidth + 'px';
        this.videoCanvas.style.height = displayHeight + 'px';

        this.annotationCanvas.width = width;
        this.annotationCanvas.height = height;
        this.annotationCanvas.style.width = displayWidth + 'px';
        this.annotationCanvas.style.height = displayHeight + 'px';

        // Store scale for coordinate conversion
        this.displayScale = scale;
        this.videoWidth = width;
        this.videoHeight = height;
    }

    displayFrame(base64Image) {
        const img = new Image();
        img.onload = () => {
            this.videoCtx.drawImage(img, 0, 0);
        };
        img.src = 'data:image/jpeg;base64,' + base64Image;
    }

    play() {
        this.socket.emit('play');
        this.isPlaying = true;
        this.updatePlayPauseButtons();
    }

    pause() {
        this.socket.emit('pause');
        this.isPlaying = false;
        this.updatePlayPauseButtons();
    }

    updatePlayPauseButtons() {
        const playBtn = document.getElementById('btn-play');
        const pauseBtn = document.getElementById('btn-pause');

        if (this.isPlaying) {
            playBtn.style.opacity = '0.5';
            pauseBtn.style.opacity = '1';
        } else {
            playBtn.style.opacity = '1';
            pauseBtn.style.opacity = '0.5';
        }
    }

    selectTool(tool) {
        this.currentTool = tool;

        document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.getElementById(`tool-${tool}`).classList.add('active');
    }

    getCanvasCoordinates(e) {
        const rect = this.videoCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.displayScale;
        const y = (e.clientY - rect.top) / this.displayScale;
        return { x: Math.round(x), y: Math.round(y) };
    }

    handleCanvasMouseDown(e) {
        if (!this.videoLoaded) return;

        const coords = this.getCanvasCoordinates(e);

        if (this.currentTool === 'point') {
            this.addPointAnnotation(coords.x, coords.y);
        } else if (this.currentTool === 'region') {
            this.isDrawingRegion = true;
            this.regionStart = coords;
        }
    }

    handleCanvasMouseMove(e) {
        if (!this.isDrawingRegion) return;

        const coords = this.getCanvasCoordinates(e);
        this.drawRegionPreview(this.regionStart, coords);
    }

    handleCanvasMouseUp(e) {
        if (!this.isDrawingRegion) return;

        const coords = this.getCanvasCoordinates(e);
        this.isDrawingRegion = false;

        // Clear preview
        this.annotationCtx.clearRect(0, 0, this.annotationCanvas.width, this.annotationCanvas.height);

        // Calculate bounding box
        const x = Math.min(this.regionStart.x, coords.x);
        const y = Math.min(this.regionStart.y, coords.y);
        const width = Math.abs(coords.x - this.regionStart.x);
        const height = Math.abs(coords.y - this.regionStart.y);

        if (width > 10 && height > 10) {
            this.addRegionAnnotation(x, y, width, height);
        }

        this.regionStart = null;
    }

    drawRegionPreview(start, end) {
        this.annotationCtx.clearRect(0, 0, this.annotationCanvas.width, this.annotationCanvas.height);

        const x = Math.min(start.x, end.x);
        const y = Math.min(start.y, end.y);
        const width = Math.abs(end.x - start.x);
        const height = Math.abs(end.y - start.y);

        this.annotationCtx.strokeStyle = '#00d4aa';
        this.annotationCtx.lineWidth = 2;
        this.annotationCtx.setLineDash([5, 5]);
        this.annotationCtx.strokeRect(x, y, width, height);
        this.annotationCtx.setLineDash([]);
    }

    addPointAnnotation(x, y) {
        const label = document.getElementById('annotation-label').value || '';
        const id = 'point_' + Date.now();

        this.socket.emit('add_point', {
            id: id,
            x: x,
            y: y,
            label: label
        });

        this.setStatus(`Added point annotation at (${x}, ${y})`);
    }

    addRegionAnnotation(x, y, width, height) {
        const label = document.getElementById('annotation-label').value || '';
        const trackerType = document.getElementById('tracker-type').value;
        const id = 'region_' + Date.now();

        this.socket.emit('add_region', {
            id: id,
            x: x,
            y: y,
            width: width,
            height: height,
            label: label,
            tracker: trackerType
        });

        this.setStatus(`Added region annotation at (${x}, ${y}) size ${width}x${height}`);
    }

    removeAnnotation(id) {
        this.socket.emit('remove_annotation', { id: id });
    }

    updateTrackingData(data) {
        this.trackingData = data;
    }

    updateAnnotationsList() {
        const list = document.getElementById('annotations-list');

        if (Object.keys(this.annotations).length === 0) {
            list.innerHTML = '<p class="empty-message">No annotations yet</p>';
            return;
        }

        list.innerHTML = '';

        for (const [id, ann] of Object.entries(this.annotations)) {
            const item = document.createElement('div');
            item.className = 'annotation-item';

            const tracking = this.trackingData.points?.[id] || this.trackingData.regions?.[id];
            const isTracking = tracking?.state === 'tracking';

            item.innerHTML = `
                <div>
                    <div class="label">${ann.label || id}</div>
                    <div class="type">${ann.type}</div>
                </div>
                <div class="status ${isTracking ? '' : 'lost'}"></div>
                <button class="remove-btn" data-id="${id}">×</button>
            `;

            list.appendChild(item);
        }

        // Add remove listeners
        list.querySelectorAll('.remove-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.removeAnnotation(btn.dataset.id);
            });
        });
    }

    updateStats(tracking) {
        // Update FPS display
        const fpsEl = document.querySelector('#fps-display .stat-value');
        fpsEl.textContent = tracking.fps?.toFixed(1) || '--';

        // Update tracking count
        const pointsTracking = Object.values(tracking.points || {})
            .filter(p => p.state === 'tracking').length;
        const totalPoints = Object.keys(tracking.points || {}).length;
        const regionsTracking = Object.values(tracking.regions || {})
            .filter(r => r.state === 'tracking').length;
        const totalRegions = Object.keys(tracking.regions || {}).length;

        const trackingEl = document.querySelector('#tracking-display .stat-value');
        trackingEl.textContent = `${pointsTracking + regionsTracking}/${totalPoints + totalRegions}`;

        // Update latency
        const latencyEl = document.querySelector('#latency-display .stat-value');
        latencyEl.textContent = `${tracking.process_time_ms?.toFixed(1) || '--'}ms`;
    }

    updateProgressBar(current, total) {
        const bar = document.getElementById('progress-bar');
        bar.value = current;

        const counter = document.getElementById('frame-counter');
        counter.textContent = `${current} / ${total}`;
    }

    updateConnectionStatus(connected) {
        const el = document.getElementById('connection-status');
        el.className = `connection ${connected ? 'connected' : 'disconnected'}`;
        el.textContent = connected ? 'Connected' : 'Disconnected';
    }

    setStatus(message) {
        document.getElementById('status-message').textContent = message;
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new HoloRayApp();
});
