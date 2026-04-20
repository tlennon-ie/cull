"""
Pipeline Dashboard - Professional Flask Web Application
Real-time monitoring of prompt-library pipeline operations

DEPENDENCIES (install via pip):
    flask==3.0.0
    flask-cors==4.0.0
    werkzeug==3.0.1
    python-dotenv==1.0.0

OPTIONAL (recommended):
    gunicorn==21.2.0  # For production deployment
    python-json-logger==2.0.7  # For structured logging

USAGE:
    python dashboard.py
    Open http://localhost:5000 in your browser

REQUIREMENTS:
    - Python 3.8+
    - Read/write access to I:\AI\openclaw\workspace\prompt-library
"""

import os
import json
import time
import logging
import threading
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from functools import wraps
from typing import Dict, List, Tuple, Any

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS

# ============================================================================
# CONFIGURATION
# ============================================================================

WORKSPACE_ROOT = r"I:\AI\openclaw\workspace"
PROMPT_LIBRARY_ROOT = os.path.join(WORKSPACE_ROOT, "prompt-library")
QUEUE_ROOT = os.path.join(PROMPT_LIBRARY_ROOT, "queue")
SORTED_ROOT = os.path.join(PROMPT_LIBRARY_ROOT, "sorted")
LOGS_ROOT = os.path.join(PROMPT_LIBRARY_ROOT, "logs_test")

# Scraper names
SCRAPERS = ["civitai", "twitter_x", "reddit", "nanobanana", "discord_ud"]

# Projects
PROJECTS = ["realistic_female_influencer"]

# Settings
SCAN_INTERVAL = 5  # seconds
RECENT_ITEMS_LIMIT = 5
WORKER_TIMEOUT = 30  # seconds to consider worker inactive

# ============================================================================
# SETUP
# ============================================================================

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA CACHE & STATE
# ============================================================================

class PipelineState:
    """Manages real-time pipeline state"""
    
    def __init__(self):
        self.lock = threading.RLock()
        self.scraper_status = {scraper: {"enabled": True} for scraper in SCRAPERS}
        self.stats = {}
        self.vision_workers = deque(maxlen=RECENT_ITEMS_LIMIT)
        self.last_scan = 0
        self.scan_in_progress = False
        
    def get_stats(self) -> Dict:
        with self.lock:
            return dict(self.stats)
    
    def get_vision_workers(self) -> List:
        with self.lock:
            return list(self.vision_workers)
    
    def toggle_scraper(self, scraper: str, enabled: bool):
        with self.lock:
            if scraper in self.scraper_status:
                self.scraper_status[scraper]["enabled"] = enabled
                logger.info(f"Scraper {scraper} {'enabled' if enabled else 'disabled'}")
    
    def add_vision_worker_entry(self, entry: Dict):
        with self.lock:
            self.vision_workers.append({
                **entry,
                "timestamp": datetime.now().isoformat()
            })
    
    def update_stats(self, new_stats: Dict):
        with self.lock:
            self.stats = new_stats

state = PipelineState()

# ============================================================================
# FILE SYSTEM SCANNING
# ============================================================================

def count_files_in_directory(path: str) -> int:
    """Count files in a directory efficiently"""
    try:
        if not os.path.isdir(path):
            return 0
        return sum(1 for _ in Path(path).rglob("*") if _.is_file())
    except Exception as e:
        logger.error(f"Error counting files in {path}: {e}")
        return 0

def get_queue_stats() -> Dict[str, Dict]:
    """Scan queue directory and return per-scraper statistics"""
    stats = {}
    
    for project in PROJECTS:
        project_queue = os.path.join(QUEUE_ROOT, project)
        if not os.path.isdir(project_queue):
            continue
            
        stats[project] = {}
        
        for scraper in SCRAPERS:
            scraper_path = os.path.join(project_queue, scraper)
            if not os.path.isdir(scraper_path):
                stats[project][scraper] = {
                    "count": 0,
                    "enabled": state.scraper_status[scraper]["enabled"],
                    "last_modified": None
                }
                continue
            
            file_count = count_files_in_directory(scraper_path)
            
            # Get last modified time
            try:
                last_modified = max(
                    (os.path.getmtime(os.path.join(root, f))
                     for root, dirs, files in os.walk(scraper_path)
                     for f in files if f.endswith('.json')),
                    default=0
                )
                last_modified = datetime.fromtimestamp(last_modified).isoformat() if last_modified else None
            except Exception:
                last_modified = None
            
            stats[project][scraper] = {
                "count": file_count,
                "enabled": state.scraper_status[scraper]["enabled"],
                "last_modified": last_modified
            }
    
    return stats

def get_sorted_stats() -> Dict[str, Dict]:
    """Scan sorted directory and return per-category statistics"""
    stats = {}
    
    for project in PROJECTS:
        project_sorted = os.path.join(SORTED_ROOT, project)
        if not os.path.isdir(project_sorted):
            continue
        
        stats[project] = {}
        
        try:
            for category in os.listdir(project_sorted):
                category_path = os.path.join(project_sorted, category)
                if os.path.isdir(category_path):
                    file_count = count_files_in_directory(category_path)
                    stats[project][category] = file_count
        except Exception as e:
            logger.error(f"Error scanning sorted directory: {e}")
    
    return stats

def get_vision_queue_recent() -> List[Dict]:
    """Get the 5 most recent items from vision queue (sorted directory)"""
    recent = []
    
    try:
        for project in PROJECTS:
            project_sorted = os.path.join(SORTED_ROOT, project)
            if not os.path.isdir(project_sorted):
                continue
            
            # Collect all files with timestamps
            all_files = []
            for category in os.listdir(project_sorted):
                category_path = os.path.join(project_sorted, category)
                if os.path.isdir(category_path):
                    for root, dirs, files in os.walk(category_path):
                        for file in files:
                            filepath = os.path.join(root, file)
                            try:
                                mtime = os.path.getmtime(filepath)
                                all_files.append({
                                    "filename": file,
                                    "category": category,
                                    "path": filepath,
                                    "mtime": mtime,
                                    "size_kb": os.path.getsize(filepath) / 1024
                                })
                            except Exception:
                                continue
            
            # Sort by modification time (newest first) and take top 5
            all_files.sort(key=lambda x: x["mtime"], reverse=True)
            recent.extend(all_files[:RECENT_ITEMS_LIMIT])
    
    except Exception as e:
        logger.error(f"Error getting vision queue recent items: {e}")
    
    return recent[:RECENT_ITEMS_LIMIT]

def scan_pipeline_state():
    """Comprehensive pipeline state scan"""
    try:
        if state.scan_in_progress:
            return
        
        state.scan_in_progress = True
        
        queue_stats = get_queue_stats()
        sorted_stats = get_sorted_stats()
        vision_recent = get_vision_queue_recent()
        
        # Calculate aggregate statistics
        aggregate = {
            "total_queue_items": sum(
                sum(s.get(scraper, {}).get("count", 0) for scraper in SCRAPERS)
                for p_stats in queue_stats.values()
                for s in [p_stats]
            ),
            "total_sorted_items": sum(
                sum(count for count in categories.values())
                for categories in sorted_stats.values()
            ),
            "queue_by_project": queue_stats,
            "sorted_by_project": sorted_stats,
            "vision_recent": vision_recent,
            "scraper_status": {s: state.scraper_status[s] for s in SCRAPERS},
            "timestamp": datetime.now().isoformat(),
            "active_workers": len(vision_recent)  # Approximation
        }
        
        state.update_stats(aggregate)
        state.last_scan = time.time()
    
    except Exception as e:
        logger.error(f"Error during pipeline scan: {e}")
    
    finally:
        state.scan_in_progress = False

def background_scanner():
    """Background thread that periodically scans the pipeline"""
    while True:
        try:
            scan_pipeline_state()
            time.sleep(SCAN_INTERVAL)
        except Exception as e:
            logger.error(f"Background scanner error: {e}")
            time.sleep(SCAN_INTERVAL)

# ============================================================================
# CLEANUP OPERATIONS
# ============================================================================

def cleanup_orphaned_files() -> Dict[str, Any]:
    """Remove orphaned/corrupted files and return cleanup report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "removed_files": 0,
        "freed_bytes": 0,
        "errors": []
    }
    
    try:
        # Clean up corrupted queue items
        for project in PROJECTS:
            for scraper in SCRAPERS:
                scraper_path = os.path.join(QUEUE_ROOT, project, scraper)
                if not os.path.isdir(scraper_path):
                    continue
                
                for root, dirs, files in os.walk(scraper_path):
                    for file in files:
                        filepath = os.path.join(root, file)
                        try:
                            if file.endswith('.json'):
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    json.load(f)
                            else:
                                # Non-JSON files
                                if os.path.getsize(filepath) == 0:
                                    os.remove(filepath)
                                    report["removed_files"] += 1
                        except (json.JSONDecodeError, IOError) as e:
                            try:
                                file_size = os.path.getsize(filepath)
                                os.remove(filepath)
                                report["removed_files"] += 1
                                report["freed_bytes"] += file_size
                                logger.info(f"Removed corrupted file: {filepath}")
                            except Exception as remove_error:
                                report["errors"].append(str(remove_error))
        
        logger.info(f"Cleanup complete: {report['removed_files']} files removed, {report['freed_bytes']} bytes freed")
    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        report["errors"].append(str(e))
    
    return report

def clear_logs() -> Dict[str, Any]:
    """Clear log files"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "cleared_logs": [],
        "errors": []
    }
    
    try:
        for log_file in os.listdir(LOGS_ROOT):
            log_path = os.path.join(LOGS_ROOT, log_file)
            if log_file.endswith('.log') and os.path.isfile(log_path):
                try:
                    with open(log_path, 'w') as f:
                        f.write("")
                    report["cleared_logs"].append(log_file)
                    logger.info(f"Cleared log file: {log_file}")
                except Exception as e:
                    report["errors"].append(f"Failed to clear {log_file}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error clearing logs: {e}")
        report["errors"].append(str(e))
    
    return report

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route("/")
def dashboard():
    """Serve the main dashboard HTML"""
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get current pipeline statistics"""
    stats = state.get_stats()
    return jsonify(stats)

@app.route("/api/scrapers", methods=["GET"])
def get_scrapers():
    """Get scraper status"""
    return jsonify({
        "scrapers": SCRAPERS,
        "status": {s: state.scraper_status[s] for s in SCRAPERS}
    })

@app.route("/api/scrapers/<scraper>/toggle", methods=["POST"])
def toggle_scraper(scraper):
    """Toggle scraper on/off"""
    if scraper not in SCRAPERS:
        return jsonify({"error": "Unknown scraper"}), 404
    
    data = request.get_json() or {}
    enabled = data.get("enabled", True)
    
    state.toggle_scraper(scraper, enabled)
    
    return jsonify({
        "scraper": scraper,
        "enabled": enabled,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/cleanup/orphaned", methods=["POST"])
def cleanup_orphaned():
    """Run orphaned file cleanup"""
    logger.info("Cleanup initiated")
    report = cleanup_orphaned_files()
    
    # Re-scan after cleanup
    scan_pipeline_state()
    
    return jsonify(report)

@app.route("/api/cleanup/logs", methods=["POST"])
def cleanup_logs():
    """Clear all log files"""
    logger.info("Log cleanup initiated")
    report = clear_logs()
    return jsonify(report)

@app.route("/api/export/stats", methods=["GET"])
def export_stats():
    """Export current statistics as JSON"""
    stats = state.get_stats()
    return jsonify({
        "export": {
            **stats,
            "export_timestamp": datetime.now().isoformat(),
            "export_format": "json",
            "source": "Pipeline Dashboard"
        }
    })

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "last_scan": datetime.fromtimestamp(state.last_scan).isoformat() if state.last_scan else None,
        "version": "1.0.0"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# ============================================================================
# DASHBOARD HTML/CSS/JS
# ============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            margin-bottom: 30px;
            border-bottom: 2px solid #1e1e1e;
            padding-bottom: 20px;
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
            font-size: 0.9em;
            color: #888;
        }

        .status-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            background: #1e1e1e;
            border: 1px solid #2e2e2e;
        }

        .status-badge.healthy {
            background: rgba(76, 175, 80, 0.1);
            border-color: #4caf50;
            color: #4caf50;
        }

        .status-badge.warning {
            background: rgba(255, 193, 7, 0.1);
            border-color: #ffc107;
            color: #ffc107;
        }

        .control-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 0.95em;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }

        .btn-primary {
            background: #667eea;
            color: white;
        }

        .btn-primary:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #2e2e2e;
            color: #e0e0e0;
            border: 1px solid #3e3e3e;
        }

        .btn-secondary:hover {
            background: #3e3e3e;
            border-color: #4e4e4e;
        }

        .btn-danger {
            background: #f44336;
            color: white;
        }

        .btn-danger:hover {
            background: #da190b;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(244, 67, 54, 0.4);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: #1a1a1a;
            border: 1px solid #2e2e2e;
            border-radius: 8px;
            padding: 20px;
            transition: all 0.3s ease;
        }

        .card:hover {
            border-color: #667eea;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
        }

        .card-title {
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 15px;
            color: #e0e0e0;
        }

        .card-subtitle {
            font-size: 0.85em;
            color: #888;
            margin-bottom: 10px;
        }

        .stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #2e2e2e;
        }

        .stat-row:last-child {
            border-bottom: none;
        }

        .stat-label {
            color: #888;
            font-size: 0.9em;
        }

        .stat-value {
            font-size: 1.3em;
            font-weight: 600;
            color: #667eea;
        }

        .stat-value.large {
            font-size: 2em;
        }

        .scraper-toggle {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: #0f0f0f;
            border-radius: 4px;
            margin-bottom: 8px;
        }

        .toggle-switch {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .switch {
            position: relative;
            display: inline-block;
            width: 50px;
            height: 24px;
        }

        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #3e3e3e;
            transition: 0.3s;
            border-radius: 24px;
        }

        .slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: 0.3s;
            border-radius: 50%;
        }

        input:checked + .slider {
            background-color: #667eea;
        }

        input:checked + .slider:before {
            transform: translateX(26px);
        }

        .queue-item {
            background: #0f0f0f;
            border-left: 3px solid #667eea;
            padding: 12px;
            margin-bottom: 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }

        .queue-item-name {
            font-weight: 500;
            color: #e0e0e0;
            margin-bottom: 4px;
            word-break: break-all;
        }

        .queue-item-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.75em;
            color: #888;
            margin-top: 6px;
        }

        .alert {
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            border-left: 4px solid;
        }

        .alert-info {
            background: rgba(33, 150, 243, 0.1);
            border-color: #2196f3;
            color: #2196f3;
        }

        .alert-success {
            background: rgba(76, 175, 80, 0.1);
            border-color: #4caf50;
            color: #4caf50;
        }

        .alert-error {
            background: rgba(244, 67, 54, 0.1);
            border-color: #f44336;
            color: #f44336;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        .grid-full {
            grid-column: 1 / -1;
        }

        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }

            .grid-2 {
                grid-template-columns: 1fr;
            }

            h1 {
                font-size: 1.8em;
            }

            .header-info {
                flex-direction: column;
                align-items: flex-start;
            }
        }

        .loading {
            display: inline-block;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .project-selector {
            padding: 10px 15px;
            background: #1a1a1a;
            border: 1px solid #2e2e2e;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .project-selector:hover {
            border-color: #667eea;
        }

        .project-selector:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .section-header {
            font-size: 1.5em;
            font-weight: 600;
            margin: 30px 0 20px 0;
            color: #e0e0e0;
            border-bottom: 2px solid #2e2e2e;
            padding-bottom: 10px;
        }

        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #888;
        }

        .empty-state-icon {
            font-size: 3em;
            margin-bottom: 10px;
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 Pipeline Dashboard</h1>
            <div class="header-info">
                <span id="timestamp" class="status-badge healthy">
                    ✓ Live • <span id="time"></span>
                </span>
                <span id="health-status" class="status-badge healthy">
                    ✓ System Healthy
                </span>
                <div class="control-group">
                    <select id="project-select" class="project-selector">
                        <option value="">All Projects</option>
                        <option value="realistic_female_influencer">Realistic Female Influencer</option>
                    </select>
                </div>
            </div>
        </header>

        <!-- Quick Stats -->
        <div class="grid">
            <div class="card">
                <div class="card-title">Queue Overview</div>
                <div class="stat-row">
                    <span class="stat-label">Total Items in Queue</span>
                    <span class="stat-value large" id="queue-total">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Sorted Items</span>
                    <span class="stat-value large" id="sorted-total">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Active Workers</span>
                    <span class="stat-value" id="active-workers">0</span>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Processing Rate</div>
                <div class="stat-row">
                    <span class="stat-label">Avg Time/Image</span>
                    <span class="stat-value" id="avg-time">-- ms</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Last Scan</span>
                    <span class="stat-value" id="last-scan">--</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Refresh Rate</span>
                    <span class="stat-value" id="refresh-rate">5s</span>
                </div>
            </div>

            <div class="card">
                <div class="card-title">System Status</div>
                <div class="stat-row">
                    <span class="stat-label">Uptime</span>
                    <span class="stat-value" id="uptime">--</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">API Status</span>
                    <span class="stat-value" id="api-status">✓ OK</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Dashboard Version</span>
                    <span class="stat-value" id="version">1.0.0</span>
                </div>
            </div>
        </div>

        <!-- Scraper Status Section -->
        <div class="section-header">Scraper Controls</div>
        <div class="grid">
            <div class="card grid-full">
                <div class="card-title">Scraper Status & Control</div>
                <div id="scraper-controls"></div>
            </div>
        </div>

        <!-- Queue Stats Section -->
        <div class="section-header">Queue Statistics by Scraper</div>
        <div class="grid">
            <div id="queue-stats-container"></div>
        </div>

        <!-- Sorted Stats Section -->
        <div class="section-header">Sorted Items by Category</div>
        <div class="grid">
            <div id="sorted-stats-container"></div>
        </div>

        <!-- Vision Queue Monitor -->
        <div class="section-header">Vision Queue Monitor (Recent Items)</div>
        <div class="card">
            <div class="card-title">5 Most Recent Items</div>
            <div id="vision-queue"></div>
        </div>

        <!-- Control Panel -->
        <div class="section-header">Control Panel</div>
        <div class="grid">
            <div class="card grid-full">
                <div class="card-title">Maintenance & Cleanup</div>
                <div class="control-group">
                    <button class="btn btn-primary" onclick="exportStats()">📥 Export Stats (JSON)</button>
                    <button class="btn btn-secondary" onclick="clearLogs()">🗑️ Clear Logs</button>
                    <button class="btn btn-danger" onclick="cleanupOrphaned()">🔧 Cleanup Orphaned Files</button>
                    <button class="btn btn-secondary" onclick="refreshData()">🔄 Force Refresh</button>
                </div>
                <div id="action-status" style="margin-top: 15px;"></div>
            </div>
        </div>

        <!-- Footer -->
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #2e2e2e; color: #888; font-size: 0.85em; text-align: center;">
            Pipeline Dashboard v1.0.0 • Last Updated: <span id="footer-time">--</span>
        </footer>
    </div>

    <script>
        const REFRESH_INTERVAL = 5000; // 5 seconds
        const API_BASE = '/api';
        let startTime = Date.now();

        // Format timestamp
        function formatTime(date) {
            return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }

        // Format uptime
        function formatUptime(ms) {
            const seconds = Math.floor(ms / 1000);
            const minutes = Math.floor(seconds / 60);
            const hours = Math.floor(minutes / 60);
            const days = Math.floor(hours / 24);

            if (days > 0) return `${days}d ${hours % 24}h`;
            if (hours > 0) return `${hours}h ${minutes % 60}m`;
            if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
            return `${seconds}s`;
        }

        // Update time display
        function updateTime() {
            const now = new Date();
            document.getElementById('time').textContent = formatTime(now);
            document.getElementById('footer-time').textContent = formatTime(now);
            document.getElementById('uptime').textContent = formatUptime(Date.now() - startTime);
        }

        // Fetch and display stats
        async function fetchStats() {
            try {
                const response = await fetch(`${API_BASE}/stats`);
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching stats:', error);
                document.getElementById('api-status').textContent = '✗ Error';
            }
        }

        // Update dashboard with stats
        function updateDashboard(data) {
            if (!data) return;

            // Update totals
            document.getElementById('queue-total').textContent = data.total_queue_items || 0;
            document.getElementById('sorted-total').textContent = data.total_sorted_items || 0;
            document.getElementById('active-workers').textContent = data.active_workers || 0;
            document.getElementById('api-status').textContent = '✓ OK';

            // Update queue stats
            updateQueueStats(data.queue_by_project);

            // Update sorted stats
            updateSortedStats(data.sorted_by_project);

            // Update vision queue
            updateVisionQueue(data.vision_recent);

            // Update timestamp
            document.getElementById('last-scan').textContent = new Date(data.timestamp).toLocaleTimeString();
        }

        // Update queue statistics display
        function updateQueueStats(queueByProject) {
            const container = document.getElementById('queue-stats-container');
            container.innerHTML = '';

            if (!queueByProject) return;

            Object.entries(queueByProject).forEach(([project, scrapers]) => {
                Object.entries(scrapers).forEach(([scraper, stats]) => {
                    const card = document.createElement('div');
                    card.className = 'card';
                    card.innerHTML = `
                        <div class="card-title">${scraper}</div>
                        <div class="card-subtitle">${project}</div>
                        <div class="stat-row">
                            <span class="stat-label">Items in Queue</span>
                            <span class="stat-value">${stats.count}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Status</span>
                            <span class="stat-value" style="color: ${stats.enabled ? '#4caf50' : '#f44336'};">
                                ${stats.enabled ? '✓ Active' : '✗ Disabled'}
                            </span>
                        </div>
                        ${stats.last_modified ? `
                            <div class="stat-row">
                                <span class="stat-label">Last Modified</span>
                                <span class="stat-value" style="font-size: 0.9em;">${new Date(stats.last_modified).toLocaleString()}</span>
                            </div>
                        ` : ''}
                    `;
                    container.appendChild(card);
                });
            });
        }

        // Update sorted statistics display
        function updateSortedStats(sortedByProject) {
            const container = document.getElementById('sorted-stats-container');
            container.innerHTML = '';

            if (!sortedByProject) return;

            Object.entries(sortedByProject).forEach(([project, categories]) => {
                Object.entries(categories).forEach(([category, count]) => {
                    const card = document.createElement('div');
                    card.className = 'card';
                    card.innerHTML = `
                        <div class="card-title">${category}</div>
                        <div class="card-subtitle">${project}</div>
                        <div class="stat-row">
                            <span class="stat-label">Items Sorted</span>
                            <span class="stat-value large">${count}</span>
                        </div>
                    `;
                    container.appendChild(card);
                });
            });

            if (Object.keys(sortedByProject).length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📭</div>No sorted items yet</div>';
            }
        }

        // Update vision queue display
        function updateVisionQueue(visionRecent) {
            const container = document.getElementById('vision-queue');
            container.innerHTML = '';

            if (!visionRecent || visionRecent.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">👁️</div>No recent items</div>';
                return;
            }

            visionRecent.forEach(item => {
                const div = document.createElement('div');
                div.className = 'queue-item';
                div.innerHTML = `
                    <div class="queue-item-name">${item.filename}</div>
                    <div class="queue-item-meta">
                        <span>Category: <strong>${item.category}</strong></span>
                        <span>${(item.size_kb).toFixed(1)} KB</span>
                    </div>
                    <div class="queue-item-meta">
                        <span>Modified: ${new Date(item.mtime * 1000).toLocaleString()}</span>
                    </div>
                `;
                container.appendChild(div);
            });
        }

        // Update scraper controls
        async function updateScraperControls() {
            try {
                const response = await fetch(`${API_BASE}/scrapers`);
                const data = await response.json();
                renderScraperControls(data.scrapers, data.status);
            } catch (error) {
                console.error('Error fetching scrapers:', error);
            }
        }

        // Render scraper control buttons
        function renderScraperControls(scrapers, status) {
            const container = document.getElementById('scraper-controls');
            container.innerHTML = '';

            scrapers.forEach(scraper => {
                const enabled = status[scraper].enabled;
                const div = document.createElement('div');
                div.className = 'scraper-toggle';
                div.innerHTML = `
                    <span>${scraper}</span>
                    <div class="toggle-switch">
                        <label class="switch">
                            <input type="checkbox" ${enabled ? 'checked' : ''} onchange="toggleScraper('${scraper}', this.checked)">
                            <span class="slider"></span>
                        </label>
                    </div>
                `;
                container.appendChild(div);
            });
        }

        // Toggle scraper
        async function toggleScraper(scraper, enabled) {
            try {
                const response = await fetch(`${API_BASE}/scrapers/${scraper}/toggle`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled })
                });
                const data = await response.json();
                showActionStatus(`✓ ${scraper} ${enabled ? 'enabled' : 'disabled'}`, 'success');
            } catch (error) {
                console.error('Error toggling scraper:', error);
                showActionStatus('✗ Failed to toggle scraper', 'error');
            }
        }

        // Export stats
        async function exportStats() {
            try {
                const response = await fetch(`${API_BASE}/export/stats`);
                const data = await response.json();
                const json = JSON.stringify(data, null, 2);
                const blob = new Blob([json], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `pipeline-stats-${new Date().toISOString()}.json`;
                a.click();
                showActionStatus('✓ Stats exported successfully', 'success');
            } catch (error) {
                console.error('Error exporting stats:', error);
                showActionStatus('✗ Failed to export stats', 'error');
            }
        }

        // Clear logs
        async function clearLogs() {
            if (!confirm('Are you sure you want to clear all logs?')) return;

            try {
                showActionStatus('Clearing logs...', 'info');
                const response = await fetch(`${API_BASE}/cleanup/logs`, { method: 'POST' });
                const data = await response.json();
                showActionStatus(`✓ Cleared ${data.cleared_logs.length} log files`, 'success');
            } catch (error) {
                console.error('Error clearing logs:', error);
                showActionStatus('✗ Failed to clear logs', 'error');
            }
        }

        // Cleanup orphaned files
        async function cleanupOrphaned() {
            if (!confirm('This may take a while. Continue?')) return;

            try {
                showActionStatus('Running cleanup...', 'info');
                const response = await fetch(`${API_BASE}/cleanup/orphaned`, { method: 'POST' });
                const data = await response.json();
                showActionStatus(`✓ Cleanup complete: ${data.removed_files} files removed, ${(data.freed_bytes / 1024 / 1024).toFixed(2)} MB freed`, 'success');
                setTimeout(fetchStats, 1000);
            } catch (error) {
                console.error('Error during cleanup:', error);
                showActionStatus('✗ Cleanup failed', 'error');
            }
        }

        // Force refresh
        function refreshData() {
            fetchStats();
            showActionStatus('✓ Data refreshed', 'success');
        }

        // Show action status
        function showActionStatus(message, type) {
            const statusEl = document.getElementById('action-status');
            statusEl.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
            if (type !== 'info') {
                setTimeout(() => { statusEl.innerHTML = ''; }, 5000);
            }
        }

        // Initialize
        function init() {
            updateTime();
            fetchStats();
            updateScraperControls();

            // Set up auto-refresh
            setInterval(updateTime, 1000);
            setInterval(fetchStats, REFRESH_INTERVAL);
            setInterval(updateScraperControls, REFRESH_INTERVAL * 2);
        }

        // Start on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
    </script>
</body>
</html>
"""

# ============================================================================
# STARTUP & MAIN
# ============================================================================

def main():
    """Main entry point"""
    # Verify paths exist
    for path in [PROMPT_LIBRARY_ROOT, QUEUE_ROOT, SORTED_ROOT, LOGS_ROOT]:
        if not os.path.isdir(path):
            logger.warning(f"Path not found: {path}")
    
    # Start background scanner thread
    scanner_thread = threading.Thread(target=background_scanner, daemon=True)
    scanner_thread.start()
    logger.info("Background scanner started")
    
    # Initial scan
    scan_pipeline_state()
    
    # Start Flask app
    logger.info("Starting Pipeline Dashboard on http://localhost:5000")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )

if __name__ == "__main__":
    main()
