#!/usr/bin/env python3
"""
HoloRay Motion-Tracked Annotation System
Main entry point for running the web application
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.app import run_server

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HoloRay Motion-Tracked Annotation System')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug)
