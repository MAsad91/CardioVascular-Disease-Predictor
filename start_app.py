#!/usr/bin/env python3
"""
Simple startup script for Render deployment
"""
import os
import sys
import time

def main():
    print("🚀 Starting Heart Care application...")
    
    # Set environment variables for production
    os.environ['FLASK_ENV'] = 'production'
    
    # Import and start the app
    try:
        from app import app
        
        # Get port from environment variable (for Render)
        port = int(os.environ.get('PORT', 8080))
        
        print(f"🌐 Application will be available on port: {port}")
        print("📚 API Documentation: /help")
        print("🏥 Health Check: /health")
        
        # Start the application
        app.run(
            host='0.0.0.0',  # Bind to all interfaces
            port=port,        # Use the port from environment
            debug=False,      # Disable debug mode for production
            threaded=True     # Enable threading for better performance
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 