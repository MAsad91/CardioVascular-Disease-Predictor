#!/usr/bin/env python3
"""
Database initialization script for Render deployment
This script ensures the database is properly set up on first deployment
"""
import os
import sys
from datetime import datetime, timezone

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db, User

def init_database():
    """Initialize the database with tables and sample data"""
    print("🔧 Initializing database for deployment...")
    
    with app.app_context():
        try:
            # Create all tables
            print("📋 Creating database tables...")
            db.create_all()
            print("✅ Database tables created successfully")
            
            # Check if admin user exists
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                print("👤 Creating admin user...")
                admin_user = User(
                    username='admin',
                    email='admin@heartcare.com',
                    first_name='Admin',
                    last_name='User',
                    role='admin',
                    is_active=True,
                    email_verified=True,
                    created_at=datetime.now(timezone.utc)
                )
                admin_user.set_password('admin123')
                db.session.add(admin_user)
                db.session.commit()
                print("✅ Admin user created (username: admin, password: admin123)")
            else:
                print("✅ Admin user already exists")
            
            # Check if any users exist
            user_count = User.query.count()
            print(f"📊 Total users in database: {user_count}")
            
            print("🎉 Database initialization completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing database: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    success = init_database()
    if success:
        print("\n✅ Database is ready for deployment!")
        sys.exit(0)
    else:
        print("\n❌ Database initialization failed!")
        sys.exit(1) 