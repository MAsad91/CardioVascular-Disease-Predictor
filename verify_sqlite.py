#!/usr/bin/env python3
"""
Verify SQLite configuration is working correctly
"""
import os
import sys
from datetime import datetime, timezone

def verify_sqlite_config():
    """Verify SQLite configuration is working correctly"""
    print("🔍 Verifying SQLite configuration...")
    
    try:
        from app import app, db, User
        
        print(f"✅ Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
        
        with app.app_context():
            # Test database connection
            print("📊 Testing database connection...")
            user_count = User.query.count()
            print(f"✅ Total users in database: {user_count}")
            
            # Test user creation
            print("👤 Testing user creation...")
            test_user = User(
                username='test_verify',
                email='test@verify.com',
                created_at=datetime.now(timezone.utc),
                is_active=True,
                email_verified=False
            )
            test_user.set_password('testpass123')
            db.session.add(test_user)
            db.session.commit()
            print("✅ User creation successful")
            
            # Verify user was created
            created_user = User.query.filter_by(username='test_verify').first()
            if created_user:
                print(f"✅ User verified: {created_user.username} ({created_user.email})")
                
                # Test password verification
                if created_user.check_password('testpass123'):
                    print("✅ Password verification successful")
                else:
                    print("❌ Password verification failed")
                
                # Clean up
                db.session.delete(created_user)
                db.session.commit()
                print("✅ Test user cleaned up")
            else:
                print("❌ User not found after creation")
                return False
            
            print("🎉 SQLite configuration verification completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = verify_sqlite_config()
    if success:
        print("\n✅ SQLite is configured correctly and working!")
        sys.exit(0)
    else:
        print("\n❌ SQLite configuration has issues!")
        sys.exit(1) 