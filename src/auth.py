"""
User Authentication Module for Streamlit Cloud Deployment
Supports individual user logins with admin role management.
"""

import streamlit as st
import hashlib
from typing import Optional, Dict

def hash_password(password: str) -> str:
    """Hash a password for secure storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def get_users() -> Dict[str, Dict]:
    """
    Get users from Streamlit secrets or return demo users for local dev.
    
    In Streamlit Cloud, configure secrets like:
    [users]
    [users.admin]
    password_hash = "hashed_password_here"
    role = "admin"
    name = "Administrator"
    
    [users.john]
    password_hash = "hashed_password_here"
    role = "user"
    name = "John Smith"
    """
    try:
        if hasattr(st, 'secrets') and 'users' in st.secrets:
            return dict(st.secrets['users'])
    except:
        pass
    
    # Default users for local development (password: "admin" and "user123")
    return {
        'admin': {
            'password_hash': hash_password('admin'),
            'role': 'admin',
            'name': 'Administrator'
        },
        'demo': {
            'password_hash': hash_password('demo123'),
            'role': 'user',
            'name': 'Demo User'
        }
    }

def authenticate(username: str, password: str) -> Optional[Dict]:
    """
    Authenticate a user and return their info if successful.
    Returns None if authentication fails.
    """
    users = get_users()
    
    if username in users:
        user = users[username]
        password_hash = hash_password(password)
        
        # Support both hashed and plain passwords (for easier setup)
        stored_hash = user.get('password_hash', '')
        if password_hash == stored_hash or password == stored_hash:
            return {
                'username': username,
                'role': user.get('role', 'user'),
                'name': user.get('name', username)
            }
    
    return None

def check_authentication() -> bool:
    """
    Check if user is authenticated. Shows login form if not.
    Returns True if authenticated, False otherwise.
    
    Usage in app.py:
        from src.auth import check_authentication, get_current_user
        
        if not check_authentication():
            st.stop()
        
        user = get_current_user()
        st.write(f"Welcome, {user['name']}!")
    """
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    
    # If already authenticated, return True
    if st.session_state['authenticated']:
        return True
    
    # Show login form
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## 🔐 Login Required")
        st.markdown("---")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if username and password:
                    user = authenticate(username, password)
                    if user:
                        st.session_state['authenticated'] = True
                        st.session_state['user'] = user
                        st.success(f"✅ Welcome, {user['name']}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter username and password")
        
        st.markdown("---")
        st.caption("Contact administrator for access")
    
    return False

def get_current_user() -> Optional[Dict]:
    """Get the currently logged in user info."""
    return st.session_state.get('user', None)

def is_admin() -> bool:
    """Check if current user is an admin."""
    user = get_current_user()
    return user is not None and user.get('role') == 'admin'

def logout():
    """Log out the current user."""
    st.session_state['authenticated'] = False
    st.session_state['user'] = None

def show_user_menu():
    """Show user info and logout button in sidebar."""
    user = get_current_user()
    if user:
        with st.sidebar:
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                role_emoji = "👑" if user['role'] == 'admin' else "👤"
                st.caption(f"{role_emoji} {user['name']}")
            with col2:
                if st.button("🚪", help="Logout"):
                    logout()
                    st.rerun()

def require_admin():
    """
    Decorator-style check for admin-only features.
    Call this at the start of admin-only sections.
    """
    if not is_admin():
        st.error("⛔ Admin access required")
        st.stop()
