import firebase_admin
from firebase_admin import credentials
from app.config import settings

_firebase_app = None


def get_firebase_app() -> firebase_admin.App:
    global _firebase_app
    if _firebase_app is None:
        import os
        key_path = settings.FIREBASE_SERVICE_ACCOUNT_KEY
        if not os.path.exists(key_path):
            raise RuntimeError(
                f"\n\n[Firebase] Service account key not found at: '{key_path}'\n"
                "  1. Go to Firebase Console -> Project Settings -> Service Accounts\n"
                "  2. Click 'Generate new private key' and download the JSON file\n"
                "  3. Save it to the path set in FIREBASE_SERVICE_ACCOUNT_KEY in .env\n"
            )
        cred = credentials.Certificate(key_path)
        _firebase_app = firebase_admin.initialize_app(cred)
    return _firebase_app
