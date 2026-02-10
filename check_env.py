try:
    import pyspark
    print("PySpark imported successfully")
except ImportError as e:
    print(f"PySpark import failed: {e}")
except Exception as e:
    print(f"PySpark error: {e}")

try:
    import flask
    print("Flask imported successfully")
except ImportError as e:
    print(f"Flask import failed: {e}")

try:
    from flask_login import LoginManager
    print("Flask-Login imported successfully")
except ImportError as e:
    print(f"Flask-Login import failed: {e}")
except Exception as e:
    print(f"Flask-Login error: {e}")
