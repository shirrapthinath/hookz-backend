import secrets

# Generate a 32-byte hex token
secret_key = secrets.token_hex(32)
print(f"Generated SECRET_KEY: {secret_key}")
print("Copy this key and set it in your Flask app configuration.")
