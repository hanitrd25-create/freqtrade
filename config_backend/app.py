import os
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Path to the config.json file within the container
# The volume mount in docker-compose will map ../user_data (host) to /app/user_data (container)
CONFIG_FILE_PATH = '/app/user_data/config.json'
# For local development without Docker, you might use:
# CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), '../user_data/config.json')

def load_config():
    """Loads the Freqtrade config from config.json."""
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return {} # Return empty dict for malformed JSON to allow partial reads/writes if applicable

def save_config(config_data):
    """Saves the Freqtrade config to config.json."""
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
        with open(CONFIG_FILE_PATH, 'w') as f:
            json.dump(config_data, f, indent=4)
        return True
    except Exception as e:
        app.logger.error(f"Error saving config: {e}")
        return False

@app.route('/api/config', methods=['GET'])
def get_config_values():
    config_data = load_config()
    if config_data is None:
        return jsonify({"error": "Config file not found"}), 404
    if not config_data: # Handles JSONDecodeError case from load_config
        return jsonify({"error": "Config file is malformed"}), 500

    values_to_return = {
        "exchange_key": config_data.get("exchange", {}).get("key"),
        "exchange_secret": config_data.get("exchange", {}).get("secret"),
        "telegram_token": config_data.get("telegram", {}).get("token"),
        "telegram_chat_id": config_data.get("telegram", {}).get("chat_id"),
        "api_server_username": config_data.get("api_server", {}).get("username"),
        "api_server_password": config_data.get("api_server", {}).get("password")
    }
    return jsonify(values_to_return), 200

@app.route('/api/config', methods=['POST'])
def update_config_values():
    current_config = load_config()
    if current_config is None:
        # If config doesn't exist, maybe we should not create it from scratch here
        # Or, define a basic structure if that's desired. For now, error out.
        return jsonify({"error": "Config file not found. Cannot update."}), 404
    # if not current_config: # Allow updating a malformed/empty file if it exists
        # current_config = {}


    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400
    except Exception as e:
        return jsonify({"error": "Failed to decode JSON payload", "details": str(e)}), 400

    updated = False

    # Update exchange settings
    if "exchange_key" in request_data:
        current_config.setdefault("exchange", {})["key"] = request_data["exchange_key"]
        updated = True
    if "exchange_secret" in request_data:
        current_config.setdefault("exchange", {})["secret"] = request_data["exchange_secret"]
        updated = True

    # Update telegram settings
    if "telegram_token" in request_data:
        current_config.setdefault("telegram", {})["token"] = request_data["telegram_token"]
        updated = True
    if "telegram_chat_id" in request_data:
        current_config.setdefault("telegram", {})["chat_id"] = request_data["telegram_chat_id"]
        updated = True

    # Update api_server settings
    if "api_server_username" in request_data:
        current_config.setdefault("api_server", {})["username"] = request_data["api_server_username"]
        updated = True
    if "api_server_password" in request_data:
        current_config.setdefault("api_server", {})["password"] = request_data["api_server_password"]
        updated = True

    if updated:
        if save_config(current_config):
            return jsonify({"message": "Configuration updated successfully"}), 200
        else:
            return jsonify({"error": "Failed to save configuration"}), 500
    else:
        return jsonify({"message": "No values provided for update"}), 200


if __name__ == '__main__':
    # Make sure to set host='0.0.0.0' to be accessible from outside the container
    app.run(host='0.0.0.0', port=5001, debug=True) # debug=True for dev, consider False for prod
