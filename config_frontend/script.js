document.addEventListener('DOMContentLoaded', () => {
    const binanceKeyInput = document.getElementById('binance_key');
    const binanceSecretInput = document.getElementById('binance_secret');
    const telegramTokenInput = document.getElementById('telegram_token');
    const telegramChatIdInput = document.getElementById('telegram_chat_id');
    const frequiUsernameInput = document.getElementById('frequi_username');
    const frequiPasswordInput = document.getElementById('frequi_password');

    const loadConfigButton = document.getElementById('load_config_button');
    const saveConfigButton = document.getElementById('save_config_button');
    const statusMessageDiv = document.getElementById('status_message');

    const API_URL = 'http://localhost:5001/api/config'; // Backend API URL

    // Function to display status messages
    function showStatus(message, isError = false) {
        statusMessageDiv.textContent = message;
        statusMessageDiv.className = isError ? 'status-error' : 'status-success';
        setTimeout(() => {
            statusMessageDiv.textContent = '';
            statusMessageDiv.className = '';
        }, 5000); // Clear message after 5 seconds
    }

    // Load configuration
    async function loadConfig() {
        statusMessageDiv.textContent = 'Loading...';
        statusMessageDiv.className = '';
        try {
            const response = await fetch(API_URL);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(`Failed to load configuration: ${errorData.detail || response.statusText}`);
            }
            const config = await response.json();

            binanceKeyInput.value = config.exchange_key || '';
            binanceSecretInput.value = config.exchange_secret || '';
            telegramTokenInput.value = config.telegram_token || '';
            telegramChatIdInput.value = config.telegram_chat_id || '';
            frequiUsernameInput.value = config.api_server_username || '';
            frequiPasswordInput.value = config.api_server_password || '';

            showStatus('Configuration loaded successfully.');
        } catch (error) {
            console.error('Error loading config:', error);
            showStatus(`Error loading configuration: ${error.message}`, true);
        }
    }

    // Save configuration
    async function saveConfig() {
        statusMessageDiv.textContent = 'Saving...';
        statusMessageDiv.className = '';

        const configData = {
            exchange_key: binanceKeyInput.value,
            exchange_secret: binanceSecretInput.value,
            telegram_token: telegramTokenInput.value,
            telegram_chat_id: telegramChatIdInput.value,
            api_server_username: frequiUsernameInput.value,
            api_server_password: frequiPasswordInput.value,
        };
        
        // Only include fields that have a value, to allow partial updates
        // However, the backend is designed to only update fields provided in the request,
        // so sending all fields (even if empty) is also fine.
        // For this implementation, we'll send all fields as collected.

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(configData),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(`Failed to save configuration: ${errorData.detail || response.statusText}`);
            }
            const result = await response.json();
            showStatus(result.message || 'Configuration saved successfully.');

        } catch (error) {
            console.error('Error saving config:', error);
            showStatus(`Error saving configuration: ${error.message}`, true);
        }
    }

    // Event Listeners
    if (loadConfigButton) {
        loadConfigButton.addEventListener('click', loadConfig);
    }
    if (saveConfigButton) {
        saveConfigButton.addEventListener('click', saveConfig);
    }

    // Optionally, load config on page load
    // loadConfig(); 
});
