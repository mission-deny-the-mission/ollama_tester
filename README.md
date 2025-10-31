# Ollama Tester

A Python test suite for parallel testing of Ollama instances and OpenAI-compatible APIs. Measures performance metrics including time to first token (TTFT) and tokens per second.

## Features

- **Parallel Testing**: Test multiple servers simultaneously
- **Ollama Support**: Native support for Ollama's `/api/generate` endpoint
- **OpenAI Compatible**: Supports any OpenAI-compatible API endpoint
- **Streaming**: Supports both streaming and non-streaming responses
- **Metrics**: Measures:
  - Time to First Token (TTFT)
  - Tokens per Second
  - Total tokens
  - Total request time

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `config.json` file in the project directory. Example:

```json
{
  "test_prompt": "Write a short story about a robot learning to paint.",
  "timeout": 300,
  "ollama_servers": [
    {
      "name": "Local Ollama",
      "base_url": "http://localhost:11434",
      "model": "llama2",
      "stream": true
    },
    {
      "name": "Remote Ollama",
      "base_url": "http://192.168.1.100:11434",
      "model": "mistral",
      "stream": true
    }
  ],
  "openai_servers": [
    {
      "name": "OpenAI API",
      "base_url": "https://api.openai.com",
      "model": "gpt-3.5-turbo",
      "api_key": "sk-...",
      "stream": true
    },
    {
      "name": "Local OpenAI Server",
      "base_url": "http://localhost:8000",
      "model": "custom-model",
      "api_key": null,
      "stream": true
    }
  ]
}
```

### Configuration Fields

- `test_prompt`: The prompt to send to all servers
- `timeout`: Request timeout in seconds (default: 300)
- `ollama_servers`: Array of Ollama server configurations
  - `name`: Display name for the server
  - `base_url`: Base URL of the Ollama instance (e.g., `http://localhost:11434`)
  - `model`: Model name to use
  - `stream`: Whether to use streaming (default: true)
- `openai_servers`: Array of OpenAI-compatible server configurations
  - `name`: Display name for the server
  - `base_url`: Base URL of the API (e.g., `https://api.openai.com` or `http://localhost:8000/v1`)
  - `model`: Model name to use
  - `api_key`: API key (optional, set to `null` if not needed)
  - `stream`: Whether to use streaming (default: true)

## Usage

### Basic Usage

```bash
python test_suite.py
```

This will use `config.json` from the current directory.

### Custom Configuration File

```bash
python test_suite.py --config my_config.json
```

### Custom Test Prompt

```bash
python test_suite.py --prompt "Explain quantum computing in simple terms"
```

### Save Results to JSON

```bash
python test_suite.py --output results.json
```

### Combined Options

```bash
python test_suite.py --config config.json --prompt "Your prompt here" --output results.json
```

## Output

The test suite prints a formatted table showing:
- Server name
- Test status (PASS/FAIL)
- Time to First Token (TTFT) in seconds
- Tokens per second
- Total tokens
- Total request time

Example output:
```
========================================================================================================================
TEST RESULTS
========================================================================================================================
Server                        Status     TTFT (s)     Tokens/s     Total Tokens    Total Time (s) 
------------------------------------------------------------------------------------------------------------------------
Local Ollama                  ✓ PASS     0.234        45.67        120             2.627          
Remote Ollama                 ✓ PASS     0.456        38.92        95              2.441          
OpenAI API                    ✓ PASS     0.123        125.50       250             1.992          
========================================================================================================================

SUMMARY (Successful tests only):
  Total servers tested: 3
  Successful: 3
  Failed: 0
  Average TTFT: 0.271s
  Average Tokens/s: 70.03
  Best Tokens/s: 125.50
```

## JSON Output Format

When using `--output`, results are saved in JSON format:

```json
[
  {
    "server_name": "Local Ollama",
    "endpoint": "http://localhost:11434",
    "success": true,
    "time_to_first_token": 0.234,
    "tokens_per_second": 45.67,
    "total_time": 2.627,
    "total_tokens": 120,
    "completion_tokens": null,
    "prompt_tokens": null,
    "error_message": null,
    "timestamp": "2024-01-15T10:30:45.123456"
  }
]
```

## Notes

- All tests run in parallel using asyncio for maximum efficiency
- Streaming mode is recommended for accurate token-per-second measurements
- Time to first token is only measured in streaming mode
- The test suite handles timeouts and connection errors gracefully

## Troubleshooting

- **Connection errors**: Check that the base URLs are correct and servers are running
- **Timeout errors**: Increase the `timeout` value in `config.json`
- **Empty responses**: Ensure the model names are correct and models are available on the servers
- **Authentication errors**: Verify API keys are correct for OpenAI-compatible endpoints

