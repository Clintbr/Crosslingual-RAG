"""
Send a prompt to Ollama and return a consistent result.

Return
    {
        "success": True,
        "response": "...",
        "error": None
    }

    or

    {
        "success": False,
        "response": None,
        "error": {
            "phase": "...",
            "type": "...",
            "message": "...",
            "status_code": ...,
        }
    }
"""

import requests

from src.config import OLLAMA_URL


def send_to_ollama(model, prompt, phase):

    if not model:
        return {
            "success": False,
            "response": None,
            "error": {
                "phase": phase,
                "type": "InvalidModel",
                "message": "No Ollama model was provided.",
                "status_code": None
            }
        }

    if not prompt:
        return {
            "success": False,
            "response": None,
            "error": {
                "phase": phase,
                "type": "InvalidPrompt",
                "message": "The prompt is empty.",
                "status_code": None
            }
        }

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        # Handle HTTP errors like 500
        if not response.ok:
            try:
                error_data = response.json()
                ollama_message = error_data.get("error", response.text)
            except ValueError:
                ollama_message = response.text

            return {
                "success": False,
                "response": None,
                "error": {
                    "phase": phase,
                    "type": "OllamaHTTPError",
                    "message": ollama_message,
                    "status_code": response.status_code
                }
            }

        # Parse JSON
        try:
            data = response.json()
        except ValueError:
            return {
                "success": False,
                "response": None,
                "error": {
                    "phase": phase,
                    "type": "InvalidJSON",
                    "message": "Ollama returned an invalid JSON response.",
                    "status_code": response.status_code
                }
            }

        # Ollama can return an error inside the JSON
        if "error" in data:
            return {
                "success": False,
                "response": None,
                "error": {
                    "phase": phase,
                    "type": "OllamaError",
                    "message": data["error"],
                    "status_code": response.status_code
                }
            }

        # Make sure the expected response exists
        if "response" not in data or not data["response"] or data["response"] is None or data["response"] is ("" or " "):
            return {
                "success": False,
                "response": None,
                "error": {
                    "phase": phase,
                    "type": "MissingResponse",
                    "message": "Ollama response does not contain a 'response' field.",
                    "status_code": response.status_code
                }
            }

        # Everything worked
        return {
            "success": True,
            "response": data["response"],
            "error": None
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "response": None,
            "error": {
                "phase": phase,
                "type": "Timeout",
                "message": "The request to Ollama timed out.",
                "status_code": None
            }
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "response": None,
            "error": {
                "phase": phase,
                "type": "ConnectionError",
                "message": "Could not connect to Ollama. Is Ollama running?",
                "status_code": None
            }
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "response": None,
            "error": {
                "phase": phase,
                "type": "RequestError",
                "message": str(e),
                "status_code": None
            }
        }

    except Exception as e:
        # Last-resort protection so this function never crashes the application
        return {
            "success": False,
            "response": None,
            "error": {
                "phase": phase,
                "type": "UnexpectedError",
                "message": str(e),
                "status_code": None
            }
        }
