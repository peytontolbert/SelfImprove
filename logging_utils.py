import logging

async def log_with_ollama(ollama, message, context=None):
    """Log messages with Ollama's assistance."""
    logger = logging.getLogger("OllamaLogger")
    prompt = f"Log this message: {message}"
    if context:
        prompt += f" | Context: {context}"
    response = await ollama.query_ollama("logging", prompt, refine=False)
    logger.info(f"Logged with Ollama: {response}")
