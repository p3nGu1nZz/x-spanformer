import asyncio
import subprocess
import re
import logging
from typing import Optional, List

from ollama import AsyncClient

from x_spanformer.agents.constants import DEFAULT_SYSTEM

logger = logging.getLogger(__name__)
Message = dict[str, str]

async def check_ollama_running() -> bool:
	"""
	Check if Ollama service is running by testing 'ollama ps' command.
	Returns True if Ollama is running, False otherwise.
	"""
	try:
		result = subprocess.run(
			["ollama", "ps"],
			capture_output=True,
			text=True,
			encoding='utf-8',
			errors='replace',
			timeout=5
		)
		return result.returncode == 0
	except subprocess.TimeoutExpired:
		logger.error("ollama ps command timed out")
		return False
	except FileNotFoundError:
		logger.error("ollama command not found - is Ollama installed?")
		return False
	except Exception as e:
		logger.error(f"Error checking ollama status: {str(e)}")
		return False

def check_model_loaded(model: str, ps_output: str) -> bool:
	"""
	Check if a specific model is loaded by parsing ollama ps output.
	Returns True if model is found in the output, False otherwise.
	"""
	# Pattern matches: "phi4-mini" or "phi4-mini:latest" or "phi4-mini:anything"
	model_pattern = re.compile(rf"^{re.escape(model)}(:|$|\s)", re.MULTILINE)
	return bool(model_pattern.search(ps_output))

async def check_ollama_connection(model: str) -> bool:
	"""
	Test if Ollama is running and the specified model is loaded.
	Returns True if model is loaded and accessible, False otherwise.
	"""
	try:
		# First check if Ollama is running
		if not await check_ollama_running():
			logger.error("Ollama service is not running")
			return False
		
		# Get current ollama ps output
		result = subprocess.run(
			["ollama", "ps"],
			capture_output=True,
			text=True,
			encoding='utf-8',
			errors='replace',
			timeout=5
		)
		
		if result.returncode != 0:
			logger.error(f"ollama ps failed: {result.stderr}")
			return False
		
		# Check if model is loaded
		if check_model_loaded(model, result.stdout):
			logger.info(f"[SUCCESS] Model {model} is loaded and ready")
			return True
		else:
			logger.error(f"Model {model} is not loaded. Please run: ollama run {model}")
			return False
			
	except Exception as e:
		logger.error(f"Error checking ollama connection: {str(e)}")
		return False

async def chat(
	model: str,
	conversation: List[Message],
	system: Optional[str] = None,
	temperature: float = 0.2
) -> str:
	client = AsyncClient()
	
	# Build messages list with system prompt if provided
	messages: List[Message] = []
	if system:
		messages.append({"role": "system", "content": system})
	messages.extend(conversation)

	# Single line truncated logging for dialogue
	user_msg = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
	user_preview = (user_msg[:80] + '...') if len(user_msg) > 80 else user_msg
	logger.info(f"Sending to {model} (T={temperature}): {user_preview}")

	response = await client.chat(
		model=model,
		messages=messages,
		options={"temperature": temperature},
		stream=False
	)
	content = response["message"]["content"]
	
	# Single line truncated response logging
	response_preview = (content[:80] + '...') if len(content) > 80 else content
	logger.info(f"Response from {model}: {response_preview}")
	
	return content