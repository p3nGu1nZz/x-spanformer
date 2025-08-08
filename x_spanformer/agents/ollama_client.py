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
	temperature: float = 0.2,
	timeout: float = 60.0
) -> str:
	client = AsyncClient(timeout=timeout)
	
	# Build messages list with system prompt if provided
	messages: List[Message] = []
	if system:
		messages.append({"role": "system", "content": system})
	messages.extend(conversation)

	# Log query size instead of full content
	user_msg = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
	query_size = len(user_msg)
	logger.info(f"Sending to {model} (T={temperature}, timeout={timeout}s, query_size={query_size} chars)")

	try:
		response = await client.chat(
			model=model,
			messages=messages,
			options={
				"temperature": temperature,
				"num_predict": 2048,  # Increased token limit for better span coverage
				"repeat_penalty": 1.1  # Slight penalty for repetition
			},
			stream=False
		)
		content = response["message"]["content"]
		
		# Log response size instead of full content
		response_size = len(content)
		logger.info(f"Response from {model} (response_size={response_size} chars)")
		
		return content
		
	except Exception as e:
		logger.error(f"Error in chat request: {e}")
		raise