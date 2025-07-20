import warnings

# Suppress RuntimeWarning about unawaited coroutines from pdf2jsonl module
# This can occur when pytest imports modules containing async functions
warnings.filterwarnings("ignore", message="coroutine.*was never awaited", category=RuntimeWarning)