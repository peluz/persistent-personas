import time
import random

def call_gemini_with_retry(model, prompt, max_retries=7, base_delay=1):
    """
    Calls the Gemini API with retry logic, handling rate limits.

    Args:
        model: The Gemini model instance.
        prompt: The prompt to send to the model.
        max_retries: The maximum number of retry attempts.
        base_delay: The initial delay in seconds before the first retry.

    Returns:
        The response from the Gemini API, or None if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            response = model.send_message(prompt)
            return response, model
        except Exception as e:
            if "429" in str(e):  # Check for rate limit error
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                print(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            elif "503" in str(e):  # Check for model overload
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                print(f"Model overloaded error. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            elif "502" in str(e):  # bad gateway error
                delay = 30  # Fixed delay
                print(f"Bad gateway error. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            elif "Server disconnected" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                print(f"Server disconnected. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                print(f"An unexpected error occurred: {e}")
                time.sleep(delay)
    print(f"Max retries ({max_retries}) reached.  Could not get a response.")
    return None