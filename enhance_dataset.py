import os
import json
import time
import requests

class OllamaEnhancer:
    def __init__(self, logs_directory, new_logs_directory):
        self.logs_directory = logs_directory
        self.new_logs_directory = new_logs_directory
        os.makedirs(self.new_logs_directory, exist_ok=True)

    async def chat_with_ollama(self, system_prompt: str, interaction: dict, retries: int = 5, delay: int = 5):
        url = "http://localhost:11434/api/generate"
        refined_system_prompt = (
            "As an AGI expert specializing in complex software systems, "
            "your task is to critically evaluate and refine the provided response provided after the system prompt and context. "
            "Ensure that the refinement enhances clarity, aligns with the organizational goals, "
            "and optimizes resource utilization. Address any nuanced user behaviors, "
            "and focus on improving the system's complex task management capabilities."
        )
        # Prepare the prompt to include the entire interaction for refinement
        combined_prompt = f"{refined_system_prompt}\n\nInteraction:\n\n{json.dumps(interaction, indent=4)}"
        
        payload = {
            "model": "llama3.1",
            "prompt": combined_prompt,
            "format": "json",
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        print(combined_prompt)
        for i in range(retries):
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()  # Ensure a 4XX/5XX error raises an exception
                response_data = response.json()  # Parse the JSON response
                print(f"response data: {response_data['response']}")
                if 'response' in response_data:
                    return response_data['response']  # Return the 'response' field
                else:
                    raise KeyError("'response' key not found in the API response")
            except (requests.exceptions.RequestException, KeyError) as e:
                if i < retries - 1:  # i is zero indexed
                    time.sleep(delay)  # wait before trying again
                else:
                    raise e  # re-raise the last exception if all retries fail

    async def enhance_logs(self):
        for filename in os.listdir(self.logs_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(self.logs_directory, filename)
                with open(file_path, 'r') as f:
                    interaction = json.load(f)

                # Enhance the interaction using Ollama
                enhanced_response = await self.chat_with_ollama(interaction['system_prompt'], interaction)

                # Add the enhanced response back into the interaction
                interaction['enhanced_response'] = enhanced_response

                # Save the enhanced interaction to the new directory
                new_file_path = os.path.join(self.new_logs_directory, filename)
                with open(new_file_path, 'w') as f:
                    json.dump(interaction, f, indent=4)

                print(f"Enhanced and saved to new_logs: {filename}")

if __name__ == "__main__":
    logs_directory = 'logs'  # Directory containing the original JSON log files
    new_logs_directory = 'new_logs'  # Directory to save the enhanced JSON log files
    enhancer = OllamaEnhancer(logs_directory, new_logs_directory)

    # Run the enhancement process
    import asyncio
    asyncio.run(enhancer.enhance_logs())
