import ollama
import json

class LLMEngine:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        print(f"LLM Meta-Mind initialized with model: '{self.model_name}'.")

    def get_response(self, system_prompt, user_prompt):
        """Generic function to get a response from the LLM."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error communicating with local LLM: {e}. Is Ollama running?"

    def analyze_and_route(self, prompt, known_skills):
        """Uses the LLM to analyze a prompt and decide which skill to use."""
        if not known_skills:
            return "generalist"
            
        system_prompt = f"""
You are a skill router for an AI system. Analyze the user's prompt and decide which of the available skills is the best fit.
The available skills are: {', '.join(known_skills)}.
Respond with a single JSON object containing one key, "skill", with the value being the name of the chosen skill, or "generalist" if none are a good fit.
"""
        try:
            response = self.get_response(system_prompt, prompt)
            skill = json.loads(response).get("skill", "generalist")
            return skill.lower().strip()
        except (json.JSONDecodeError, AttributeError):
            # If the LLM fails to produce valid JSON, fallback to a simpler check
            for skill in known_skills:
                if skill in prompt.lower(): return skill
            return "generalist"

    def generate_synthetic_data(self, skill_description, examples):
        """Uses the LLM to generate a synthetic dataset for training a new specialist."""
        print(f"Meta-Mind: Generating synthetic data for skill '{skill_description}'...")
        system_prompt = f"""
You are a data generation engine for an AI. Your task is to create a high-quality dataset for training a new specialist model.
The specialist needs to learn the following skill: "{skill_description}".
The user has provided these examples:
{examples}

Based on this, generate 50 diverse JSON objects, each with an "input" and "output" key, that would be perfect for training this specialist.
Your response MUST be a single JSON list containing these 50 objects. Ensure the inputs and outputs are varied and cover edge cases.
"""
        try:
            response = self.get_response(system_prompt, "Please generate the dataset now.")
            # Clean the response to ensure it's valid JSON
            clean_response = response.strip().replace("```json", "").replace("```", "")
            dataset = json.loads(clean_response)
            print(f"  -> Successfully generated a dataset of {len(dataset)} examples.")
            return dataset
        except Exception as e:
            print(f"LLM data generation error: {e}")
            return None