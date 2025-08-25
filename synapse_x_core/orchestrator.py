import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import json
from .octaves import SpecialistOctave
from .llm_engine import LLMEngine

# --- Simple character-level tokenizer for demo purposes ---
MAX_LEN = 128
def text_to_tensor(text):
    tensor = torch.zeros(MAX_LEN)
    for i, char in enumerate(text[:MAX_LEN]):
        tensor[i] = ord(char) / 255.0
    return tensor.unsqueeze(0)

def tensor_to_text(tensor):
    chars = [chr(int(c * 255)) for c in tensor.squeeze(0) if c > 0]
    return "".join(chars).strip()

class SynapseX:
    def __init__(self, skill_library_path="./trained_skills"):
        self.skill_library_path = skill_library_path
        self.skills = {}
        self.meta_mind = LLMEngine()
        os.makedirs(skill_library_path, exist_ok=True)
        self.load_skills()

    def load_skills(self):
        print("Synapse Core: Loading specialist skills from library...")
        for filename in os.listdir(self.skill_library_path):
            if filename.endswith(".meta"):
                skill_name = filename.replace(".meta", "")
                meta_path = os.path.join(self.skill_library_path, filename)
                model_path = os.path.join(self.skill_library_path, f"{skill_name}.pth")
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    model = SpecialistOctave(meta['input_dim'], meta['output_dim'], meta['z_dim'])
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    self.skills[skill_name] = {'model': model, 'meta': meta}
                    print(f"  -> Loaded skill: '{skill_name}'")
                except Exception as e:
                    print(f"  Warning: Could not load skill '{skill_name}'. Error: {e}")

    def process(self, prompt):
        """Main entry point for any user request."""
        if prompt.lower().startswith("learn skill"):
            return self.learn_new_skill_flow(prompt)

        known_skill_names = list(self.skills.keys())
        skill_to_use = self.meta_mind.analyze_and_route(prompt, known_skill_names)
        
        if skill_to_use == "generalist" or skill_to_use not in self.skills:
            print("  -> No specialist found. Routing to generalist Meta-Mind.")
            return self.meta_mind.get_response("You are a helpful general-purpose AI assistant.", prompt)
        
        print(f"  -> Meta-Mind routed to '{skill_to_use}' specialist.")
        specialist = self.skills[skill_to_use]['model']
        
        # Execute the specialist
        input_tensor = text_to_tensor(prompt)
        output_tensor = specialist.execute(input_tensor)
        return tensor_to_text(output_tensor)

    def learn_new_skill_flow(self, learn_prompt):
        """Handles the entire process of forging a new skill."""
        try:
            parts = learn_prompt.split(';')
            skill_def = parts[0].replace("learn skill", "").strip()
            skill_name, skill_description = skill_def.split(':', 1)
            skill_name = skill_name.strip().lower().replace(" ", "_")
            examples = "; ".join(parts[1:]).strip()
        except Exception:
            return "Error: Invalid 'learn' command format. Use: 'learn skill <name>: <description>; <input1> -> <output1>'"

        synthetic_data = self.meta_mind.generate_synthetic_data(skill_description, examples)
        if not synthetic_data or len(synthetic_data) < 10:
            return "Meta-Mind failed to generate sufficient training data. Please provide clearer examples."

        try:
            input_data = torch.stack([text_to_tensor(item['input']).squeeze(0) for item in synthetic_data])
            output_data = torch.stack([text_to_tensor(item['output']).squeeze(0) for item in synthetic_data])
            
            new_octave = self.train_new_specialist(skill_name, input_data, output_data)
            self.skills[skill_name] = {'model': new_octave, 'meta': {'input_dim': MAX_LEN, 'output_dim': MAX_LEN, 'z_dim': 32}}
            return f"Success! A new specialist skill named '{skill_name}' has been trained, frozen, and added to the library."
        except Exception as e:
            return f"An error occurred during training: {e}"

    def train_new_specialist(self, skill_name, input_data, output_data):
        """The core training and freezing process."""
        print(f"--- Forging new specialist: '{skill_name}' ---")
        model = SpecialistOctave(input_dim=MAX_LEN, output_dim=MAX_LEN, z_dim=32)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        dataset = torch.utils.data.TensorDataset(input_data, output_data)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        for epoch in range(30): # Short but effective training for demo
            for input_batch, output_batch in data_loader:
                optimizer.zero_grad()
                predicted_output, _, _ = model(input_batch)
                loss = F.mse_loss(predicted_output, output_batch)
                loss.backward()
                optimizer.step()
        
        model_path = os.path.join(self.skill_library_path, f"{skill_name}.pth")
        meta_path = os.path.join(self.skill_library_path, f"{skill_name}.meta")
        torch.save(model.state_dict(), model_path)
        with open(meta_path, 'w') as f:
            json.dump({'input_dim': MAX_LEN, 'output_dim': MAX_LEN, 'z_dim': 32}, f)
        print(f"  -> Specialist '{skill_name}' has been forged and saved.")
        model.eval()
        return model