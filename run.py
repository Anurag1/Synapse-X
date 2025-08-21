from synapse_x_core.orchestrator import SynapseX

def main():
    print("\n" + "="*50)
    print("      Welcome to Project Synapse-X")
    print(" An AI that forges new specialist skills on demand.")
    print("="*50)
    print("COMMANDS:")
    print("  1. To teach a new skill, use the format:")
    print("     learn skill <name>: <description>; <input1> -> <output1>; <example2> ...")
    print("\n  2. To use a skill or the generalist AI, just type a prompt.")
    print("  3. Type 'exit' to quit.")
    print("-"*50)

    sapient_ai = SynapseX()

    while True:
        try:
            prompt = input("\n[You]> ")
            if prompt.lower() == 'exit':
                print("Shutting down Synapse-X.")
                break
            
            response = sapient_ai.process(prompt)
            print(f"[Synapse-X]> {response}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()