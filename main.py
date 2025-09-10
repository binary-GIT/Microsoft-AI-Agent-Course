# main.py

from agent.agent import run_agent

def main():
    print("🤖 Dictionary Agent (Agentic AI)")
    print("Type 'exit' to quit.\n")

    # Multi-turn memory (simple list of interactions)
    conversation_history = []

    while True:
        user_input = input("Enter a word or question (or 'exit'): ").strip()
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        # Include previous conversation context for more agentic behavior
        context = ""
        if conversation_history:
            context = "Here is the previous conversation:\n"
            for i, (q, a) in enumerate(conversation_history[-5:], 1):  # last 5 interactions
                context += f"{i}. Q: {q}\n   A: {a}\n"
            context += "\n"

        query_with_context = context + user_input

        try:
            # Run the agent
            response = run_agent(query_with_context)
            print(f"\n🔎 Answer: {response}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")
            response = "No response."

        # Save interaction in conversation history
        conversation_history.append((user_input, response))

if __name__ == "__main__":
    main()
