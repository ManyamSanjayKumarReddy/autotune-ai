"""
CLI entry point.
Note: conversations_text is a pre-formatted string — simulating what the backend sends.
Usage: python -m agent_v1.graph.app
"""
from agent_v1.graph.graph import run_pipeline

# Simulate what the backend would send — already formatted plain text
SAMPLE_CONVERSATIONS_TEXT = """--- Conversation 1 ---
USER: What are the eligibility criteria for PM Kisan Samman Nidhi Yojana?
ASSISTANT: I'm not sure about the exact eligibility, you may want to check the official website.
USER: That's not helpful. I need specific details.

--- Conversation 2 ---
USER: How do I apply for Ayushman Bharat health insurance?
ASSISTANT: Ayushman Bharat is a health scheme. Please visit a government portal.
USER: Which portal? Your answer is too vague.

--- Conversation 3 ---
USER: What documents are needed for PM Awas Yojana?
ASSISTANT: You need identity documents.
USER: Which ones exactly? This doesn't help me at all."""


if __name__ == "__main__":
    print("Starting AutoTune AI Pipeline...\n")
    final = run_pipeline(
        conversations_text=SAMPLE_CONVERSATIONS_TEXT,
        batch_id="test-batch-001"
    )
    print("\nPipeline complete.")