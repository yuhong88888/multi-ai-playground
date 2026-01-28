#!/usr/bin/env python3
"""
Example: Discussion with human participation
This demonstrates how humans can join AI agents in a discussion.
"""

from discussion import DiscussionBuilder


def main():
    # Create a discussion about remote work with human participation
    discussion = (
        DiscussionBuilder("The future of remote work and collaboration")
        .add_ai_agent("Practical Pete", "pragmatist")
        .add_ai_agent("Creative Chris", "creative")
        .add_human_agent("You")  # Human participant
        .set_max_turns(4)
        .build()
    )
    
    print("\n💡 TIP: When it's your turn, share your thoughts!")
    print("       Type 'skip' to pass your turn, or 'quit' to exit.\n")
    
    # Run the discussion
    discussion.run()


if __name__ == "__main__":
    main()
