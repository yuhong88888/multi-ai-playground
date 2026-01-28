#!/usr/bin/env python3
"""
Example: Large diverse discussion
This demonstrates a discussion with many agents with different personalities.
"""

from discussion import DiscussionBuilder


def main():
    # Create a rich discussion about innovation
    discussion = (
        DiscussionBuilder("Innovation in technology and its impact on society")
        .add_ai_agent("Alex the Optimist", "optimist")
        .add_ai_agent("Blake the Skeptic", "skeptic")
        .add_ai_agent("Casey the Analyst", "analyst")
        .add_ai_agent("Drew the Creative", "creative")
        .add_ai_agent("Elle the Pragmatist", "pragmatist")
        .add_human_agent("Moderator")  # Human can moderate
        .set_max_turns(3)
        .build()
    )
    
    print("\n🎯 This is a diverse discussion with 5 AI agents and 1 human!")
    print("   Each agent brings a unique perspective.\n")
    
    # Run the discussion
    discussion.run()


if __name__ == "__main__":
    main()
