#!/usr/bin/env python3
"""
Example: Simple discussion with AI agents only
This demonstrates a basic discussion without human participation.
"""

from discussion import DiscussionBuilder


def main():
    # Create a discussion about AI ethics
    discussion = (
        DiscussionBuilder("Ethical implications of AI in society")
        .add_ai_agent("Optimist Oliver", "optimist")
        .add_ai_agent("Skeptical Sarah", "skeptic")
        .add_ai_agent("Analytical Amy", "analyst")
        .set_max_turns(3)
        .build()
    )
    
    # Run the discussion
    discussion.run()


if __name__ == "__main__":
    main()
