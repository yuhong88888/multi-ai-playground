#!/usr/bin/env python3
"""
Demo script to showcase the multi-agent discussion system.
This runs a quick demonstration of the system in action.
"""

from discussion import DiscussionBuilder


def main():
    print("\n" + "="*70)
    print("   🚀 MULTI-AI PLAYGROUND DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows multiple AI agents discussing a topic together.")
    print("Each agent has a unique personality and perspective.\n")
    
    # Create a demonstration discussion
    discussion = (
        DiscussionBuilder("Artificial Intelligence and the Future of Work")
        .add_ai_agent("Optimistic Olivia", "optimist")
        .add_ai_agent("Skeptical Sam", "skeptic")
        .add_ai_agent("Analytical Alice", "analyst")
        .set_max_turns(2)
        .build()
    )
    
    # Run the discussion
    discussion.run()
    
    print("\n" + "="*70)
    print("   ✨ DEMO COMPLETE!")
    print("="*70)
    print("\n💡 To run your own discussions:")
    print("   python main.py 'Your topic here'")
    print("\n💡 To include human participation:")
    print("   python main.py 'Your topic here'  (human included by default)")
    print("\n💡 For more options:")
    print("   python main.py --help")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
