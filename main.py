#!/usr/bin/env python3
"""
Multi-AI Playground - Main CLI Application
A system where multiple AI agents can discuss topics with each other
and welcome human participants to join the conversation.
"""

import sys
import argparse
from discussion import DiscussionBuilder


def create_default_discussion(topic: str, include_human: bool = True, turns: int = 5):
    """Create a discussion with default configuration"""
    builder = DiscussionBuilder(topic)
    
    # Add diverse AI agents with different personalities
    builder.add_ai_agent("Alex", "optimist")
    builder.add_ai_agent("Blake", "skeptic")
    builder.add_ai_agent("Casey", "analyst")
    builder.add_ai_agent("Drew", "creative")
    
    # Add human participant if requested
    if include_human:
        builder.add_human_agent("Human Participant")
    
    builder.set_max_turns(turns)
    
    return builder.build()


def create_custom_discussion(topic: str):
    """Create a custom discussion with user input"""
    print("\n🎯 CUSTOM DISCUSSION SETUP")
    print("=" * 60)
    
    builder = DiscussionBuilder(topic)
    
    # Ask about AI agents
    print("\n🤖 Available AI Personalities:")
    print("1. optimist - Positive and enthusiastic")
    print("2. skeptic - Critical and questioning")
    print("3. analyst - Data-driven and logical")
    print("4. creative - Innovative and imaginative")
    print("5. pragmatist - Practical and action-oriented")
    
    while True:
        try:
            num_ai_input = input("\nHow many AI agents? (1-10): ").strip()
            num_ai = int(num_ai_input) if num_ai_input else 3
            if 1 <= num_ai <= 10:
                break
            else:
                print("⚠️  Please enter a number between 1 and 10.")
        except ValueError:
            print("⚠️  Invalid input. Please enter a number between 1 and 10.")
    
    personalities = ["optimist", "skeptic", "analyst", "creative", "pragmatist"]
    agent_names = ["Alex", "Blake", "Casey", "Drew", "Elle", "Finn", "Grace", "Henry", "Iris", "Jack"]
    
    for i in range(num_ai):
        personality = personalities[i % len(personalities)]
        name = agent_names[i % len(agent_names)]
        builder.add_ai_agent(name, personality)
        print(f"  ✓ Added {name} ({personality})")
    
    # Ask about human participation
    include_human = input("\nInclude human participation? (y/n): ").strip().lower() == 'y'
    if include_human:
        human_name = input("Your name (default: Human): ").strip() or "Human"
        builder.add_human_agent(human_name)
        print(f"  ✓ Added {human_name} (human)")
    
    # Set turns
    while True:
        try:
            turns_input = input("\nNumber of discussion turns (default: 5): ").strip()
            turns = int(turns_input) if turns_input else 5
            if turns >= 1:
                builder.set_max_turns(turns)
                break
            else:
                print("⚠️  Please enter a positive number.")
        except ValueError:
            print("⚠️  Invalid input. Please enter a positive number.")
    
    print("\n✅ Discussion configured successfully!")
    return builder.build()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-AI Playground - AI agents discussing topics with human participation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a discussion with default settings
  python main.py "The future of artificial intelligence"
  
  # AI-only discussion (no human)
  python main.py "Climate change solutions" --no-human
  
  # Custom number of turns
  python main.py "Space exploration" --turns 8
  
  # Custom setup with interactive configuration
  python main.py "Education technology" --custom
        """
    )
    
    parser.add_argument(
        "topic",
        help="The topic for the multi-agent discussion"
    )
    
    parser.add_argument(
        "--no-human",
        action="store_true",
        help="Run discussion with AI agents only (no human participation)"
    )
    
    parser.add_argument(
        "--turns",
        type=int,
        default=5,
        help="Number of discussion turns (default: 5)"
    )
    
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom setup wizard to configure the discussion"
    )
    
    args = parser.parse_args()
    
    # Display welcome banner
    print("\n" + "=" * 60)
    print("🎭  MULTI-AI PLAYGROUND")
    print("=" * 60)
    print("Multiple AI agents discussing topics together")
    print("with human participation welcome!")
    print("=" * 60 + "\n")
    
    try:
        # Create and run discussion
        if args.custom:
            discussion = create_custom_discussion(args.topic)
        else:
            discussion = create_default_discussion(
                args.topic,
                include_human=not args.no_human,
                turns=args.turns
            )
        
        discussion.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
