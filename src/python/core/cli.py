"""
Command Line Interface for FLIM-FRET Analysis Pipeline

Handles argument parsing, user interaction, and menu systems.
"""

import argparse
import sys
import os
from typing import Dict, Any, Optional
from pathlib import Path

# ANSI color codes for rainbow effect
class Colors:
    red = '\033[31m'
    orange = '\033[38;5;208m'
    yellow = '\033[33m'
    green = '\033[32m'
    blue = '\033[34m'
    indigo = '\033[38;5;54m'
    violet = '\033[35m'
    reset = '\033[0m'
    bold = '\033[1m'

def colorize(text: str, color: str) -> str:
    """Apply color to text"""
    return f"{Colors.bold}{color}{text}{Colors.reset}"

def show_header() -> None:
    """Display colorful FLIM ASCII art header"""
    header = [
        "  ",
        "                                                                          ",
        "      ███████╗██╗     ██╗███╗   ███╗    ███████╗██████╗ ███████╗████████╗ ",
        "      ██╔════╝██║     ██║████╗ ████║    ██╔════╝██╔══██╗██╔════╝╚══██╔══╝ ",
        "      █████╗  ██║     ██║██╔████╔██║    █████╗  ██████╔╝█████╗     ██║    ",
        "      ██╔══╝  ██║     ██║██║╚██╔╝██║    ██╔══╝  ██╔══██╗██╔══╝     ██║    ",
        "      ██║     ███████╗██║██║ ╚═╝ ██║    ██║     ██║  ██║███████╗   ██║    ",
        "      ╚═╝     ╚══════╝╚═╝╚═╝     ╚═╝    ╚═╝     ╚═╝  ╚═╝╚══════╝   ╚═╝    ",
        "                                                                          ",
        "  ",
        ""
    ]

    rainbow_colors = [
        Colors.red,
        Colors.orange, 
        Colors.yellow,
        Colors.green,
        Colors.blue,
        Colors.indigo,
        Colors.violet
    ]

    print('')  # Empty line at start
    
    for i, line in enumerate(header):
        if i < 2 or i > 7:  # Empty lines
            print(line)
        else:
            # Split the line to color FLIM and FRET separately
            # The line contains both FLIM and FRET ASCII art
            # Split at the middle to separate FLIM (left) and FRET (right)
            line_length = len(line)
            mid_point = line_length // 2
            
            # Color the left half (FLIM) in green and right half (FRET) in red
            flim_part = line[:mid_point]
            fret_part = line[mid_point:]
            
            colored_line = colorize(flim_part, Colors.green) + colorize(fret_part, Colors.red)
            print(colored_line)


class CLIError(Exception):
    """Custom exception for CLI-related errors."""
    pass


class PipelineCLI:
    """
    Command line interface for the FLIM-FRET analysis pipeline.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="FLIM-FRET Analysis Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run complete processing pipeline
  python run_pipeline.py --input /path/to/data --output /path/to/output --processing
  
  # Run only preprocessing
  python run_pipeline.py --input /path/to/data --output /path/to/output --preprocessing
  
  # Run only wavelet filtering (requires preprocessed data)
  python run_pipeline.py --input /path/to/data --output /path/to/output --filter
  
  # Interactive mode (shows menu)
  python run_pipeline.py --input /path/to/data --output /path/to/output
            """
        )

        # Directory arguments (optional, will prompt if not provided)
        parser.add_argument(
            "--input", 
            help="Input directory containing raw FLIM-FRET .bin files (will prompt if not provided)"
        )
        parser.add_argument(
            "--output", 
            help="Base output directory for all pipeline stages (will prompt if not provided)"
        )
        

        
        # Workflow groupings
        parser.add_argument(
            "--preprocessing", 
            action="store_true", 
            help="Run Stages 1-2A: convert files, phasor transformation, and organize files"
        )
        parser.add_argument(
            "--processing", 
            action="store_true", 
            help="Run Stages 1-2B: preprocessing and processing (TIFF to NPZ: wavelet filtering, lifetime calculation)"
        )
        
        # Individual stages
        parser.add_argument(
            "--set-directories", 
            action="store_true", 
            help="Set default input and output directories"
        )
        parser.add_argument(
            "--visualize", 
            action="store_true", 
            help="Run Stage 3: Interactive phasor visualization and plot generation"
        )
        parser.add_argument(
            "--segment", 
            action="store_true", 
            help="Run Stage 4: Interactive phasor segmentation (GMM or manual)"
        )
        parser.add_argument(
            "--lifetime-images", 
            action="store_true", 
            help="Run lifetime image generation from NPZ files"
        )
        parser.add_argument(
            "--average-lifetime", 
            action="store_true", 
            help="Calculate average lifetime from segmented data"
        )

        
        # Interactive mode
        parser.add_argument(
            "--interactive", 
            action="store_true", 
            help="Run in interactive mode for user input (affects GMM segmentation)"
        )
        
        # Configuration
        parser.add_argument(
            "--config", 
            help="Path to configuration file (default: config/config.json)"
        )
        
        # Logging
        parser.add_argument(
            "--log-level", 
            choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
            default="INFO",
            help="Set logging level (default: INFO)"
        )
        
        return parser
    
    def parse_args(self, args: Optional[list] = None) -> argparse.Namespace:
        """
        Parse command line arguments.
        
        Args:
            args: List of arguments to parse (if None, uses sys.argv)
            
        Returns:
            Parsed arguments namespace
        """
        parsed_args = self.parser.parse_args(args)
        self._validate_args(parsed_args)
        return parsed_args
    
    def _validate_args(self, args: argparse.Namespace) -> None:
        """
        Validate parsed arguments.
        
        Args:
            args: Parsed arguments
            
        Raises:
            CLIError: If arguments are invalid
        """
        # Validate input directory if provided
        if args.input and not os.path.isdir(args.input):
            raise CLIError(f"Input directory '{args.input}' does not exist or is not a directory")
        
    def show_interactive_menu(self, args: argparse.Namespace) -> argparse.Namespace:
        """
        Show interactive menu if no specific stages are selected.
        
        Args:
            args: Parsed arguments
            
        Returns:
            Updated arguments with user selection
        """
        # Check if any stage is already selected
        stage_flags = [
            args.set_directories, args.preprocessing, args.processing, 
            args.visualize, args.segment, args.lifetime_images, 
            args.average_lifetime
        ]
        
        if any(stage_flags):
            return args  # Stage already selected, no need for menu
        
        # Show menu
        show_header()
        print(colorize("  🔬 Welcome FLIM-FRET analysis user! 🔬", Colors.bold))
        print("")
        print(colorize("MENU:", Colors.bold))
        print(colorize("1. Set Input/Output Directories", Colors.green))
        print(colorize("2. Preprocessing (.bin to .tif)", Colors.yellow))
        print(colorize("3. Preprocessing + Processing (.bin to .npz)", Colors.yellow))
        print(colorize("4. Visualization (interactive phasor plots)", Colors.yellow))
        print(colorize("5. Segmentation (interactive phasor segmentation - GMM or manual)", Colors.yellow))
        print(colorize("6. Average Lifetime (calculate average lifetime from segmented data)", Colors.yellow))
        print(colorize("7. Lifetime Images (generate lifetime images from NPZ files)", Colors.yellow))
        print(colorize("8. Exit", Colors.red))
        
        # Get user choice
        choice = input("Select an option (1-8): ").strip().lower()
        
        # Update args based on choice
        if choice == "1":
            args.set_directories = True
        elif choice == "2":
            args.preprocessing = True
        elif choice == "3":
            args.processing = True
        elif choice == "4":
            args.visualize = True
        elif choice == "5":
            args.segment = True
        elif choice == "6":
            args.average_lifetime = True
        elif choice == "7":
            args.lifetime_images = True
        elif choice == "8" or choice == "q" or choice == "quit":
            print("Exiting.")
            return None  # Signal to exit
        else:
            print("Invalid choice. Please enter a number between 1-8 or 'q' to quit.")
            return args  # Return current args to continue loop
        
        return args
    
    def get_directory_choice(self, prompt: str, directories: Dict[str, str], 
                           counts: Optional[Dict[str, int]] = None) -> str:
        """
        Get user choice between directories.
        
        Args:
            prompt: Prompt message to show
            directories: Dictionary of choice_key -> directory_path
            counts: Optional dictionary of choice_key -> file_count
            
        Returns:
            Selected directory path
        """
        print(f"\n{prompt}")
        
        choices = list(directories.keys())
        for i, (key, path) in enumerate(directories.items(), 1):
            count_info = ""
            if counts and key in counts:
                count_info = f" ({counts[key]} files found)"
            print(f"  [{i}] {key}{count_info}")
        
        while True:
            try:
                choice_num = input(f"Select option (1-{len(choices)}, default: 1): ").strip()
                if choice_num == "" or choice_num == "1":
                    selected_key = choices[0]
                    break
                elif choice_num.isdigit() and 1 <= int(choice_num) <= len(choices):
                    selected_key = choices[int(choice_num) - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(choices)}.")
            except (ValueError, IndexError):
                print(f"Please enter a number between 1 and {len(choices)}.")
        
        selected_path = directories[selected_key]
        print(f"→ Using {selected_key}: {selected_path}")
        return selected_path
    
    def get_yes_no_choice(self, prompt: str, default: bool = True) -> bool:
        """
        Get yes/no choice from user.
        
        Args:
            prompt: Prompt message
            default: Default choice if user just presses enter
            
        Returns:
            User's choice as boolean
        """
        default_str = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        
        if response == "":
            return default
        elif response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'.")
            return self.get_yes_no_choice(prompt, default)
    
    def get_choice(self, prompt: str, choices: list, default: int = 1) -> int:
        """
        Get numbered choice from user.
        
        Args:
            prompt: Prompt message
            choices: List of choice descriptions
            default: Default choice number (1-indexed)
            
        Returns:
            Selected choice number (1-indexed)
        """
        print(f"\n{prompt}")
        for i, choice in enumerate(choices, 1):
            marker = " (default)" if i == default else ""
            print(f"  [{i}] {choice}{marker}")
        
        while True:
            try:
                response = input(f"Select option (1-{len(choices)}, default: {default}): ").strip()
                if response == "":
                    return default
                elif response.isdigit() and 1 <= int(response) <= len(choices):
                    return int(response)
                else:
                    print(f"Please enter a number between 1 and {len(choices)}.")
            except ValueError:
                print(f"Please enter a number between 1 and {len(choices)}.")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments using the CLI class.
    
    Returns:
        Parsed arguments
    """
    cli = PipelineCLI()
    args = cli.parse_args()
    args = cli.show_interactive_menu(args)
    return args


def create_cli() -> PipelineCLI:
    """
    Create and return a CLI instance.
    
    Returns:
        Configured CLI instance
    """
    return PipelineCLI() 