"""
Entry point for the Avellaneda Market Making Bot.

Usage:
    python -m market_maker_bot.main --config config.yaml
    python -m market_maker_bot.main --dry-run --sample-data
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

from .config import BotConfig, MarketConfig, AvellanedaConfig, load_config_from_dict
from .models import OutcomeMode
from .bot import AvellanedaMarketMaker


def setup_logging(debug: bool = False) -> None:
    """Configure logging for the bot."""
    level = logging.DEBUG if debug else logging.INFO
    format_str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def load_config(config_path: str) -> BotConfig:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return load_config_from_dict(data)


def create_sample_config() -> BotConfig:
    """Create a sample configuration for testing."""
    return BotConfig(
        node_url=os.environ.get("TN_NODE_URL", "http://localhost:8484"),
        private_key=os.environ.get("TN_PRIVATE_KEY", ""),
        markets=[
            MarketConfig(
                query_id=1,
                stream_id="example_stream",
                data_provider="0x1234567890abcdef1234567890abcdef12345678",
                name="Example Market",
                outcome_mode=OutcomeMode.YES_ONLY,
                order_amount=100,
            ),
        ],
        avellaneda=AvellanedaConfig(
            risk_factor=1.0,
            inventory_target_base_pct=50.0,
            volatility_buffer_size=60,
            default_volatility=5.0,
            min_spread=0.0,  # Percentage of mid price
            order_refresh_time=10.0,
        ),
        order_book_poll_interval=2.0,
        inventory_refresh_interval=30.0,
        dry_run=True,
        debug=True,
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Avellaneda Market Making Bot for TrufNetwork",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config.yaml
  %(prog)s --config config.yaml --dry-run
  %(prog)s --sample-data --dry-run

Environment variables:
  TN_NODE_URL      TrufNetwork node URL (default: http://localhost:8484)
  TN_PRIVATE_KEY   Private key for signing transactions
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without executing orders",
    )

    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Use sample configuration for testing",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.config and not args.sample_data:
        parser.error("Either --config or --sample-data is required")

    # Load configuration
    if args.sample_data:
        config = create_sample_config()
    else:
        try:
            config = load_config(args.config)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except yaml.YAMLError as e:
            print(f"Error parsing config: {e}", file=sys.stderr)
            return 1

    # Apply command-line overrides
    if args.dry_run:
        config.dry_run = True
    if args.debug:
        config.debug = True

    # Setup logging
    setup_logging(debug=config.debug)

    logger = logging.getLogger(__name__)
    logger.info("Avellaneda Market Making Bot starting...")

    # Validate private key
    if not config.private_key and not config.dry_run:
        logger.error(
            "No private key configured. Set TN_PRIVATE_KEY environment variable "
            "or use --dry-run mode."
        )
        return 1

    # Create and run bot
    bot = AvellanedaMarketMaker(config)

    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
