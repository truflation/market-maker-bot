# TrufNetwork Avellaneda Market Making Bot

A market making bot for TrufNetwork prediction markets that implements the Avellaneda-Stoikov (A-S) strategy, adapted for binary options pricing.

## Features

- **Avellaneda-Stoikov Strategy**: Implements the classic A-S market making model with adaptations for binary options
- **Automatic Volatility Estimation**: Uses RMS of consecutive mid-price differences for real-time volatility
- **Black-Scholes Initial Pricing**: Prices new markets with no order book using Black-Scholes binary option pricing
- **Per-Market Inventory Management**: Tracks YES/NO share positions independently per market
- **Dynamic Spread Adjustment**: Adjusts spreads based on volatility, inventory, and order book depth
- **Atomic Order Updates**: Uses `change_bid()`/`change_ask()` for efficient order modifications
- **Graceful Shutdown**: Cancels all orders on shutdown

## Installation

```bash
cd market-maker-bot
pip install -e ".[dev]"
```

## Quick Start

1. Copy the example configuration:
   ```bash
   cp config.example.yaml config.yaml
   ```

2. Edit `config.yaml` with your market details and credentials

3. Run in dry-run mode to test:
   ```bash
   market-maker-bot --config config.yaml --dry-run
   ```

4. Run for real:
   ```bash
   export TN_PRIVATE_KEY="your_private_key_here"
   market-maker-bot --config config.yaml
   ```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TN_NODE_URL` | TrufNetwork node URL | `http://localhost:8484` |
| `TN_PRIVATE_KEY` | Private key for signing transactions | (required) |

## Configuration

See `config.example.yaml` for all available options. Key parameters:

### Avellaneda Parameters

- `risk_factor`: Risk aversion factor (γ) - higher = wider spreads
- `order_amount_shape_factor`: Eta (η) for asymmetric order sizing (0-1)
- `min_spread`: Minimum spread as % of mid price (default: 0)
- `max_spread`: Maximum spread in cents (default: 20.0)
- `inventory_target_base_pct`: Target percentage of value in shares (default: 50%)
- `order_optimization_enabled`: Jump to best bid+1 / best ask-1 (default: true)
- `order_levels`: Number of orders on each side (default: 1)
- `filled_order_delay`: Seconds to wait after fill (default: 60)

### Market Configuration

```yaml
markets:
  - query_id: 1
    stream_id: "stream_id_for_black_scholes"
    data_provider: "0x..."
    name: "Market Name"
    outcome_mode: "yes"  # "yes", "no", or "both"
    order_amount: 100
```

## How It Works

### Pricing Model

The bot uses the Avellaneda-Stoikov formulas adapted for binary options:

```
Reservation Price: r = mid_price - q × γ × σ × T
Optimal Spread:    δ = γ × σ × T + (2/γ) × ln(1 + γ/κ)
Optimal Bid:       r - δ/2
Optimal Ask:       r + δ/2
```

Where:
- `q` = inventory deviation from target (-1 to +1)
- `γ` = risk aversion factor
- `σ` = volatility in cents (absolute)
- `κ` = order book depth factor
- `T` = time horizon (1.0 for infinite)

### Volatility Sources (Priority Order)

1. **Order book mid-price history**: RMS of consecutive price differences
2. **Black-Scholes from underlying stream**: For new markets with no trading data
3. **Configurable default**: Fallback floor value

### Initial Pricing (New Markets)

When a market has no order book data, the bot:
1. Fetches historical records from the underlying primitive stream
2. Calculates stream volatility (Yang-Zhang for hourly, Close-to-Close for daily)
3. Prices the binary option using Black-Scholes
4. Maps fair value (0-1 probability) to price in cents (1-99)

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_avellaneda.py -v

# Run with coverage
pytest tests/ --cov=market_maker_bot
```

## Architecture

```
src/market_maker_bot/
├── main.py              # Entry point
├── config.py            # Configuration models
├── bot.py               # Main bot orchestrator
├── models.py            # Data models
├── market.py            # Market state management
├── pricing/
│   ├── avellaneda.py    # A-S pricing model
│   ├── black_scholes.py # Binary option pricing
│   └── inventory.py     # Per-market inventory
├── indicators/
│   ├── volatility.py    # Order book volatility (RMS)
│   ├── stream_volatility.py  # Yang-Zhang/Close-to-Close
│   └── depth.py         # Order book depth for kappa
└── utils/
    └── ring_buffer.py   # Efficient circular buffer
```

## Advanced Configuration

### Using a Different Node (Testnet)

To connect to a different node (e.g., testnet instead of mainnet), set the `TN_NODE_URL` environment variable:

```bash
# Use testnet
export TN_NODE_URL="https://gateway.testnet.truf.network"

# Or set in .env file
echo 'TN_NODE_URL=https://gateway.testnet.truf.network' >> .env
```

You can also set the node URL directly in your `config.yaml`:

```yaml
node_url: "https://gateway.testnet.truf.network"
```

### Pre-Approved Streams

The bot includes pre-approved streams for market making in `config.py`:

| Key | Stream ID | Name |
|-----|-----------|------|
| `us_inflation_yoy` | `st1e321de22ece39a258bc2588dd2871` | US Inflation YoY |
| `us_cpi_index` | `st8f1e62d3a130572ec468dda082f889` | US CPI Index |
| `us_cpi_index_alt` | `st1d6d41423cd9746a81ea6063b1345e` | US CPI Index Alt |
| `eu_inflation_yoy` | `ste03c2844c591a10d8a524d14d23066` | EU Inflation YoY |
| `eu_cpi_index` | `ste909219dce3f693c61a0f187758fb0` | EU CPI Index |
| `egg_price` | `stf6584cf470744723c90130130cb7db` | Egg Price |

Default data provider: `0x4710a8d8f0d845da110086812a32de6d90d7ff5c` (Truflation)

### Adding Custom Streams

To add custom streams or modify existing ones, edit the `APPROVED_STREAMS` dictionary in `src/market_maker_bot/config.py`:

```python
# In src/market_maker_bot/config.py

APPROVED_STREAMS: Dict[str, ApprovedStream] = {
    # Existing streams...
    "us_inflation_yoy": ApprovedStream(
        stream_id="st1e321de22ece39a258bc2588dd2871",
        name="US Inflation YoY",
        description="US Year-over-Year Inflation Rate",
    ),
    # Add your custom stream:
    "my_custom_stream": ApprovedStream(
        stream_id="your_custom_stream_id_here",
        name="My Custom Stream",
        description="Description of your stream",
        data_provider="0x...",  # Optional: defaults to Truflation provider
    ),
}
```

Then reference it in your `config.yaml`:

```yaml
markets:
  - query_id: 1
    stream_id: "your_custom_stream_id_here"
    data_provider: "0x..."
    name: "My Custom Stream"
    outcome_mode: "yes"
    order_amount: 100
    enabled: true
```

**MarketConfig parameters:**
- `query_id`: Unique market identifier
- `stream_id`: The TRUF Network stream identifier for Black-Scholes pricing
- `data_provider`: Data provider address for the stream
- `name`: Human-readable name for logging
- `outcome_mode`: `"yes"`, `"no"`, or `"both"` - which outcomes to market make
- `order_amount`: Order size in shares (default: 100)
- `enabled`: Set to `false` to disable a market (default: `true`)
- `gamma`: Optional market-specific risk factor override
- `min_spread`: Optional market-specific minimum spread override

### Order State Persistence

The bot persists its order state to a JSON file (`bot_order_state.json` by default) to support:

- **Restart Recovery**: Bot's own orders are recovered after restart
- **Manual Order Protection**: Orders placed manually outside the bot are ignored

Configure with:
```yaml
order_state_file: "bot_order_state.json"
cancel_open_orders_on_exit: true  # Set to false to leave orders open on shutdown
```

## References

- [Avellaneda & Stoikov (2008)](https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf) - "High-frequency trading in a limit order book"
- [Yang & Zhang (2000)](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00280) - "Drift Independent Volatility Estimation Based on High, Low, Open, and Close Prices"
