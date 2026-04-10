# Truflation Market Maker Bot

An open-source market making bot for [Truflation](https://truflation.com) prediction markets on the [TRUF.NETWORK](https://truf.network). Implements the Avellaneda-Stoikov strategy adapted for binary options, with Black-Scholes pricing for new markets.

Run it to provide two-sided liquidity on prediction markets and earn a share of the 2% settlement fees.

## Background

### What is TRUF.NETWORK?

[TRUF.NETWORK](https://truf.network) is a decentralized oracle network built by [Truflation](https://truflation.com) for publishing and consuming real-world economic data (CPI, inflation, commodity prices, etc.). It powers prediction markets that let users trade on the future value of these data streams.

- **Website**: https://truf.network
- **Documentation**: https://docs.truf.network
- **Token & Governance**: https://docs.truf.network/token-governance/tokenomics
- **Data explorer**: https://trufscan.io
- **GitHub**: https://github.com/trufnetwork

### What are TT and TT2?

**TT** and **TT2** are the testnet tokens used as collateral for prediction market orders on TRUF.NETWORK testnet. They have no monetary value and are only used for testing the protocol before mainnet. On mainnet, the corresponding collateral token is $TRUF.

- **TT2** is the newer testnet collateral token (used by the current bot deployment)
- **TT** is the legacy testnet token
- To run this bot on mainnet, you'll need $TRUF tokens instead: https://docs.truf.network/token-governance/get-truf-token

### Prediction Markets on TRUF.NETWORK

Each market is a binary prediction on a data stream outcome (e.g., "Will US CPI YoY be between 1.3% and 1.5% on April 10?"). Markets have two sides: YES and NO. Share prices range from 1c to 99c, where `YES price + NO price = 100c`. Liquidity providers earn a portion of the 2% settlement fee based on how long their orders stayed within the rewards-eligible spread.

## What It Does

- Continuously quotes bids and asks on both YES and NO outcomes for configured markets
- Prices new markets using Black-Scholes when order book data is unavailable
- Dynamically adjusts spreads based on volatility, inventory position, and order book depth
- Places multiple order levels (L0-L4) per side for deeper liquidity
- Pulls all liquidity 15 minutes before settlement to protect capital from oracle risk
- Recovers order state across restarts so it doesn't leave orphan orders behind

## How Market Makers Earn Rewards

TRUF.NETWORK charges a 2% fee on market settlements. A portion of that fee is distributed to liquidity providers based on:

- **Eligibility**: Orders must be paired (buy on one outcome + sell on the opposite at complementary prices summing to 100c)
- **Spread tightness**: Tighter spreads (closer to mid) earn higher scores via dynamic spread tiers
- **Duration**: Rewards are proportional to "liquidity-hours" - how long your eligible orders stay on the book
- **Size**: Larger orders above the minimum size earn proportionally more

Per-block snapshots track LP positions while the market is live. Rewards are calculated and distributed atomically at settlement.

## Features

- **Avellaneda-Stoikov strategy** adapted for 1c-99c binary option pricing
- **Dual-sided quoting**: places orders on both YES and NO sides of the order book
- **Black-Scholes initial pricing** for markets with no existing liquidity, using historical stream volatility
- **Multi-level orders**: configurable L0-L4 order levels per side for depth
- **Per-market inventory management**: tracks YES/NO share positions independently
- **Pre-settlement liquidity pull**: configurable cutoff (default 15 min) to exit before settlement
- **Stale order detection**: handles "order not found" errors gracefully
- **Dry-run mode**: simulate without placing real orders
- **Order state persistence**: recovers from restarts without losing track of placed orders
- **Graceful shutdown**: cancels all orders on exit (configurable)

## Requirements

- Python 3.12+
- A TRUF.NETWORK private key (generate one with `--generate-key`)
- USDC (mainnet) or TT2 (testnet) tokens to collateralize orders

## Installation

```bash
git clone https://github.com/truflation/market-maker-bot.git
cd market-maker-bot
pip install -e ".[dev]"
```

## Quick Start

1. Copy the example configuration:
   ```bash
   cp config.example.yaml config.yaml
   ```

2. Edit `config.yaml` with your target markets.

3. Set your private key:
   ```bash
   export TN_PRIVATE_KEY="your_private_key_here"
   ```

4. Test in dry-run mode first:
   ```bash
   market-maker-bot --config config.yaml --dry-run
   ```

5. Run live:
   ```bash
   market-maker-bot --config config.yaml
   ```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TN_NODE_URL` | TRUF.NETWORK gateway URL | `https://gateway.mainnet.truf.network` |
| `TN_PRIVATE_KEY` | Private key (64 hex chars, no `0x` prefix) | (required) |
| `MM_HEARTBEAT_FILE` | Optional heartbeat file path for external monitoring | (none) |

## Configuration

See `config.example.yaml` for all available options. Key parameters:

### Avellaneda Strategy Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `risk_factor` (gamma) | Risk aversion; higher = wider spreads | 0.1 |
| `min_spread` | Minimum spread as % of mid price | 0 |
| `max_spread` | Maximum spread in cents | 20.0 |
| `inventory_target_base_pct` | Target % of value in shares | 50 |
| `order_optimization_enabled` | Jump to best bid+1 / best ask-1 | true |
| `order_levels` | Number of orders on each side (L0-L4) | 1 |
| `level_distances` | Distance between levels as % of optimal spread | 25.0 |
| `filled_order_delay` | Seconds to wait after a fill | 60 |
| `pre_settlement_cutoff` | Seconds before settle to pull liquidity | 900 (15 min) |

### Market Configuration

```yaml
markets:
  - query_id: 1
    stream_id: "st1e321de22ece39a258bc2588dd2871"
    data_provider: "0x4710a8d8f0d845da110086812a32de6d90d7ff5c"
    name: "US Inflation YoY"
    outcome_mode: "both"  # "yes", "no", or "both"
    order_amount: 100
    enabled: true
```

**Market parameters:**
- `query_id`: Unique market identifier (from the protocol)
- `stream_id`: Underlying data stream for Black-Scholes pricing
- `data_provider`: Data provider address for the stream
- `name`: Human-readable name for logging
- `outcome_mode`: `"yes"`, `"no"`, or `"both"` - which outcomes to market make
- `order_amount`: Order size in shares
- `enabled`: Set to `false` to pause a market

## How It Works

### Pricing Model

The bot uses Avellaneda-Stoikov adapted for binary options (1c-99c range):

```
Reservation Price: r = mid_price - q * gamma * sigma * T
Optimal Spread:    delta = gamma * sigma * T + (2/gamma) * ln(1 + gamma/kappa)
Optimal Bid:       r - delta/2
Optimal Ask:       r + delta/2
```

Where:
- `q` = inventory deviation from target (-1 to +1)
- `gamma` = risk aversion factor
- `sigma` = volatility in cents
- `kappa` = order book depth factor
- `T` = time to expiry in years

### Order Placement Flow

1. **Bid orders** use `place_buy_order(outcome, price, amount)` directly
2. **Ask orders** work in two steps:
   - `place_split_limit_order(true_price)` mints YES+NO share pairs and lists NO at `100 - true_price`
   - `place_sell_order(outcome=True, price)` then sells the retained YES shares as a YES ask
   - This ensures asks land on **both sides** of the order book

### Volatility Estimation (Priority Order)

1. **Order book mid-price history**: RMS of consecutive price differences
2. **Black-Scholes from underlying stream**: For new markets with no trading data
   - Yang-Zhang for hourly streams
   - Close-to-Close for daily/monthly streams
3. **Configurable floor**: Minimum 30% annual volatility by default (ensures meaningful spreads)

### Pre-Settlement Cutoff

15 minutes before a market's settle_time, the bot:
1. Stops placing new orders for that market
2. Cancels all existing orders
3. Skips the market on subsequent cycles

This protects capital from oracle/settlement risk.

## Testnet vs Mainnet

To run against testnet for testing:

```bash
export TN_NODE_URL="https://gateway.testnet.truf.network"
```

Or set `node_url` in `config.yaml`:

```yaml
node_url: "https://gateway.testnet.truf.network"
```

## Testing

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_avellaneda.py -v

# With coverage
pytest tests/ --cov=market_maker_bot
```

## Architecture

```
src/market_maker_bot/
├── main.py                       # CLI entry point
├── config.py                     # Configuration models
├── bot.py                        # Main bot orchestrator
├── models.py                     # Data models
├── market.py                     # Market state management
├── order_state.py                # Order state persistence
├── pricing/
│   ├── avellaneda.py             # A-S pricing model
│   ├── black_scholes.py          # Binary option pricing
│   └── inventory.py              # Per-market inventory
├── indicators/
│   ├── volatility.py             # Order book volatility (RMS)
│   ├── stream_volatility.py      # Yang-Zhang / Close-to-Close
│   └── depth.py                  # Order book depth for kappa
└── utils/
    └── ring_buffer.py            # Efficient circular buffer
```

## Pre-Approved Streams

The bot ships with pre-approved streams in `src/market_maker_bot/config.py`. You can add custom streams by editing `APPROVED_STREAMS`:

```python
APPROVED_STREAMS: Dict[str, ApprovedStream] = {
    # ... existing ...
    "my_stream": ApprovedStream(
        stream_id="your_stream_id_here",
        name="My Custom Stream",
        description="Description",
        data_provider="0x...",  # Optional: defaults to Truflation provider
    ),
}
```

Then reference in `config.yaml`:

```yaml
markets:
  - query_id: 42
    stream_id: "your_stream_id_here"
    data_provider: "0x..."
    name: "My Custom Stream"
    outcome_mode: "both"
    order_amount: 100
```

## Risk Disclaimer

This software is provided as-is with no warranty. Market making involves financial risk, including:

- **Adverse selection**: informed traders may pick off stale quotes
- **Inventory risk**: holding shares exposes you to settlement outcomes
- **Oracle risk**: data provider issues can affect settlement
- **Smart contract risk**: protocol bugs or exploits
- **Gas/fee costs**: transaction fees eat into profits

The pre-settlement cutoff is a risk mitigation but does not eliminate all risks. Only provide liquidity with capital you can afford to lose. Understand the protocol mechanics before running live.

## Related Tools

- [Liquidity Provider Bot](https://github.com/truflation/liquidity-provider-bot) - Simpler bounds-based liquidity provider

## References

- [Avellaneda & Stoikov (2008)](https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf) - High-frequency trading in a limit order book
- [Yang & Zhang (2000)](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00280) - Drift Independent Volatility Estimation

## License

MIT
