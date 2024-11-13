# Receipt Analyzer

An intelligent OCR-based receipt analysis tool that extracts key information from receipt images.

## Features

- Extracts merchant name, date, amount, and currency
- Supports multiple receipt types (invoice, receipt, order)
- Handles multiple currency formats (USD, EUR, GBP, CNY, HKD, JPY)
- Provides confidence scores for extracted data
- Includes fuzzy matching for better accuracy
- Caches results for improved performance

## Requirements

- Python 3.7+
- doctr
- matplotlib
- fuzzywuzzy
- python-Levenshtein (optional, for better performance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/xer0bit/receipt-analyzer.git
cd receipt-analyzer
pip install -r requirements.txt
```

## Usage

### Basic usage example:

```python
from analyzer import analyze_receipt

result = analyze_receipt('path/to/receipt.jpg')
print(result)
```

## Output Format

The analyzer returns a dictionary containing:

- `merchant_name`: Extracted business name
- `bill_date`: Date in YYYY-MM-DD format
- `amount`: Transaction amount
- `currency`: Detected currency code
- `description`: Transaction description
- `type`: Receipt type (invoice/receipt/order)
- `confidence_score`: Confidence level (0-1)

## Configuration

Customize the behavior by modifying `config.py`:

- Supported currencies
- Keywords for receipt classification
- Path configurations

## Error Handling

The analyzer includes validation checks and will raise `ValueError` for invalid receipts. Always wrap the `analyze_receipt` call in a try-except block:

```python
try:
    result = analyze_receipt('path/to/receipt.jpg')
except ValueError as e:
    print(f"Error: {e}")
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.