from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import matplotlib.pyplot as plt
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from config import CONFIG, KEYWORDS
from fuzzywuzzy import fuzz
from functools import lru_cache

class ReceiptAnalyzer:
    def __init__(self):
        self.model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        self.receipt_types = {
            'invoice': ['invoice', 'bill to', 'payment due', 'invoice no'],
            'receipt': ['receipt', 'thank you', 'served by', 'cashier'],
            'order': ['order', 'delivery', 'purchase order', 'shipping']
        }
        self.currency_patterns = {
            'HKD': [r'HK\$', r'HKD', r'港币', r'港幣'],
            'CNY': [r'¥', r'CNY', r'RMB', r'元', r'CHY', r'人民币'],
            'USD': [r'USD', r'US\$', r'\$(?!HK)'],  # $ not followed by HK
            'EUR': [r'€', r'EUR'],
            'GBP': [r'£', r'GBP'],
            'JPY': [r'JPY', r'円', r'¥']
        }

    @lru_cache(maxsize=100)
    def analyze_receipt(self, image_path: str) -> Dict:
        doc = DocumentFile.from_images(image_path)
        result = self.model(doc)
        
        extracted_data = {
            'merchant_name': None,
            'bill_date': None,
            'amount': None,
            'description': None,
            'type': None,
            'currency': None,
            'confidence_score': 0.0
        }

        text_blocks = self._extract_text_blocks(result)
        
        # Detect currency first
        currency, currency_confidence = self._detect_currency(text_blocks)
        extracted_data['currency'] = currency
        
        # Process other fields
        extracted_data['bill_date'] = self._find_date(text_blocks)
        extracted_data['merchant_name'], merchant_confidence = self._find_merchant(text_blocks)
        
        # Update amounts with detected currency
        amounts = self._extract_amounts(text_blocks, currency)
        extracted_data.update(amounts)
        extracted_data['description'] = self._get_description(text_blocks)
        extracted_data['type'] = self._classify_receipt_type(text_blocks)
        
        # Add debug output before validation
        print("\nExtracted Data before validation:")
        print(f"Text blocks found: {len(text_blocks)}")
        print(f"Extracted data: {extracted_data}")
        
        # Validate results
        extracted_data['confidence_score'] = self._calculate_confidence(extracted_data)
        
        if not self._validate_receipt(extracted_data):
            raise ValueError("Invalid receipt data detected")
            
        return extracted_data

    def _extract_text_blocks(self, result) -> List[str]:
        text_blocks = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    text = ' '.join(word.value for word in line.words)
                    text_blocks.append(text.strip())
        return text_blocks

    def _find_date(self, text_blocks: List[str]) -> Optional[str]:
        date_keywords = ['date:', 'dated:', 'bill date:', 'invoice date:']
        for text in text_blocks:
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in date_keywords):
                # Prioritize lines with date keywords
                if match := self._extract_date_from_text(text):
                    return match
        
        # Fallback to general date search
        for text in text_blocks:
            if match := self._extract_date_from_text(text):
                return match
        return None

    def _extract_date_from_text(self, text: str) -> Optional[str]:
        date_patterns = [
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',      # DD-MM-YYYY
            r'\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}'  # DD MMM YYYY
        ]
        
        for pattern in date_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                try:
                    return self._normalize_date(match.group(0))
                except ValueError:
                    continue
        return None

    def _normalize_date(self, date_str: str) -> str:
        # Remove any  characters if present
        date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '')
        
        # Try different date formats
        date_formats = [
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%m-%d-%Y',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d %b %Y',
            '%d %B %Y'
        ]
        
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        raise ValueError(f"Unable to parse date: {date_str}")

    def _normalize_currency(self, currency: str) -> Optional[str]:
        if not currency:
            return None
            
        currency = currency.strip().upper()
        
        # Expanded currency mapping with variants
        currency_map = {
            '$': 'USD',
            'HK$': 'HKD',
            'HKD': 'HKD',
            '港币': 'HKD',
            '港幣': 'HKD',
            '£': 'GBP',
            '€': 'EUR',
            '¥': 'CNY',
            'CNY': 'CNY',
            'CHY': 'CNY',
            'RMB': 'CNY',
            '元': 'CNY',
            '人民币': 'CNY',
            'JPY': 'JPY',
            '円': 'JPY'
        }
        
        return currency_map.get(currency, currency)

    def _extract_amounts(self, text_blocks: List[str], detected_currency: Optional[str] = None) -> Dict:
        amounts = {'amount': None, 'currency': detected_currency}
        total_indicators = ['total', 'amount', 'sum', 'due', 'pay', 'balance']
        
        # Use detected currency in pattern if available
        if detected_currency:
            currencies = self.currency_patterns.get(detected_currency, [])
            amount_pattern = fr'({"|".join(currencies)})?\s*(\d+[.,]\d{{2}})'
        else:
            amount_pattern = self._get_amount_pattern()
            
        # First pass: Look for amounts with total indicators
        for text in reversed(text_blocks):  # Start from bottom
            text_lower = text.lower()
            if any(indicator in text_lower for indicator in total_indicators):
                if match := re.search(amount_pattern, text):
                    print(f"Found amount with indicator: {text}")
                    amounts['amount'] = self._normalize_amount(match.group(2))
                    amounts['currency'] = self._normalize_currency(match.group(1))
                    return amounts

        # Second pass: Look for largest amount in last 10 lines
        amounts_found = []
        for text in text_blocks[-10:]:
            if match := re.search(amount_pattern, text):
                try:
                    amount = float(self._normalize_amount(match.group(2)))
                    curr = self._normalize_currency(match.group(1))
                    amounts_found.append((amount, curr, text))
                except ValueError:
                    continue

        if amounts_found:
            # Sort by amount value, get largest
            largest = max(amounts_found, key=lambda x: x[0])
            print(f"Selected largest amount: {largest[2]}")
            amounts['amount'] = str(largest[0])
            amounts['currency'] = largest[1]

        return amounts

    def _extract_items(self, text_blocks: List[str]) -> List[Dict]:
        items = []
        currencies = '|'.join(map(re.escape, CONFIG["SUPPORTED_CURRENCIES"]))
        item_pattern = fr'(.*?)\s*({currencies})?\s*(\d+[.,]\d{{2}})'
        
        for text in text_blocks:
            if match := re.search(item_pattern, text):
                items.append({
                    'description': match.group(1).strip(),
                    'price': self._normalize_amount(match.group(3)),
                    'currency': self._normalize_currency(match.group(2))
                })
        return items

    def _validate_receipt(self, data: Dict) -> bool:
        print("\nValidation Details:")
        print(f"Merchant Name: {data['merchant_name']}")
        print(f"Amount: {data['amount']}")
        print(f"Date: {data['bill_date']}")
        
        # Require at least merchant name OR amount
        if not (data['merchant_name'] or data['amount']):
            print("Validation failed: Missing both merchant name and amount")
            return False
            
        return True

    def _find_merchant(self, text_blocks: List[str]) -> Tuple[Optional[str], float]:
        merchant_indicators = ['ltd', 'limited', 'inc', 'corp', 'co', 'company', 'store', 'restaurant', 'shop']
        
        # First pass: Look for business indicators
        for text in text_blocks[:7]:  # Check first 7 lines
            cleaned_text = self._preprocess_text(text)
            if any(indicator in cleaned_text for indicator in merchant_indicators):
                merchant = re.sub(r'\s*(ltd|limited|inc|corp)\.?\s*$', '', text, flags=re.IGNORECASE)
                print(f"Found merchant with indicator: {merchant}")
                return merchant.strip(), 0.9
        
        # Second pass: Look for longest capitalized line
        candidates = []
        for text in text_blocks[:5]:
            text = text.strip()
            if (len(text) > 3 and 
                not self._is_unwanted_merchant_line(text.lower()) and
                any(c.isupper() for c in text)):
                candidates.append((text, len(text)))
        
        if candidates:
            # Choose the longest candidate
            merchant = max(candidates, key=lambda x: x[1])[0]
            print(f"Found merchant by capitalization: {merchant}")
            return merchant, 0.7
        
        # Last resort: first non-empty line
        for text in text_blocks[:3]:
            if len(text.strip()) > 3:
                print(f"Found merchant as first line: {text}")
                return text.strip(), 0.5
                
        return None, 0.0

    def _calculate_merchant_confidence(self, merchant: str) -> float:
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on various factors
        if len(merchant.split()) > 1:  # Multiple words likely a business name
            confidence += 0.1
        if any(word.isupper() for word in merchant.split()):  # Contains uppercase words
            confidence += 0.1
        if len(merchant) > 3 and not merchant.isdigit():  # Reasonable length, not just numbers
            confidence += 0.1
            
        return min(confidence, 1.0)

    def _get_description(self, text_blocks: List[str]) -> Optional[str]:
        # Look for description or item details
        description_markers = ['description:', 'item:', 'product:', 'service:']
        
        for text in text_blocks:
            text_lower = text.lower()
            if any(marker in text_lower for marker in description_markers):
                # Extract text after the marker
                for marker in description_markers:
                    if marker in text_lower:
                        return text.split(marker)[-1].strip()
                        
        # Fallback: Use first item description if available
        if items := self._extract_items(text_blocks):
            return items[0].get('description', '')
            
        return None

    def _find_subtotal(self, text_blocks: List[str]) -> Optional[str]:
        subtotal_keywords = ['subtotal', 'sub-total', 'sub total', 'net']
        for text in text_blocks:
            text_upper = text.upper()
            if any(keyword in text_upper for keyword in subtotal_keywords):
                if match := re.search(self._get_amount_pattern(), text):
                    return self._normalize_amount(match.group(2))
        return None

    def _get_amount_pattern(self) -> str:
        """Centralized amount pattern generation"""
        currencies = '|'.join(map(re.escape, CONFIG["SUPPORTED_CURRENCIES"]))
        return fr'({currencies})?\s*(\d+[.,]\d{{2}})'

    def _classify_receipt_type(self, text_blocks: List[str]) -> str:
        text_content = ' '.join(text_blocks).lower()
        type_scores = {}
        
        for receipt_type, keywords in self.receipt_types.items():
            score = sum(keyword in text_content for keyword in keywords)
            type_scores[receipt_type] = score
            
        if not type_scores or max(type_scores.values()) == 0:
            return 'general'
            
        return max(type_scores.items(), key=lambda x: x[1])[0]

    def _calculate_confidence(self, data: Dict) -> float:
        # Enhanced confidence calculation
        confidence = 0.0
        weights = {
            'bill_date': 0.2,
            'amount': 0.25,
            'merchant_name': 0.2,
            'description': 0.15,
            'currency': 0.1,
            'type': 0.1
        }
        
        # Base confidence from required fields
        for field, weight in weights.items():
            if field == 'description' and data[field]:
                items_confidence = min(len(data[field]) * 0.05, weight)
                confidence += items_confidence
            elif data.get(field):
                confidence += weight
        
        # Additional validation checks
        if data.get('subtotal') and data.get('amount'):
            if float(data['subtotal']) < float(data['amount']):
                confidence += 0.1
                
        if data.get('description') and data.get('amount'):
            items_total = sum(float(item['price']) for item in data['description'])
            total = float(data['amount'])
            if 0.9 <= items_total/total <= 1.1:
                confidence += 0.1
                
        return round(min(confidence, 1.0), 2)

    def _is_unwanted_merchant_line(self, text: str) -> bool:
        """Check if a line should be excluded from merchant detection"""
        unwanted = ['total', 'tax', 'date', 'tel', 'receipt', 'cash', 'card', 
                   'change', 'balance', 'due', 'paid', 'amount', 'time']
        return any(word in text for word in unwanted)

    def _normalize_amount(self, amount_str: str) -> str:
        # Remove any whitespace and currency symbols
        amount_str = amount_str.strip()
        
        # Replace comma with dot for decimal point standardization
        amount_str = amount_str.replace(',', '.')
        
        # Convert to float and back to string to standardize format
        try:
            amount_float = float(amount_str)
            return f"{amount_float:.2f}"
        except ValueError:
            raise ValueError(f"Unable to parse amount: {amount_str}")

    def _preprocess_text(self, text: str) -> str:
        """Clean and standardize text for better matching"""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        return text

    def _detect_currency(self, text_blocks: List[str]) -> Tuple[Optional[str], float]:
        """Detect currency with confidence score"""
        currency_counts = {}
        currency_positions = {}
        
        # Combine all currency patterns
        for currency, patterns in self.currency_patterns.items():
            combined_pattern = '|'.join(patterns)
            for i, text in enumerate(text_blocks):
                matches = re.finditer(combined_pattern, text, re.IGNORECASE)
                for match in matches:
                    currency_counts[currency] = currency_counts.get(currency, 0) + 1
                    # Store position of currency mention
                    if currency not in currency_positions:
                        currency_positions[currency] = i

        if not currency_counts:
            return None, 0.0

        # Get the most frequent currency
        max_currency = max(currency_counts.items(), key=lambda x: x[1])
        currency_code = max_currency[0]
        confidence = min(max_currency[1] / len(text_blocks), 0.9)  # Cap at 0.9

        # Boost confidence if currency appears near amounts
        if self._is_currency_near_amounts(currency_code, currency_positions[currency_code], text_blocks):
            confidence = min(confidence + 0.1, 1.0)

        return currency_code, confidence

    def _is_currency_near_amounts(self, currency: str, position: int, text_blocks: List[str], window: int = 2) -> bool:
        """Check if currency appears near amount patterns"""
        start = max(0, position - window)
        end = min(len(text_blocks), position + window + 1)
        
        amount_pattern = r'\d+[.,]\d{2}'
        for i in range(start, end):
            if re.search(amount_pattern, text_blocks[i]):
                return True
        return False

    def visualize_result(self, doc_result):
        # Display the OCR result with bounding boxes
        doc_result.show()
        plt.show()

    def __str__(self):
        """Pretty print the important fields"""
        return f"""
Receipt Details:
--------------
Merchant: {self.merchant_name}
Date: {self.bill_date}
Amount: {self.currency}{self.amount}
Type: {self.type}
Description: {self.description}
Confidence: {self.confidence_score}
"""

if __name__ == "__main__":
    analyzer = ReceiptAnalyzer()
    receipt_path = "test.jpg"
    try:
        results = analyzer.analyze_receipt(receipt_path)
        print("\nFinal Results:")
        print("-" * 30)
        print(f"Merchant: {results['merchant_name']}")
        print(f"Date: {results['bill_date']}")
        print(f"Amount: {results['amount']} {results['currency']}")
        print(f"Type: {results['type']}")
        print(f"Description: {results['description']}")
        print(f"Confidence: {results['confidence_score']}")
    except Exception as e:
        print(f"Error processing receipt: {str(e)}")