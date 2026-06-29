import re
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

def tokenize_text(text):
    """
    Tokenize and normalize English text into an ordered list of tokens.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: Ordered list of normalized tokens
    """
    if not text or not text.strip():
        return []
    
    tokens = []
    i = 0
    current_token = ""
    
    while i < len(text):
        char = text[i]
        
        # Skip whitespace but use it to separate tokens
        if char.isspace():
            if current_token:
                tokens.append(_normalize_token(current_token))
                current_token = ""
            i += 1
            continue
        
        # Handle decimal numbers specially
        if char == '.' and _is_decimal_point(text, i, current_token):
            current_token += char
            i += 1
            continue
        
        # All punctuation and symbols are separated as individual tokens
        if not char.isalnum():
            # First, add any accumulated token
            if current_token:
                tokens.append(_normalize_token(current_token))
                current_token = ""
            # Then add the punctuation/symbol as its own token
            tokens.append(char)
            i += 1
            continue
        
        # Regular alphanumeric characters
        current_token += char
        i += 1
    
    # Add final token if exists
    if current_token:
        tokens.append(_normalize_token(current_token))
    
    return tokens

def _is_decimal_point(text, pos, current_token):
    """
    Check if a period is part of a decimal number
    """
    # Must have digits before and after the period
    if not current_token or not current_token[-1].isdigit():
        return False
    
    # Check if there's a digit after the period
    if pos < len(text) - 1 and text[pos + 1].isdigit():
        return True
    
    return False

def _normalize_token(token):
    """
    Normalize a token (lowercase alphabetic parts, keep numbers as-is)
    """
    if not token:
        return token
    
    # If token is purely numeric (including decimals), keep as-is
    if _is_numeric(token):
        return token
    
    # If token contains letters, lowercase them
    if any(c.isalpha() for c in token):
        return token.lower()
    
    # For other cases (pure punctuation, etc.), keep as-is
    return token

def _is_numeric(token):
    """
    Check if a token is numeric (including decimals)
    """
    # Handle decimal numbers
    if '.' in token:
        parts = token.split('.')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return True
    
    # Handle regular numbers
    return token.isdigit()

def test_tokenizer():
    """
    Test the tokenizer with the provided example and other cases
    """
    test_cases = [
        "NAC has developed a National HIV/AIDS/STI/TB Intervention Strategic Plan (2002-2005) thataims to reduce the HIV prevalence rate among Zambians from 19.3% to 11.7% and improve the healthstatus of people living with HIV/AIDS by 2005.",
        "Hello, world!",
        "I can't believe it's 3.14 degrees.",
        "The cost is $29.99.",
    ]
    
    print("Tokenization Results:")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        tokens = tokenize_text(text)
        print(f"Test {i}:")
        print(f"Input:  {text}")
        print(f"Tokens: {tokens}")
        print(f"Count:  {len(tokens)}")
        print("-" * 80)
    
    # Special test for the exact example provided
    expected_input = "NAC has developed a National HIV/AIDS/STI/TB Intervention Strategic Plan (2002-2005) thataims to reduce the HIV prevalence rate among Zambians from 19.3% to 11.7% and improve the healthstatus of people living with HIV/AIDS by 2005."
    expected_output = ['nac', 'has', 'developed', 'a', 'national', 'hiv', '/', 'aids', '/', 'sti', '/', 'tb', 'intervention', 'strategic', 'plan', '(', '2002', '-', '2005', ')', 'that', 'aims', 'to', 'reduce', 'the', 'hiv', 'prevalence', 'rate', 'among', 'zambians', 'from', '19.3', '%', 'to', '11.7', '%', 'and', 'improve', 'the', 'health', 'status', 'of', 'people', 'living', 'with', 'hiv', '/', 'aids', 'by', '2005', '.']
    
    actual_output = tokenize_text(expected_input)
    
    print("EXACT MATCH TEST:")
    print(f"Expected: {expected_output}")
    print(f"Actual:   {actual_output}")
    print(f"Match: {actual_output == expected_output}")
    
    if actual_output != expected_output:
        print("\nDifferences:")
        for i, (exp, act) in enumerate(zip(expected_output, actual_output)):
            if exp != act:
                print(f"  Position {i}: expected '{exp}', got '{act}'")
        if len(expected_output) != len(actual_output):
            print(f"  Length difference: expected {len(expected_output)}, got {len(actual_output)}")

def process_tokens_file():
    """
    Process the first ten lines from tokens.txt file
    """
    try:
        with open('tokens.txt', 'r', encoding='utf-8') as file:
            lines = []
            for i, line in enumerate(file):
                if i >= 10:  # Only read first 10 lines
                    break
                lines.append(line.rstrip('\n\r'))  # Remove newline characters but keep other whitespace
        
        print("Processing first 10 lines from tokens.txt:")
        print("=" * 80)
        
        for i, line in enumerate(lines, 1):
            if line.strip():  # Only process non-empty lines
                tokens = tokenize_text(line)
                print(f"Line {i}:")
                print(f"Input:  {line}")
                print(f"Tokens: {tokens}")
                print(f"Count:  {len(tokens)}")
                print("-" * 80)
    
    except FileNotFoundError:
        print("Error: tokens.txt file not found in the current directory.")
        print("Please make sure the tokens.txt file exists.")
    except Exception as e:
        print(f"Error reading tokens.txt: {e}")


def detect_sentence_boundaries(line):
    """
    Detect sentence boundaries in a line of text and return their zero-based character offsets.
    
    Args:
        line (str): Input line of text
        
    Returns:
        list: [count, offset1, offset2, ...] where count is number of sentences
              and offsets are zero-based positions of sentence-ending characters
    """
    if not line or not line.strip():
        return [0]
    
    # Common abbreviations that don't end sentences
    abbreviations = {
        'dr', 'mr', 'mrs', 'ms', 'prof', 'inc', 'ltd', 'etc', 'vs', 'st', 'ave',
        'blvd', 'dept', 'gov', 'sen', 'rep', 'gen', 'col', 'maj', 'capt', 'lt',
        'sgt', 'corp', 'pvt', 'jr', 'sr', 'phd', 'md', 'dds', 'dvm', 'esq',
        'co', 'corp', 'llc', 'llp', 'assn', 'org', 'univ', 'inst', 'dept',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'a.m', 'p.m', 'am', 'pm',
        'u.s', 'u.k', 'e.g', 'i.e', 'al', 'no', 'vol', 'pp', 'ch', 'sec', 'fig',
        'ft', 'in', 'yd', 'mi', 'oz', 'lb', 'pt', 'qt', 'gal', 'mph', 'rpm'
    }
    
    sentence_endings = []
    i = 0
    
    while i < len(line):
        char = line[i]
        
        # Check for sentence-ending punctuation
        if char in '.!?':
            is_sentence_end = True
            
            # Special handling for periods
            if char == '.':
                # Check if it's part of a number (decimal)
                if (i > 0 and i < len(line) - 1 and 
                    line[i-1].isdigit() and line[i+1].isdigit()):
                    is_sentence_end = False
                
                # Check if it's an abbreviation
                elif i > 0:
                    # Look backwards to find the start of the word before the period
                    word_start = i - 1
                    while word_start >= 0 and (line[word_start].isalnum() or line[word_start] == '.'):
                        word_start -= 1
                    word_start += 1
                    
                    # Extract the word (without the final period)
                    if word_start < i:
                        word = line[word_start:i].lower().replace('.', '')
                        if word in abbreviations:
                            # Check what comes after the period
                            if i < len(line) - 1:
                                next_char = line[i + 1]
                                # If followed by lowercase letter or another period, likely abbreviation
                                if next_char.islower() or next_char == '.':
                                    is_sentence_end = False
                                # If followed by space then lowercase, likely abbreviation
                                elif (next_char.isspace() and i < len(line) - 2 and 
                                      line[i + 2].islower()):
                                    is_sentence_end = False
            
            # Check for multiple punctuation (like "!!!" or "...")
            if char == '.' and i < len(line) - 1 and line[i + 1] == '.':
                # Skip ellipsis - find the end of the sequence
                while i < len(line) - 1 and line[i + 1] == '.':
                    i += 1
                # The sentence ends at the last period in the sequence
            
            if is_sentence_end:
                sentence_endings.append(i)
        
        i += 1
    
    # Check if the line ends without punctuation
    if line and not line.rstrip()[-1] in '.!?':
        # Find the last non-whitespace character
        last_char_pos = len(line.rstrip()) - 1
        if last_char_pos >= 0:
            sentence_endings.append(last_char_pos)
    
    # Return count followed by offsets
    return [len(sentence_endings)] + sentence_endings

def process_sentence_file(filename):
    """
    Process a file with sentences and detect sentence boundaries.
    
    Args:
        filename (str): Name of the input file
    """
    try:
        print(f"Processing sentence boundaries from {filename}:")
        print("=" * 80)
        
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.rstrip('\n\r')  # Remove newline but keep other whitespace
                if line.strip():  # Only process non-empty lines
                    result = detect_sentence_boundaries(line)
                    print(f"Line {line_num}: {line}")
                    print(f"Output: {' '.join(map(str, result))}")
                    print("-" * 80)
    
    except FileNotFoundError:
        print(f"Error: {filename} file not found in the current directory.")
    except Exception as e:
        print(f"Error processing {filename}: {e}")


def nltk_tokenize_text(text):
    """
    Tokenize text using NLTK's word_tokenize and normalize tokens.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: Ordered list of normalized tokens using NLTK
    """
    if not text or not text.strip():
        return []
    
    # Use NLTK's word_tokenize
    tokens = word_tokenize(text)
    
    # Normalize tokens similar to the original method
    normalized_tokens = []
    for token in tokens:
        normalized_tokens.append(_normalize_token(token))
    
    return normalized_tokens


def nltk_process_tokens_file():
    """
    Process the first ten lines from tokens.txt file using only NLTK methods
    """
    try:
        with open('tokens.txt', 'r', encoding='utf-8') as file:
            lines = []
            for i, line in enumerate(file):
                if i >= 10:  # Only read first 10 lines
                    break
                lines.append(line.rstrip('\n\r'))
        
        print("Processing first 10 lines from tokens.txt (NLTK only):")
        print("=" * 80)
        
        for i, line in enumerate(lines, 1):
            if line.strip():  # Only process non-empty lines
                tokens = nltk_tokenize_text(line)
                print(f"Line {i}:")
                print(f"Input:  {line}")
                print(f"Tokens: {tokens}")
                print(f"Count:  {len(tokens)}")
                print("-" * 80)
    
    except FileNotFoundError:
        print("Error: tokens.txt file not found in the current directory.")
        print("Please make sure the tokens.txt file exists.")
    except Exception as e:
        print(f"Error reading tokens.txt: {e}")

def nltk_process_sentence_file(filename):
    """
    Process a file with sentences using only NLTK methods.
    
    Args:
        filename (str): Name of the input file
    """
    try:
        print(f"Processing sentence boundaries from {filename} (NLTK only):")
        print("=" * 80)
        
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.rstrip('\n\r')
                if line.strip():  # Only process non-empty lines
                    result = nltk_detect_sentence_boundaries(line)
                    print(f"Line {line_num}: {line}")
                    print(f"Output: {' '.join(map(str, result))}")
                    print("-" * 80)
    
    except FileNotFoundError:
        print(f"Error: {filename} file not found in the current directory.")
    except Exception as e:
        print(f"Error processing {filename}: {e}")


def nltk_detect_sentence_boundaries(line):
    """
    Detect sentence boundaries using NLTK's sent_tokenize and return character offsets.
    
    Args:
        line (str): Input line of text
        
    Returns:
        list: [count, offset1, offset2, ...] where count is number of sentences
              and offsets are zero-based positions of sentence-ending characters
    """
    if not line or not line.strip():
        return [0]
    
    # Use NLTK's sent_tokenize to split into sentences
    sentences = sent_tokenize(line)
    
    if not sentences:
        return [0]
    
    sentence_endings = []
    current_pos = 0
    
    for sentence in sentences:
        # Find this sentence in the original line
        sentence_start = line.find(sentence, current_pos)
        if sentence_start == -1:
            # Fallback: search from current position
            sentence_start = current_pos
        
        # The sentence ends at the last character of the sentence
        sentence_end = sentence_start + len(sentence) - 1
        sentence_endings.append(sentence_end)
        
        # Update current position for next search
        current_pos = sentence_start + len(sentence)
    
    # Return count followed by offsets
    return [len(sentence_endings)] + sentence_endings


if __name__ == "__main__":
    #test_tokenizer()
    #process_tokens_file()
    #process_sentence_file('sentences.txt')
    """
    print("\n" + "="*80)
    print("Processing tokens file with NLTK only...")
    nltk_process_tokens_file()
    """
    
    print("\n" + "="*80)
    print("Processing sentence file with NLTK only...")
    nltk_process_sentence_file('sentences.txt')
    
