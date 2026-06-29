import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def process_with_nltk():
    """
    Process input files using NLTK's word_tokenize and sent_tokenize methods.
    Processes both tokens.txt and sentences.txt files.
    """
    print("Processing files with NLTK tokenizers...")
    print("=" * 80)
    
    # Process tokens.txt file
    print("PROCESSING TOKENS.TXT:")
    print("-" * 40)
    try:
        with open('tokens.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Tokenize using NLTK's word_tokenize
        nltk_tokens = word_tokenize(content)
        
        print(f"Total tokens found by NLTK: {len(nltk_tokens)}")
        print(f"First 20 tokens: {nltk_tokens[:20]}")
        print(f"Last 20 tokens: {nltk_tokens[-20:]}")
        
        # Count unique tokens
        unique_tokens = set(nltk_tokens)
        print(f"Unique tokens: {len(unique_tokens)}")
        
        # Show some statistics
        token_lengths = [len(token) for token in nltk_tokens]
        print(f"Average token length: {sum(token_lengths) / len(token_lengths):.2f}")
        print(f"Shortest token: {min(token_lengths)} characters")
        print(f"Longest token: {max(token_lengths)} characters")
        
    except FileNotFoundError:
        print("Error: tokens.txt file not found.")
    except Exception as e:
        print(f"Error processing tokens.txt: {e}")
    
    print("\n" + "=" * 80)
    
    # Process sentences.txt file
    print("PROCESSING SENTENCES.TXT:")
    print("-" * 40)
    try:
        with open('sentences.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Tokenize sentences using NLTK's sent_tokenize
        nltk_sentences = sent_tokenize(content)
        
        print(f"Total sentences found by NLTK: {len(nltk_sentences)}")
        print(f"First 3 sentences:")
        for i, sentence in enumerate(nltk_sentences[:3], 1):
            print(f"  {i}. {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
        
        # Tokenize each sentence into words
        all_words = []
        for sentence in nltk_sentences:
            words = word_tokenize(sentence)
            all_words.extend(words)
        
        print(f"\nTotal words across all sentences: {len(all_words)}")
        print(f"Unique words: {len(set(all_words))}")
        
        # Show sentence length statistics
        sentence_lengths = [len(word_tokenize(sent)) for sent in nltk_sentences]
        print(f"Average words per sentence: {sum(sentence_lengths) / len(sentence_lengths):.2f}")
        print(f"Shortest sentence: {min(sentence_lengths)} words")
        print(f"Longest sentence: {max(sentence_lengths)} words")
        
        # Show some example sentences with their word counts
        print(f"\nSample sentences with word counts:")
        for i, (sentence, word_count) in enumerate(zip(nltk_sentences[:5], sentence_lengths[:5])):
            print(f"  {i+1}. ({word_count} words) {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
        
    except FileNotFoundError:
        print("Error: sentences.txt file not found.")
    except Exception as e:
        print(f"Error processing sentences.txt: {e}")
    
    print("\n" + "=" * 80)
    print("NLTK processing completed!")

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

def process_tokens_file_stats():
    """
    Process the entire tokens.txt file and collect statistics
    """
    try:
        token_counts = {}  # Dictionary to store token frequencies
        total_tokens = 0
        line_count = 0
        
        print("Processing entire tokens.txt file...")
        
        with open('tokens.txt', 'r', encoding='utf-8') as file:
            for line in file:
                line = line.rstrip('\n\r')
                if line.strip():  # Only process non-empty lines
                    line_count += 1
                    tokens = tokenize_text(line)
                    
                    # Count each token
                    for token in tokens:
                        total_tokens += 1
                        token_counts[token] = token_counts.get(token, 0) + 1
        
        # Sort tokens by frequency (descending order)
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate vocabulary size
        vocab_size = len(token_counts)
        
        # Calculate hapax legomena (tokens that occur exactly once)
        singletons = sum(1 for count in token_counts.values() if count == 1)
        singleton_percentage = (singletons / vocab_size) * 100 if vocab_size > 0 else 0
        
        # Print statistics
        print(f"{line_count} Number of lines")
        print(f"{vocab_size} Number of unique tokens (vocabulary size)")
        print(f"{total_tokens} Number of tokens (collection size)")
        
        # Print top tokens at specific ranks
        ranks_to_show = [1, 2, 3, 4, 5, 100, 500, 1000, 5000, 10000]
        
        for rank in ranks_to_show:
            if rank <= len(sorted_tokens):
                token, count = sorted_tokens[rank - 1]  # rank is 1-based, list is 0-based
                if rank <= 5:
                    print(f"{rank}. {token} {count}")
                elif rank == 100:
                    print("...")
                    print(f"{rank}. {token} {count}")
                else:
                    print(f"{rank}. {token} {count}")
        
        # Print hapax legomena statistics
        print(f"{singletons} Number of singleton terms")
        print(f"{singleton_percentage:.3f}% Percentage of singletons")
        
        return {
            'line_count': line_count,
            'vocab_size': vocab_size,
            'total_tokens': total_tokens,
            'token_counts': token_counts,
            'sorted_tokens': sorted_tokens,
            'singletons': singletons,
            'singleton_percentage': singleton_percentage
        }
    
    except FileNotFoundError:
        print("Error: tokens.txt file not found in the current directory.")
        print("Please make sure the tokens.txt file exists.")
        return None
    except Exception as e:
        print(f"Error processing tokens.txt: {e}")
        return None


def plot_zipf_analysis(stats_data):
    """
    Create a log-log plot to analyze Zipf's law for the token frequencies
    """
    if not stats_data:
        print("No statistics data available for plotting.")
        return
    
    sorted_tokens = stats_data['sorted_tokens']
    
    # Extract ranks (1-based) and frequencies
    ranks = list(range(1, len(sorted_tokens) + 1))
    frequencies = [count for token, count in sorted_tokens]
    
    # Create the log-log plot
    plt.figure(figsize=(12, 8))
    
    # Plot actual data
    plt.loglog(ranks, frequencies, 'b.', alpha=0.6, markersize=3, label='Actual Data')
    
    # Calculate theoretical Zipf's law line
    # Zipf's law: frequency = k / rank, where k is approximately the frequency of the most common word
    k = frequencies[0]  # Frequency of most common word
    theoretical_freq = [k / rank for rank in ranks]
    
    plt.loglog(ranks, theoretical_freq, 'r-', alpha=0.8, linewidth=2, label=f"Zipf's Law (k={k})")
    
    # Add labels and title
    plt.xlabel('Rank (log scale)', fontsize=12)
    plt.ylabel('Frequency (log scale)', fontsize=12)
    plt.title('Zipf\'s Law Analysis: Frequency vs. Rank (Log-Log Plot)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some statistics to the plot
    plt.text(0.05, 0.95, f'Vocabulary Size: {len(sorted_tokens):,}\n'
                         f'Total Tokens: {stats_data["total_tokens"]:,}\n'
                         f'Most Frequent: "{sorted_tokens[0][0]}" ({sorted_tokens[0][1]:,})',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('zipf_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate some Zipf's law metrics
    print("\nZipf's Law Analysis:")
    print("=" * 50)
    
    # Calculate rank * frequency for top terms to see if it's roughly constant
    print("Rank × Frequency products for top 20 terms:")
    for i in range(min(20, len(sorted_tokens))):
        rank = i + 1
        token, freq = sorted_tokens[i]
        product = rank * freq
        print(f"{rank:2d}. {token:15s} freq={freq:5d} rank×freq={product:6d}")
    
    # Calculate correlation coefficient between log(rank) and log(frequency)
    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)
    correlation = np.corrcoef(log_ranks, log_frequencies)[0, 1]
    
    print(f"\nCorrelation coefficient between log(rank) and log(frequency): {correlation:.4f}")
    print("(Perfect Zipf's law would show correlation ≈ -1.0)")
    
    # Calculate the slope of the log-log relationship
    slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)
    print(f"Slope of log-log relationship: {slope:.4f}")
    print("(Perfect Zipf's law would show slope ≈ -1.0)")
    
    return {
        'correlation': correlation,
        'slope': slope,
        'intercept': intercept,
        'ranks': ranks,
        'frequencies': frequencies
    }


if __name__ == "__main__":
    # Process files with NLTK
    process_with_nltk()
    
    print("\n" + "=" * 80)
    print("FULL FILE STATISTICS (Custom Tokenizer):")
    print("=" * 80)
    stats = process_tokens_file_stats()
    if stats:
        print("\n" + "=" * 80)
        print("ZIPF'S LAW ANALYSIS:")
        print("=" * 80)
        zipf_results = plot_zipf_analysis(stats)

    #test_tokenizer()
    #process_tokens_file()
