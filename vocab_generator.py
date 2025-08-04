import json
from transformers import AutoTokenizer

def generate_vocab_json(base_model="gpt2", output_path="vocab.json"):
    """
    Generate a vocab.json file based on an existing tokenizer
    
    Args:
        base_model: Base model to use for tokenizer (default: "gpt2")
        output_path: Output path for vocab.json
    """
    try:
        # Load the tokenizer
        print(f"Loading tokenizer from {base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Get the vocabulary
        vocab = tokenizer.get_vocab()
        
        # Sort by token ID for consistency
        sorted_vocab = dict(sorted(vocab.items(), key=lambda x: x[1]))
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_vocab, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Vocabulary saved to {output_path}")
        print(f"📊 Vocabulary size: {len(vocab)}")
        print(f"🔤 Sample tokens: {list(vocab.keys())[:10]}")
        
        return vocab
        
    except Exception as e:
        print(f"❌ Error generating vocabulary: {e}")
        return None

def generate_custom_vocab(vocab_size=50257, output_path="vocab.json"):
    """
    Generate a custom vocabulary for Mamba models
    
    Args:
        vocab_size: Size of vocabulary (default: 50257 for GPT-2 compatibility)
        output_path: Output path for vocab.json
    """
    vocab = {}
    
    # Add special tokens
    special_tokens = {
        "<pad>": 0,
        "<unk>": 1, 
        "<s>": 2,
        "</s>": 3,
        "<mask>": 4
    }
    
    vocab.update(special_tokens)
    token_id = len(special_tokens)
    
    # Add common characters and symbols
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?/~`"
    
    for char in chars + symbols + " \n\t":
        if char not in vocab:
            vocab[char] = token_id
            token_id += 1
    
    # Add common subword tokens (simplified BPE-style)
    common_subwords = [
        "the", "and", "ing", "ion", "tion", "er", "re", "ed", "in", "on",
        "at", "is", "it", "or", "as", "are", "was", "be", "an", "have",
        "had", "has", "will", "would", "could", "should", "can", "may",
        "not", "but", "if", "when", "where", "what", "how", "who", "why"
    ]
    
    for word in common_subwords:
        if word not in vocab:
            vocab[word] = token_id
            token_id += 1
    
    # Fill remaining slots with dummy tokens
    while token_id < vocab_size:
        vocab[f"<extra_token_{token_id}>"] = token_id
        token_id += 1
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Custom vocabulary saved to {output_path}")
    print(f"📊 Vocabulary size: {len(vocab)}")
    
    return vocab

if __name__ == "__main__":
    # Option 1: Generate from existing model (recommended)
    print("Option 1: Generate from GPT-2 tokenizer")
    generate_vocab_json("gpt2", "vocab.json")
    
    print("\n" + "="*50 + "\n")
    
    # Option 2: Generate custom vocabulary
    print("Option 2: Generate custom vocabulary")
    generate_custom_vocab(50257, "custom_vocab.json")
    
    print("\n🔍 Choose the vocab.json that matches your model's expected vocabulary size!") 