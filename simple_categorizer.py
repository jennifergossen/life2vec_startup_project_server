def _categorize_startup_tokens_simple(self):
    """
    Simple categorizer based on your actual token structure
    Generated automatically from your project analysis
    """
    token_types = {}
    
    for token in self.concept_tokens:
        assigned = False
        
        # Check all_tokens
        if not assigned and token in ['Goodness Kayode', 'Pleasant Grove', '2023-11-21']:  # Add your full token list here
            token_types[token] = 'all_tokens'
            assigned = True
            
        # Default categories
        if not assigned:
            if token.startswith('[') and token.endswith(']'):
                token_types[token] = 'special'
            elif '_' in token:
                prefix = token.split('_')[0]
                token_types[token] = f'prefix_{prefix.lower()}'
            else:
                token_types[token] = 'other'
    
    return token_types
