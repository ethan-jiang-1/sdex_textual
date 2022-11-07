#Get token ids for our placeholder and initializer token. This code block will complain if initializer string is not a single token
# Add the placeholder token in tokenizer
def add_new_tokens_to_tokenizer(tokenizer, initializer_tokens, placeholder_tokens):
    print(tokenizer)

    #num_added_tokens = tokenizer.add_tokens(placeholder_token)
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_tokens}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    print(num_added_tokens)

    for initializer_token, placeholder_token in zip(initializer_tokens, placeholder_tokens):
        # Convert the initializer_token, placeholder_token to ids
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

        print(initializer_token, token_ids)

        print('initializer_token_id', initializer_token_id)
        print('placeholder_token_id', placeholder_token_id)

    print(tokenizer)

def list_new_tokens(tokenizer):
    _new_ids = [tokenizer.vocab_size - id for id in [-3, -2, -1,0,1,2,3,4,5,6,7,8]]
    _new_ids = sorted(_new_ids)
    _new_tokens = tokenizer.convert_ids_to_tokens(_new_ids)
    print(_new_ids)
    print(_new_tokens)

def add_new_token_to_encoder_with_init_embedding(tokenizer, text_encoder, initializer_tokens, placeholder_tokens):
    # vocab_size of tokenizer does not change -- but, as long as text_encoder change, we are fine

    print("before adding more embedding", text_encoder.get_input_embeddings())
    text_encoder.resize_token_embeddings(len(tokenizer))
    print("after adding more embedding ", text_encoder.get_input_embeddings())

    for initializer_token, placeholder_token in zip(initializer_tokens, placeholder_tokens):
        placeholder_token_id = tokenizer.encode(placeholder_token, add_special_tokens=False)[0]
        initializer_token_id = tokenizer.encode(initializer_token, add_special_tokens=False)[0]
        print("placeholder_token_id", placeholder_token_id, initializer_token)
        print("initializer_token_id", initializer_token_id, placeholder_token)

        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
        print(token_embeds.shape)

