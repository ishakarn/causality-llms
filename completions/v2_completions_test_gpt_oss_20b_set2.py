import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# GPT model
MODEL_NAME = "openai/gpt-oss-20b"
OUTPUT_FILE = "completions/outputs/gptoss20b_full_audit_set2.txt"

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model, device

# PM: I apologize in advance for this mess of code, I was trying to get it ready to test as fast as possible. You shouldn't need to modify anything under this line.



RAW_TOPK = 1000
SYSTEM_PROMPT = "Continue the text."
prefix = "{\"question_id\": 3509, \"desc_id\": \"blood_pressure-mediation-nie-model306-spec6-q1\", \"given_info\": \""
full_prompt = "{\"question_id\": 3509, \"desc_id\": \"blood_pressure-mediation-nie-model306-spec6-q1\", \"given_info\": \"For people not taking any medication and with low blood pressure, the probability of healthy heart is 48%. For people not taking any medication and with high blood pressure, the probability of healthy heart is 20%. For people taking medication and with low blood pressure, the probability of healthy heart is 77%. For people taking medication and with high blood pressure, the probability of healthy heart is 46%. For people not taking any medication, the probability of high blood pressure is 37%. For people taking medication, the probability of high blood pressure is 60%. Does medication negatively affect heart condition through blood pressure?"


PUNCTUATION_TOKENS = ["?", ",", ".", "%"]
ALLOWED_PUNCTUATION_INSIDE_GENERATED_TOKEN = ["?", ",", ".", "'", "%"]


# ------------------------------------------------------------
# Custom tokenization of the target query
# ------------------------------------------------------------

def tokenize_query_text(query_text):
    """
    Custom tokenizer.

    Splits into:
    - word tokens:   Husband, don't, does
    - number tokens: 50, 5
    - punctuation:   ?, ,, ., %

    Spaces are not kept as tokens.

    Example:
    "Husband has 50%. If husband does not, then"
    becomes:
    "Husband", "has", "50", "%", ".", "If", "husband", "does", "not", ",", "then"
    """
    tokens = []
    i = 0

    while i < len(query_text):
        ch = query_text[i]

        # Skip spaces between target tokens.
        if ch.isspace():
            i += 1
            continue

        # Number token
        if ch.isdigit():
            start = i
            while i < len(query_text) and query_text[i].isdigit():
                i += 1
            tokens.append(query_text[start:i])
            continue

        # Punctuation token
        if ch in PUNCTUATION_TOKENS:
            tokens.append(ch)
            i += 1
            continue

        # Word token. Apostrophes are treated as part of words.
        # Example: don't
        if ch.isalpha() or ch == "'":
            start = i
            while i < len(query_text):
                if query_text[i].isalpha() or query_text[i] == "'":
                    i += 1
                else:
                    break
            tokens.append(query_text[start:i])
            continue

        # Stop on anything else, such as the closing JSON quote.
        break

    return tokens


# ------------------------------------------------------------
# Generated-token validity
# ------------------------------------------------------------

def is_valid_token(trimmed_generated_text, previous_true_token, current_true_token):
    """
    A generated token is invalid if it:
    - is empty
    - contains non-ASCII characters
    - contains uppercase letters
      except: if previous true token was ".", then the first letter may be uppercase
    - contains punctuation other than ?, ,, ., ', %
    - contains newlines, quotes, backslashes, spaces, etc.

    Additional target-aware rule:
    - If the current true token is NOT a number and NOT one of ?, ,, ., %,
      then generated candidates containing digits or punctuation marks ?, ,, ., %
      are invalid and do not count toward the adjusted rank.

    Apostrophes are still allowed for word tokens like "don't".
    """
    if trimmed_generated_text == "":
        return False

    # Check the basic character-level validity rules.
    index = 0
    while index < len(trimmed_generated_text):
        ch = trimmed_generated_text[index]

        # Reject non-ASCII characters.
        if ord(ch) >= 128:
            return False

        # Lowercase letters are valid.
        if ch.islower():
            index += 1
            continue

        # Digits are allowed at this basic stage.
        # A later target-aware rule may still reject them.
        if ch.isdigit():
            index += 1
            continue

        # Uppercase letters are only allowed as the first character
        # immediately after a true period token.
        if ch.isupper():
            if previous_true_token == "." and index == 0:
                index += 1
                continue
            return False

        # Allowed punctuation characters.
        if ch in ALLOWED_PUNCTUATION_INSIDE_GENERATED_TOKEN:
            index += 1
            continue

        # Everything else is invalid:
        # spaces, newlines, quotes, backslashes, slashes, colons, brackets, etc.
        return False

    # Determine whether the current true token is a number.
    current_is_number = True
    if current_true_token == "":
        current_is_number = False
    else:
        for ch in current_true_token:
            if not ch.isdigit():
                current_is_number = False
                break

    # Determine whether the current true token is one of the punctuation tokens.
    current_is_punctuation = current_true_token in PUNCTUATION_TOKENS

    # If we are looking for a word token, do not count raw candidates
    # that contain digits or punctuation marks ?, ,, ., %.
    #
    # Apostrophe is not rejected here because words like "don't" need it.
    if not current_is_number and not current_is_punctuation:
        generated_is_number = True
        for ch in trimmed_generated_text:
            if not ch.isdigit():
                generated_is_number = False
                break

        generated_is_punctuation = trimmed_generated_text in PUNCTUATION_TOKENS

        if generated_is_number or generated_is_punctuation:
            return False

    return True


# ------------------------------------------------------------
# Correct-match logic
# ------------------------------------------------------------

def is_correct_match(trimmed_generated_text, current_true_token, next_true_token):
    """
    Decide whether a valid generated token counts as correctly predicting
    the current true token.

    Comparison rules:
    - ignore whitespace on both sides
    - compare case-insensitively
    - if the generated text is less than 4 characters, it usually must match
      the full correct string
    - exception for apostrophes:
      for a token like "don't", "don", "don'", and "don't" are valid,
      but "do", "don'e", and "don'ts" are not
    - if the generated token is longer than the current correct token,
      temporarily append the next token in the sequence and compare against that
    """
    generated = trimmed_generated_text.strip().lower()
    current = current_true_token.lower()

    if next_true_token is None:
        combined = current
    else:
        combined = current + next_true_token.lower()

    # If the generated text spills past the current token,
    # compare against current + next.
    if len(generated) > len(current):
        reference = combined
    else:
        reference = current

    # Generated text cannot be longer than the reference.
    if len(generated) > len(reference):
        return False

    # Special apostrophe handling for cases like "don't".
    apostrophe_index = -1
    i = 0
    while i < len(current):
        if current[i] == "'":
            apostrophe_index = i
            break
        i += 1

    if apostrophe_index != -1:
        # For "don't", apostrophe_index is 3.
        # This allows "don" because it gets all letters before the apostrophe.
        # It rejects "do".
        if len(generated) < 4:
            if len(generated) < apostrophe_index:
                return False
            return current.startswith(generated)

    # Normal short-token behavior:
    # If the generated string is less than 4 chars, it must equal the reference.
    if len(generated) < 4:
        return generated == reference

    # Normal long-token behavior:
    # At least the first 4 chars are correct, and nothing later differs.
    return reference.startswith(generated)


# ------------------------------------------------------------
# Model loading and model input construction
# ------------------------------------------------------------

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model, device


def build_model_inputs(tokenizer, device, visible_text):
    """
    The model tokenizer is still needed to convert the string into model input IDs.

    The target-side tokenization/ranking logic does NOT use the model tokenizer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": visible_text},
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    return model_inputs.to(device)


def get_raw_topk_candidates(model, tokenizer, model_inputs):
    """
    Get the raw top 1000 next-token candidates from the model.
    """
    with torch.no_grad():
        outputs = model(**model_inputs)

    next_token_logits = outputs.logits[0, -1]
    probs = torch.softmax(next_token_logits, dim=-1)

    vocab_size = next_token_logits.shape[0]

    k = RAW_TOPK
    if k > vocab_size:
        k = vocab_size

    top_probs, top_ids = torch.topk(probs, k=k)

    raw_candidates = []
    raw_rank = 1

    while raw_rank <= len(top_ids):
        token_id = top_ids[raw_rank - 1].item()
        probability = top_probs[raw_rank - 1].item()

        # Preserve the tokenizer's decoded spacing for the audit log.
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

        # But use trimmed form for validity and matching.
        trimmed = decoded.strip()

        raw_candidates.append(
            {
                "raw_rank": raw_rank,
                "token_id": token_id,
                "decoded": decoded,
                "trimmed": trimmed,
                "probability": probability,
            }
        )

        raw_rank += 1

    return raw_candidates


def filter_valid_candidates(raw_candidates, previous_true_token, current_true_token):
    """
    Remove invalid generated tokens and assign adjusted ranks to the remaining ones.
    """
    valid_candidates = []
    adjusted_rank = 1

    for candidate in raw_candidates:
        trimmed = candidate["trimmed"]

        if not is_valid_token(trimmed, previous_true_token, current_true_token):
            continue

        valid_candidates.append(
            {
                "adjusted_rank": adjusted_rank,
                "raw_rank": candidate["raw_rank"],
                "token_id": candidate["token_id"],
                "decoded": candidate["decoded"],
                "trimmed": candidate["trimmed"],
                "probability": candidate["probability"],
            }
        )

        adjusted_rank += 1

    return valid_candidates


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():
    if not full_prompt.startswith(prefix):
        raise ValueError("prefix is not an exact prefix of full_prompt")

    tokenizer, model, device = load_model_and_tokenizer()

    # Inline replacement for extract_query_text_after_prefix(...).
    query_text = full_prompt[len(prefix):]

    tokens = tokenize_query_text(query_text)

    if len(tokens) == 0:
        raise ValueError("No tokens were extracted from the query text.")

    # This is the reconstructed context after the prefix.
    # It is always built from the true token sequence, never from model output.
    context = ""

    final_results = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"MODEL_NAME = {MODEL_NAME}\n")
        file_handle.write(f"SYSTEM_PROMPT = {SYSTEM_PROMPT!r}\n")
        file_handle.write(f"RAW_TOPK = {RAW_TOPK}\n")
        file_handle.write("\n")

        file_handle.write("PREFIX:\n")
        file_handle.write(repr(prefix) + "\n")
        file_handle.write("\n")

        file_handle.write("FULL PROMPT:\n")
        file_handle.write(repr(full_prompt) + "\n")
        file_handle.write("\n")

        file_handle.write("TOKENIZED QUERY TOKENS:\n")
        token_index = 0
        while token_index < len(tokens):
            file_handle.write(f"{token_index + 1:3d}. {repr(tokens[token_index])}\n")
            token_index += 1

        file_handle.write("\n")
        file_handle.write("=" * 120 + "\n")
        file_handle.write("\n")

        step_index = 0

        while step_index < len(tokens):
            previous_true_token = None
            if step_index > 0:
                previous_true_token = tokens[step_index - 1]

            current_true_token = tokens[step_index]

            next_true_token = None
            if step_index + 1 < len(tokens):
                next_true_token = tokens[step_index + 1]

            # The visible context fed to the model is always:
            # prefix + true reconstructed context.
            visible_text = prefix + context

            # Required safety check.
            # If this ever fails, the context reconstruction logic is wrong.
            if not full_prompt.startswith(visible_text):
                raise ValueError(
                    "Reconstructed context mismatch.\n"
                    f"prefix + context = {repr(visible_text)}\n"
                    f"full_prompt     = {repr(full_prompt)}"
                )

            model_inputs = build_model_inputs(tokenizer, device, visible_text)
            raw_candidates = get_raw_topk_candidates(model, tokenizer, model_inputs)
            valid_candidates = filter_valid_candidates(
                raw_candidates,
                previous_true_token,
                current_true_token,
            )

            # Inline replacement for find_correct_rank(...).
            correct_rank = None
            matching_candidate = None

            for candidate in valid_candidates:
                if is_correct_match(
                    candidate["trimmed"],
                    current_true_token,
                    next_true_token,
                ):
                    correct_rank = candidate["adjusted_rank"]
                    matching_candidate = candidate
                    break

            final_results.append(
                {
                    "token_index": step_index + 1,
                    "token": current_true_token,
                    "rank": correct_rank,
                }
            )

            # ------------------------------------------------------------
            # Detailed audit log for this step
            # ------------------------------------------------------------

            file_handle.write("-" * 120 + "\n")
            file_handle.write(f"STEP: {step_index + 1}/{len(tokens)}\n")
            file_handle.write(f"PREVIOUS TRUE TOKEN: {repr(previous_true_token)}\n")
            file_handle.write(f"CURRENT TRUE TOKEN:  {repr(current_true_token)}\n")
            file_handle.write(f"NEXT TRUE TOKEN:     {repr(next_true_token)}\n")
            file_handle.write("\n")

            file_handle.write("VISIBLE CONTEXT FED TO MODEL:\n")
            file_handle.write(repr(visible_text) + "\n")
            file_handle.write("\n")

            file_handle.write(f"QUERY STARTSWITH(prefix + context): {full_prompt.startswith(visible_text)}\n")
            file_handle.write("\n")

            if matching_candidate is None:
                file_handle.write("MATCHED RANK: NF\n")
                file_handle.write("MATCHED CANDIDATE: None\n")
            else:
                file_handle.write(f"MATCHED RANK: {correct_rank}\n")
                file_handle.write(
                    "MATCHED CANDIDATE: "
                    f"raw_rank={matching_candidate['raw_rank']}, "
                    f"token_id={matching_candidate['token_id']}, "
                    f"decoded={repr(matching_candidate['decoded'])}, "
                    f"trimmed={repr(matching_candidate['trimmed'])}, "
                    f"prob={matching_candidate['probability']:.8f}\n"
                )

            file_handle.write("\n")

            file_handle.write("RAW TOP-1000 CANDIDATES:\n")
            raw_index = 0
            while raw_index < len(raw_candidates):
                candidate = raw_candidates[raw_index]

                valid_flag = is_valid_token(
                    candidate["trimmed"],
                    previous_true_token,
                    current_true_token,
                )

                file_handle.write(
                    f"{candidate['raw_rank']:4d}. "
                    f"token_id={candidate['token_id']:<8} "
                    f"decoded={repr(candidate['decoded'])} "
                    f"trimmed={repr(candidate['trimmed'])} "
                    f"valid={valid_flag} "
                    f"prob={candidate['probability']:.8f}\n"
                )

                raw_index += 1

            file_handle.write("\n")

            file_handle.write("VALID CANDIDATES WITH ADJUSTED RANKS:\n")
            valid_index = 0
            while valid_index < len(valid_candidates):
                candidate = valid_candidates[valid_index]

                is_match = is_correct_match(
                    candidate["trimmed"],
                    current_true_token,
                    next_true_token,
                )

                file_handle.write(
                    f"{candidate['adjusted_rank']:4d}. "
                    f"(raw {candidate['raw_rank']:4d}) "
                    f"token_id={candidate['token_id']:<8} "
                    f"decoded={repr(candidate['decoded'])} "
                    f"trimmed={repr(candidate['trimmed'])} "
                    f"match={is_match} "
                    f"prob={candidate['probability']:.8f}\n"
                )

                # Requested change:
                # Once the matching candidate is printed, stop printing the adjusted ranking list.
                if is_match:
                    break

                valid_index += 1

            file_handle.write("\n")

            file_handle.write("CONTEXT BEFORE APPENDING TRUE TOKEN:\n")
            file_handle.write(repr(context) + "\n")

            # Append ONLY the true next token from the target sequence.
            # Never use the model's generated text to update context.
            context = context + current_true_token

            # Inline replacement for append_true_token_to_context(...).
            if next_true_token is not None:
                if next_true_token not in PUNCTUATION_TOKENS:
                    context = context + " "

            file_handle.write("CONTEXT AFTER APPENDING TRUE TOKEN:\n")
            file_handle.write(repr(context) + "\n")
            file_handle.write("\n")

            # Safety check after context update.
            if not full_prompt.startswith(prefix + context):
                raise ValueError(
                    "After appending the true token, the reconstructed context no longer matches the prompt.\n"
                    f"prefix + context = {repr(prefix + context)}\n"
                    f"full_prompt     = {repr(full_prompt)}"
                )

            step_index += 1

        # ------------------------------------------------------------
        # Final simplified output
        # ------------------------------------------------------------

        file_handle.write("#" * 120 + "\n")
        file_handle.write("FINAL SIMPLIFIED TOKEN -> RANK LIST\n")
        file_handle.write("#" * 120 + "\n")

        result_index = 0
        while result_index < len(final_results):
            item = final_results[result_index]

            if item["rank"] is None:
                rank_text = "NF"
            else:
                rank_text = str(item["rank"])

            file_handle.write(f"{repr(item['token'])}: {rank_text}\n")

            result_index += 1

    print(f"Saved full audit output to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()