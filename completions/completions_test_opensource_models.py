import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os

# ── Model selection ────────────────────────────────────────────────────────────
# Set MODEL env var when submitting, e.g.:
#   MODEL=gptoss   sbatch run_completions_gptoss.sh
#   MODEL=olmo32b  sbatch run_completions_olmo32b.sh
#   MODEL=qwen3b   sbatch run_completions_qwen3b.sh
# Falls back to gptoss if not set.

# (model_id, output_file, use_chat_template)
# use_chat_template=False feeds the raw prefix text directly — needed for base
# models that don't follow chat formatting (e.g. GPT-OSS-20B).
_MODEL_CONFIGS = {
    "gptoss":  ("openai/gpt-oss-20b",            "completions/outputs/gptoss20b_full_audit.txt",   False),
    "olmo32b": ("allenai/OLMo-3.1-32B-Instruct", "completions/outputs/olmo32b_base_full_audit.txt", True),
    "qwen3b":  ("Qwen/Qwen2.5-3B-Instruct",      "completions/outputs/qwen3b_base_full_audit.txt",  True),
}

_model_key = os.environ.get("MODEL", "gptoss")
if _model_key not in _MODEL_CONFIGS:
    raise ValueError(f"Unknown MODEL={_model_key!r}. Choose from: {list(_MODEL_CONFIGS)}")

MODEL_NAME, OUTPUT_FILE, USE_CHAT_TEMPLATE = _MODEL_CONFIGS[_model_key]
print(f"Running completions audit for: {MODEL_NAME}  →  {OUTPUT_FILE}")

def load_model_and_tokenizer():
    import os
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",   # spreads across available GPUs; handles 32B on 80GB A100
        token=hf_token,
    )
    model.eval()

    device = next(model.parameters()).device
    return tokenizer, model, device

# PM: I apologize in advance for this mess of code, I was trying to get it ready to test as fast as possible. You shouldn't need to modify anything under this line.



RAW_TOPK = 1000
SYSTEM_PROMPT = "Continue the text."
prefix = "[{\"question_id\": 19, \"desc_id\": \"alarm-mediation-nie-model1-spec1-q1\", \"given_info\": \""
full_prompt = "[{\"question_id\": 19, \"desc_id\": \"alarm-mediation-nie-model1-spec1-q1\", \"given_info\": \"For husbands that don't set the alarm and wives that don't set the alarm, the probability of ringing alarm is 6%. For husbands that don't set the alarm and wives that set the alarm, the probability of ringing alarm is 51%. For husbands that set the alarm and wives that don't set the alarm, the probability of ringing alarm is 47%. For husbands that set the alarm and wives that set the alarm, the probability of ringing alarm is 92%. For husbands that don't set the alarm, the probability of alarm set by wife is 45%. For husbands that set the alarm, the probability of alarm set by wife is 63%. Does husband negatively affect alarm clock through wife?\""


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------

def write_line(file_handle, text=""):
    file_handle.write(text + "\n")


def is_ascii_character(ch):
    return ord(ch) < 128


def is_word_character(ch):
    # A "word" token can contain letters and apostrophes.
    # Example: don't
    return ch.isalpha() or ch == "'"


def is_number_character(ch):
    return ch.isdigit()


def is_punctuation_token(token_text):
    return token_text in ["?", ",", ".", "%"]


def decode_model_token(tokenizer, token_id):
    # We keep the tokenizer's original spacing behavior here because
    # we want to see exactly what the model token decodes to.
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def trim_whitespace(text):
    return text.strip()


# ------------------------------------------------------------
# Custom tokenization of the target query
# ------------------------------------------------------------

def extract_query_text_after_prefix(full_text, prefix_text):
    """
    We only tokenize the natural-language part after the prefix.

    The full string is JSON-like, so after the natural-language query there is
    a closing quote. We stop tokenization when we hit something that is not:
    - whitespace
    - a word character
    - a digit
    - one of ?, . , %

    That way we do not try to treat the closing JSON quote as a target token.
    """
    if not full_text.startswith(prefix_text):
        raise ValueError("prefix is not an exact prefix of full_prompt")

    remaining = full_text[len(prefix_text):]
    return remaining


def tokenize_query_text(query_text):
    """
    Custom tokenizer, exactly as requested.

    It splits into:
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

        # Skip spaces between tokens.
        if ch.isspace():
            i += 1
            continue

        # Number token
        if is_number_character(ch):
            start = i
            while i < len(query_text) and is_number_character(query_text[i]):
                i += 1
            tokens.append(query_text[start:i])
            continue

        # Punctuation token
        if ch in ["?", ",", ".", "%"]:
            tokens.append(ch)
            i += 1
            continue

        # Word token
        if is_word_character(ch):
            start = i
            while i < len(query_text) and is_word_character(query_text[i]):
                i += 1
            tokens.append(query_text[start:i])
            continue

        # If we hit something else, we stop.
        # In this prompt that should be the closing quote after the natural language.
        break

    return tokens


# ------------------------------------------------------------
# Reconstructing the context exactly by the user's rule
# ------------------------------------------------------------

def append_true_token_to_context(existing_context, this_token, next_token):
    """
    Reconstruct the context ONLY from the true next token sequence, never from the model.

    Rule:
    if nexttoken == punctuation:
        context += thistoken
    else:
        context += thistoken + space

    For the final token there is no future query, so we do not force a trailing
    space there. That keeps the final reconstructed context readable.
    """
    new_context = existing_context + this_token

    if next_token is None:
        return new_context

    if is_punctuation_token(next_token):
        return new_context

    return new_context + " "


# ------------------------------------------------------------
# Validity check for generated candidate tokens
# ------------------------------------------------------------

def is_valid_generated_token(trimmed_generated_text, previous_true_token):
    """
    A generated token is invalid if it:
    - is empty
    - contains non-ASCII characters
    - contains uppercase letters
      (except: if previous true token was ".", then the first letter may be uppercase)
    - contains punctuation other than ?, ,, ., ', %
    - contains quotes, backslashes, newlines, spaces, or any other disallowed symbol

    IMPORTANT:
    This function works on the whitespace-trimmed decoded model token.
    """
    if trimmed_generated_text == "":
        return False

    index = 0
    while index < len(trimmed_generated_text):
        ch = trimmed_generated_text[index]

        # Reject non-ASCII immediately.
        if not is_ascii_character(ch):
            return False

        # Lowercase letters are always fine.
        if ch.islower():
            index += 1
            continue

        # Digits are always fine.
        if ch.isdigit():
            index += 1
            continue

        # Uppercase letters are only allowed in one specific case:
        # the first character may be uppercase if the previous true token was "."
        if ch.isupper():
            if previous_true_token == "." and index == 0:
                index += 1
                continue
            return False

        # Allowed punctuation
        if ch in ["?", ",", ".", "'", "%"]:
            index += 1
            continue

        # Everything else is invalid:
        # spaces, tabs, newlines, quotes, backslashes, slashes, colons, brackets, etc.
        return False

    return True


# ------------------------------------------------------------
# Matching logic for "is this generated token considered correct?"
# ------------------------------------------------------------

def generated_token_matches_target(trimmed_generated_text, current_true_token, next_true_token):
    """
    Matching rule, exactly as requested.

    Comparison rules:
    - trim whitespace on both sides first
    - compare case-insensitively
    - if generated text length is < 4, compare it to the FULL correct string
    - if generated text length is >= 4, the first 4 chars must be correct and
      no later part of the generated token may differ
    - if the model generates a token longer than the current correct token,
      temporarily append the next token in the sequence and compare against that

    Examples that this function is intended to satisfy:
    "50", "50" = true
    "5", "50" = false
    "500", "50" = false
    "husband", "Husbands" = true
    "husbands", "husband" = false
    "se", "set" = false
    "set", "set" = true
    "set", "sets" = false
    "sets", "set" + "the" = false
    "alarm,", "alarm" + "," = true
    "alarm", "alarm" = true
    ". ", "." = true
    "50%", "50" + "%" = true
    "%.", "%" + "." = true
    "don't", "don't" = true
    "don", "don't" = false
    "don'", "don't" = true
    """
    generated = trim_whitespace(trimmed_generated_text).lower()
    current = current_true_token.lower()

    if next_true_token is None:
        combined = current
    else:
        combined = current + next_true_token.lower()

    # If the generated text is longer than the current token,
    # allow it to spill into the next token by comparing against combined.
    if len(generated) > len(current):
        reference = combined
    else:
        reference = current

    # If generated text is shorter than 4 chars, it must match the full reference exactly.
    if len(generated) < 4:
        return generated == reference

    # Otherwise:
    # - the generated text must not exceed the reference
    # - the reference must start with the generated text
    if len(generated) > len(reference):
        return False

    return reference.startswith(generated)


# ------------------------------------------------------------
# Self-tests for the matching rule and validity rule
# ------------------------------------------------------------

def run_self_tests():
    """
    The user explicitly asked that these checks pass before returning the final program.
    We enforce that here with assertions.
    """

    # --------------------
    # Matching tests
    # --------------------
    assert generated_token_matches_target("50", "50", None) is True
    assert generated_token_matches_target("5", "50", None) is False
    assert generated_token_matches_target("500", "50", None) is False

    assert generated_token_matches_target("husband", "Husbands", None) is True
    assert generated_token_matches_target("husbands", "husband", None) is False

    assert generated_token_matches_target("se", "set", None) is False
    assert generated_token_matches_target("set", "set", None) is True
    assert generated_token_matches_target("set", "sets", None) is False
    assert generated_token_matches_target("sets", "set", "the") is False

    assert generated_token_matches_target("alarm,", "alarm", ",") is True
    assert generated_token_matches_target("alarm", "alarm", None) is True

    assert generated_token_matches_target(". ", ".", None) is True

    assert generated_token_matches_target("50%", "50", "%") is True
    assert generated_token_matches_target("%.", "%", ".") is True

    assert generated_token_matches_target("don't", "don't", None) is True
    assert generated_token_matches_target("don", "don't", None) is False
    assert generated_token_matches_target("don'", "don't", None) is True

    # This one depends on both validity and matching.
    # Previous true token is ".", so uppercase first letter is allowed.
    assert is_valid_generated_token("For", ".") is True
    assert generated_token_matches_target("For", "For", None) is True

    # --------------------
    # Validity tests
    # --------------------
    assert is_valid_generated_token("", None) is False
    assert is_valid_generated_token("hello", None) is True
    assert is_valid_generated_token("Hello", None) is False
    assert is_valid_generated_token("Hello", ".") is True
    assert is_valid_generated_token("HELLO", ".") is False
    assert is_valid_generated_token("abc?", None) is True
    assert is_valid_generated_token("abc,", None) is True
    assert is_valid_generated_token("abc.", None) is True
    assert is_valid_generated_token("abc%", None) is True
    assert is_valid_generated_token("abc'", None) is True
    assert is_valid_generated_token("abc\"", None) is False
    assert is_valid_generated_token("abc\\", None) is False
    assert is_valid_generated_token("abc/", None) is False
    assert is_valid_generated_token("abc:", None) is False
    assert is_valid_generated_token("abc(", None) is False
    assert is_valid_generated_token("abc ", None) is False
    assert is_valid_generated_token("a b", None) is False
    assert is_valid_generated_token("naïve", None) is False


# ------------------------------------------------------------
# Model loading and next-token distribution
# ------------------------------------------------------------




def build_model_inputs(tokenizer, device, visible_text):
    """
    We still need the model tokenizer to turn the string into model input IDs,
    but the TARGET tokenization/ranking logic now uses the custom rules above,
    not the model tokenizer.
    """
    if USE_CHAT_TEMPLATE:
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
    else:
        # Raw completion mode: feed the text directly with no chat wrapping.
        # Needed for base models like GPT-OSS-20B that don't use chat templates.
        model_inputs = tokenizer(visible_text, return_tensors="pt")

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
        decoded = decode_model_token(tokenizer, token_id)
        trimmed = trim_whitespace(decoded)

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


def filter_valid_candidates(raw_candidates, previous_true_token):
    """
    Remove invalid generated tokens and assign adjusted ranks to the remaining ones.
    """
    valid_candidates = []
    adjusted_rank = 1

    for candidate in raw_candidates:
        trimmed = candidate["trimmed"]

        if not is_valid_generated_token(trimmed, previous_true_token):
            continue

        new_entry = {
            "adjusted_rank": adjusted_rank,
            "raw_rank": candidate["raw_rank"],
            "token_id": candidate["token_id"],
            "decoded": candidate["decoded"],
            "trimmed": candidate["trimmed"],
            "probability": candidate["probability"],
        }

        valid_candidates.append(new_entry)
        adjusted_rank += 1

    return valid_candidates


def find_correct_rank(valid_candidates, current_true_token, next_true_token):
    """
    Find the first valid candidate that counts as a correct prediction.
    """
    for candidate in valid_candidates:
        trimmed = candidate["trimmed"]

        if generated_token_matches_target(trimmed, current_true_token, next_true_token):
            return candidate["adjusted_rank"], candidate

    return None, None


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():
    # First, make sure the matching logic and validity logic behave as requested.
    run_self_tests()

    tokenizer, model, device = load_model_and_tokenizer()

    query_text = extract_query_text_after_prefix(full_prompt, prefix)
    tokens = tokenize_query_text(query_text)

    if len(tokens) == 0:
        raise ValueError("No tokens were extracted from the query text.")

    # This is the reconstructed context after the prefix.
    # It starts empty because we begin querying on the first token after the prefix.
    context = ""

    # We store one final simplified result per target token.
    final_results = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file_handle:
        write_line(file_handle, f"MODEL_NAME = {MODEL_NAME}")
        write_line(file_handle, f"SYSTEM_PROMPT = {SYSTEM_PROMPT!r}")
        write_line(file_handle, f"RAW_TOPK = {RAW_TOPK}")
        write_line(file_handle)
        write_line(file_handle, "PREFIX:")
        write_line(file_handle, repr(prefix))
        write_line(file_handle)
        write_line(file_handle, "FULL PROMPT:")
        write_line(file_handle, repr(full_prompt))
        write_line(file_handle)
        write_line(file_handle, "TOKENIZED QUERY TOKENS:")
        token_index = 0
        while token_index < len(tokens):
            write_line(file_handle, f"{token_index + 1:3d}. {repr(tokens[token_index])}")
            token_index += 1
        write_line(file_handle)
        write_line(file_handle, "=" * 120)
        write_line(file_handle)

        step_index = 0

        while step_index < len(tokens):
            previous_true_token = None
            if step_index > 0:
                previous_true_token = tokens[step_index - 1]

            current_true_token = tokens[step_index]

            next_true_token = None
            if step_index + 1 < len(tokens):
                next_true_token = tokens[step_index + 1]

            # Reconstruct exactly as requested:
            # prefix + context
            visible_text = prefix + context

            # Safety check requested by the user.
            if not full_prompt.startswith(visible_text):
                raise ValueError(
                    "Reconstructed context mismatch.\n"
                    f"prefix + context = {repr(visible_text)}\n"
                    f"full_prompt     = {repr(full_prompt)}"
                )

            model_inputs = build_model_inputs(tokenizer, device, visible_text)
            raw_candidates = get_raw_topk_candidates(model, tokenizer, model_inputs)
            valid_candidates = filter_valid_candidates(raw_candidates, previous_true_token)
            correct_rank, matching_candidate = find_correct_rank(
                valid_candidates,
                current_true_token,
                next_true_token,
            )

            # Save the simplified final result for this token.
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
            write_line(file_handle, "-" * 120)
            write_line(file_handle, f"STEP: {step_index + 1}/{len(tokens)}")
            write_line(file_handle, f"PREVIOUS TRUE TOKEN: {repr(previous_true_token)}")
            write_line(file_handle, f"CURRENT TRUE TOKEN:  {repr(current_true_token)}")
            write_line(file_handle, f"NEXT TRUE TOKEN:     {repr(next_true_token)}")
            write_line(file_handle)
            write_line(file_handle, "VISIBLE CONTEXT FED TO MODEL:")
            write_line(file_handle, repr(visible_text))
            write_line(file_handle)
            write_line(file_handle, f"QUERY STARTSWITH(prefix + context): {full_prompt.startswith(visible_text)}")
            write_line(file_handle)

            if matching_candidate is None:
                write_line(file_handle, "MATCHED RANK: NF")
                write_line(file_handle, "MATCHED CANDIDATE: None")
            else:
                write_line(file_handle, f"MATCHED RANK: {correct_rank}")
                write_line(
                    file_handle,
                    "MATCHED CANDIDATE: "
                    f"raw_rank={matching_candidate['raw_rank']}, "
                    f"token_id={matching_candidate['token_id']}, "
                    f"decoded={repr(matching_candidate['decoded'])}, "
                    f"trimmed={repr(matching_candidate['trimmed'])}, "
                    f"prob={matching_candidate['probability']:.8f}"
                )

            write_line(file_handle)
            write_line(file_handle, "RAW TOP-1000 CANDIDATES:")
            raw_index = 0
            while raw_index < len(raw_candidates):
                candidate = raw_candidates[raw_index]
                valid_flag = is_valid_generated_token(candidate["trimmed"], previous_true_token)
                write_line(
                    file_handle,
                    f"{candidate['raw_rank']:4d}. "
                    f"token_id={candidate['token_id']:<8} "
                    f"decoded={repr(candidate['decoded'])} "
                    f"trimmed={repr(candidate['trimmed'])} "
                    f"valid={valid_flag} "
                    f"prob={candidate['probability']:.8f}"
                )
                raw_index += 1

            write_line(file_handle)
            write_line(file_handle, "VALID CANDIDATES WITH ADJUSTED RANKS:")
            valid_index = 0
            while valid_index < len(valid_candidates):
                candidate = valid_candidates[valid_index]
                is_match = generated_token_matches_target(
                    candidate["trimmed"],
                    current_true_token,
                    next_true_token,
                )
                write_line(
                    file_handle,
                    f"{candidate['adjusted_rank']:4d}. "
                    f"(raw {candidate['raw_rank']:4d}) "
                    f"token_id={candidate['token_id']:<8} "
                    f"decoded={repr(candidate['decoded'])} "
                    f"trimmed={repr(candidate['trimmed'])} "
                    f"match={is_match} "
                    f"prob={candidate['probability']:.8f}"
                )
                valid_index += 1

            write_line(file_handle)
            write_line(file_handle, "CONTEXT BEFORE APPENDING TRUE TOKEN:")
            write_line(file_handle, repr(context))

            # IMPORTANT:
            # We append ONLY the true next token from the sequence,
            # never anything generated by the model.
            context = append_true_token_to_context(context, current_true_token, next_true_token)

            write_line(file_handle, "CONTEXT AFTER APPENDING TRUE TOKEN:")
            write_line(file_handle, repr(context))
            write_line(file_handle)

            # Optional sanity check after updating context as well.
            # This should still be a valid prefix of the full prompt.
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
        write_line(file_handle, "#" * 120)
        write_line(file_handle, "FINAL SIMPLIFIED TOKEN -> RANK LIST")
        write_line(file_handle, "#" * 120)

        result_index = 0
        while result_index < len(final_results):
            item = final_results[result_index]
            if item["rank"] is None:
                rank_text = "NF"
            else:
                rank_text = str(item["rank"])

            write_line(
                file_handle,
                f"{repr(item['token'])}: {rank_text}"
            )
            result_index += 1

    print(f"Saved full audit output to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()