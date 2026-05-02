# --- 2.2a Sequential Scaling Implementation ---

def _encode(text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)

# Ensure these are defined for the specific model
THINK_CLOSE_TOKEN_IDS = _encode(THINK_CLOSE) 
WAIT_TOKEN_IDS        = _encode(" Wait")        

def budget_force_generate(
    problem: str,
    budget: int,
    strategy: str = "stop",          # "stop" or "wait"
    max_answer_tokens: int = SEQ_MAX_ANSWER_TOKENS,
) -> dict:
    """
    Generate one completion with a thinking-token budget using KV caching.
    Includes low-temperature sampling to prevent repetitive looping (flatlining).
    """
    prompt = build_prompt(problem, enable_thinking=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated     = []   
    thinking_count = 0   
    in_thinking   = True 
    forced_close  = False

    # Initialize KV Cache
    current_input = input_ids
    past_key_values = None

    with torch.no_grad():
        for _ in range(budget + max_answer_tokens + 128):  
            outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
            logits  = outputs.logits[0, -1, :]  
            past_key_values = outputs.past_key_values 

            # ── Intervention logic [cite: 104, 105] ──
            if in_thinking:
                if strategy == "stop" and thinking_count >= budget and not forced_close:
                    # Force-close: next token is </think> [cite: 104]
                    next_token_id = THINK_CLOSE_TOKEN_IDS[0]
                    forced_close  = True
                elif strategy == "wait" and thinking_count < budget:
                    # Suppress </think> tags to force more thinking [cite: 105]
                    for tid in THINK_CLOSE_TOKEN_IDS:
                        logits[tid] = float("-inf")
                    
                    # Pseudo-greedy sampling to prevent looping collapse
                    probs = torch.softmax(logits / 0.1, dim=-1) 
                    next_token_id = int(torch.multinomial(probs, num_samples=1))
                else:
                    probs = torch.softmax(logits / 0.1, dim=-1)
                    next_token_id = int(torch.multinomial(probs, num_samples=1))
            else:
                probs = torch.softmax(logits / 0.1, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1))

            generated.append(next_token_id)

            # Track thinking boundary using Token IDs
            if in_thinking:
                if next_token_id in THINK_CLOSE_TOKEN_IDS:
                    in_thinking = False   
                else:
                    thinking_count += 1

            # Exit conditions
            if next_token_id == tokenizer.eos_token_id:
                break
            if not in_thinking and (len(generated) - thinking_count) >= max_answer_tokens:
                break

            # Update input for next step (only the last token)
            current_input = torch.tensor([[next_token_id]], device=device)

    # skip_special_tokens=False is CRITICAL for Qualitative Analysis
    output = tokenizer.decode(generated, skip_special_tokens=False)
    
    return {
        "output":          output,
        "thinking_tokens": thinking_count,
        "total_tokens":    len(generated),
    }

# ── Run sequential scaling ────────────────────────────────────────────────────
SEQ_STRATEGY = "stop"   
seq_results = {}   

for budget in SEQ_BUDGETS:
    print(f"\n=== Sequential budget={budget} ({SEQ_STRATEGY}) ===")
    records = []
    for i, (problem, gold) in enumerate(zip(PROBLEMS, GOLD_ANSWERS)):
        res = budget_force_generate(problem, budget=budget, strategy=SEQ_STRATEGY)
        
        exact = extract_answer(res["output"], mode="exact_match")
        flex  = extract_answer(res["output"], mode="flexible_extract")
        
        rec = {
            "problem":         problem,
            "gold":            gold,
            "output":          res["output"],
            "exact_ans":       exact,
            "flex_ans":        flex,
            "exact_correct":   grade(exact, gold),
            "flex_correct":    grade(flex,  gold),
            "thinking_tokens": res["thinking_tokens"],
            "total_tokens":    res["total_tokens"],
        }
        records.append(rec)
        print(f"  [{i+1:2d}/30] gold={gold:3d} exact={exact} flex={flex} "
              f"think={res['thinking_tokens']} total={res['total_tokens']}")
    
    seq_results[budget] = records
    torch.cuda.empty_cache() # Clear VRAM after each budget sweep

# Save for the writeup
with open("results_2_2_sequential.json", "w") as f:
    import json
    json.dump(seq_results, f)
print("\nSaved results_2_2_sequential.json")