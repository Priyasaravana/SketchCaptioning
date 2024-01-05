def name_exp(experiment, task):
    ex_key = experiment.get_key()
    exp_name = f"{task}: {ex_key[:9]}"
    return exp_name


def format_attention_output(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError(
                "The attention tensor does not have the correct number of"
                "dimensions. Make sure you set "
                "output_attentions=True when initializing your model."
            )
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def format_special_chars(tokens):
    return [
        t.replace("Ġ", " ")
        .replace("▁", " ")
        .replace("</w>", "")
        .replace("ĊĊ", " ")
        .replace("Ċ", " ")
        for t in tokens
    ]


def get_attn_data(
    model,
    tokenizer,
    text_a,
    text_b=None,
    return_token_type_ids=False,
    prettify_tokens=True,
    layer=None,
    heads=None,
):
    if return_token_type_ids:
        inputs = tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        token_type_ids = inputs["token_type_ids"]

    else:
        inputs = tokenizer.encode_plus(
            text_a, return_tensors="pt", add_special_tokens=True
        )

    input_ids = inputs["input_ids"]
    attention = model(input_ids)[-1]
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)

    if prettify_tokens:
        tokens = format_special_chars(tokens)

    attn = format_attention_output(attention)
    attn_data = {
        "all": {"attn": attn.tolist(), "left_text": tokens, "right_text": tokens}
    }

    if text_b is not None:
        text_b_start = token_type_ids[0].tolist().index(1)

        slice_a = slice(
            0, text_b_start
        )  # Positions corresponding to sentence A in input
        slice_b = slice(
            text_b_start, len(tokens)
        )  # Position corresponding to sentence B in input
        attn_data["aa"] = {
            "attn": attn[:, :, slice_a, slice_a].tolist(),
            "left_text": tokens[slice_a],
            "right_text": tokens[slice_a],
        }
        attn_data["bb"] = {
            "attn": attn[:, :, slice_b, slice_b].tolist(),
            "left_text": tokens[slice_b],
            "right_text": tokens[slice_b],
        }
        attn_data["ab"] = {
            "attn": attn[:, :, slice_a, slice_b].tolist(),
            "left_text": tokens[slice_a],
            "right_text": tokens[slice_b],
        }
        attn_data["ba"] = {
            "attn": attn[:, :, slice_b, slice_a].tolist(),
            "left_text": tokens[slice_b],
            "right_text": tokens[slice_a],
        }

    attn_seq_len = len(attn_data["all"]["attn"][0][0])
    if attn_seq_len != len(tokens):
        raise ValueError(
            f"Attention has {attn_seq_len} positions, while number of tokens is {len(tokens)}"
        )

    return attn_data


def text_generation_viz(prompts, model_version):
    task = "text-gen"
    experiment = comet_ml.Experiment()
    experiment.set_name(name_exp(experiment, task))
    experiment.add_tag(task)
    experiment.log_parameter("model_version", model_version)

    do_lower_case = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_version, do_lower_case=do_lower_case
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_version, output_attentions=True, pad_token_id=tokenizer.eos_token_id
    )

    text_generation = pipeline("text-generation", model=model, tokenizer=tokenizer)
    for prompt in tqdm(prompts):
        generated_text = text_generation(prompt, max_length=10, do_sample=False)[0][
            "generated_text"
        ]
        attn_data = get_attn_data(model, tokenizer, generated_text)

        viz_params = {
            "attention": attn_data,
            "default_filter": "all",
            "bidirectional": False,
            "display_mode": "light",
            "layer": None,
            "head": None,
        }
        experiment.log_asset_data(viz_params, f"attn-view-tg-{prompt}.json")

    experiment.end()


def sentiment_viz(prompts, model_version):
    task = "sentiment"
    experiment = comet_ml.Experiment()
    experiment.set_name(name_exp(experiment, task))
    experiment.add_tag(task)
    experiment.log_parameter("model_version", model_version)

    do_lower_case = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_version, do_lower_case=do_lower_case
    )
    tokenizer.pad_token = "[PAD]"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_version, output_attentions=True, pad_token_id=tokenizer.eos_token_id
    )

    sentiment_analysis = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )
    for prompt in tqdm(prompts):
        prediction = sentiment_analysis(prompt)[0]
        attn_data = get_attn_data(model, tokenizer, prompt)
        viz_params = {
            "attention": attn_data,
            "default_filter": "all",
            "bidirectional": False,
            "display_mode": "light",
            "layer": None,
            "head": None,
        }
        experiment.log_asset_data(
            viz_params,
            f"attn-view-sentiment-{prompt}-{prediction['label']}-{round(prediction['score'], 2)}.json",
        )

    experiment.end()


def qa_viz(context, prompts, model_version):
    task = "question-answering"
    experiment = comet_ml.Experiment()
    experiment.set_name(name_exp(experiment, task))
    experiment.add_tag(task)
    experiment.log_parameter("model_version", model_version)
    experiment.log_text(str(context))

    do_lower_case = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_version, do_lower_case=do_lower_case
    )
    tokenizer.pad_token = "[PAD]"
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_version, output_attentions=True, pad_token_id=tokenizer.eos_token_id
    )

    qa = pipeline("question-answering", model=model, tokenizer=tokenizer)
    for prompt in tqdm(prompts):
        prediction = qa(question=prompt, context=context)
        answer = prediction["answer"]

        start = prediction["start"]
        end = prediction["end"]

        attn_data = get_attn_data(
            model, tokenizer, prompt, text_b=answer, return_token_type_ids=True
        )
        viz_params = {
            "attention": attn_data,
            "default_filter": "all",
            "bidirectional": False,
            "display_mode": "light",
            "layer": None,
            "head": None,
        }
        experiment.log_asset_data(
            viz_params,
            f"attn-view-qa-{prompt}-score-{round(prediction['score'], 3)}-start-{start}-end-{end}.json",
        )

    experiment.end()