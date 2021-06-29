import torch


def predict_factuality(model, tokenizer, conditioning_model, input_text, precondition_topk=200, do_sample=False,
                       min_length=30, max_length=90, condition_lambda=1.0, device='cuda', repetition_penalty=1.2,
                       no_repeat_ngram_size=2):
    with torch.no_grad():
        batch_size = len(input_text)

        # assumes initially all same length.
        encoder_input_ids = [tokenizer.encode(it, return_tensors='pt').to(device) for it in input_text]  # batch x seq
        encoder_input_ids = torch.cat(encoder_input_ids, dim=0)

        input_ids= torch.LongTensor([[tokenizer.bos_token_id]]).to(device)

        cur_len = 1
        temperature = 1.0
        top_k = 50
        top_p = 1.0
        pad_token_id = tokenizer.pad_token_id
        bad_words_ids = [[pad_token_id]]
        eos_token_id = tokenizer.eos_token_id
        attention_mask = encoder_input_ids.new_ones(encoder_input_ids.shape)
        use_cache = True

        encoder_outputs = model.get_encoder()(encoder_input_ids, attention_mask=attention_mask, output_attentions=True,
                                              output_hidden_states=True, return_dict=True)
        model_specific_kwargs = {'encoder_outputs': encoder_outputs}


        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
        eos_tensor = torch.zeros_like(encoder_input_ids[:,0:1]).unsqueeze(1).fill_(eos_token_id).expand(-1, precondition_topk, -1).int()

        past = None
        while cur_len < max_length:
            model_inputs = model.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )

            outputs = model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            scores = model.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            top_logits, top_indices = scores.topk(precondition_topk, dim=1)  # batch x topk

            # TODO: Yiping - I added this line. need to verify whether it's correct
            # https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation/issues/1
            top_logits = torch.softmax(top_logits, dim=1)

            if condition_lambda == 0:
                condition_logits = torch.zeros_like(top_logits).float()
            else:
                # Prepare the input to the entailment model [CLS] document [SEP] summary [SEP]
                tplus1_candidates = torch.cat(
                    [encoder_input_ids.unsqueeze(1).expand(-1, precondition_topk, -1),
                     input_ids.unsqueeze(1).expand(-1, precondition_topk, -1)[:, :,1:],
                     top_indices.unsqueeze(2),
                     eos_tensor], dim=2) # batch x topk x seq+1, with pad dropped

                # TODO: prepare the model input data and pass as model_input arg
                seq_len = tplus1_candidates.shape[2]
                condition_attention_mask = torch.ones(precondition_topk, seq_len, dtype=torch.long).to(device)
                token_type_ids = torch.zeros(precondition_topk, seq_len, dtype=torch.long).to(device)
                token_type_ids[:, encoder_input_ids.shape[1]:] = 1
                condition_model_input = {'input_ids': tplus1_candidates.flatten(0, 1),
                                         'token_type_ids': token_type_ids, 'attention_mask': condition_attention_mask}


                condition_logits = conditioning_model(**condition_model_input)[0]  # batch*topk x seq+1
                # the second column corresponds to the label "entail". Can be different for different fine-tuned models
                condition_logits = torch.log_softmax(condition_logits, dim=1)[:, 1]
                condition_logits = condition_logits.view(batch_size, precondition_topk) # batch x topk of last formality pred

            #pdb.set_trace()
            full_logits = top_logits + condition_lambda * condition_logits
            if do_sample:
                raise NotImplementedError
            else:
                # Greedy decoding
                next_token = top_indices[torch.arange(batch_size).to(top_indices.device), torch.argmax(full_logits, dim=-1)]

            # if do_sample:
            #     # Temperature (higher temperature => more likely to sample low probability tokens)
            #     if temperature != 1.0:
            #         scores = scores / temperature
            #     # Top-p/top-k filtering
            #     next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            #     # Sample
            #     probs = F.softmax(next_token_logscores, dim=-1)
            #     next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            # else:
            #     # Greedy decoding
            #     next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if model.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return [tokenizer.decode(s[1:-1]) for s in input_ids.tolist()]  # 1: to delete the pad token
