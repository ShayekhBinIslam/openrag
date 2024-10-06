#!/usr/bin/python
# -*- coding: UTF-8 -*-

from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import torch
import numpy as np
from tqdm import tqdm
import json
import argparse
from tqdm import tqdm
from openrag.utils import (
    PROMPT_DICT,
    TASK_INST,
    load_jsonlines,
    control_tokens,
    load_special_tokens,
)
from openrag.metrics import match, accuracy

seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer

from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer, stops = [], encounters=1, device="cuda"):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = [stop.to(device) for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if stop == last_token:
                return True
        return False


def call_model_rerank_w_scores_batch(
    stopping_criteria,
    tokenizer,
    prompt,
    evidences,
    model,
    max_new_tokens=15,
    ret_tokens=None,
    rel_tokens=None,
    grd_tokens=None,
    ut_tokens=None,
    use_seqscore=False,
    threshold=0.5,
    w_rel=1.0,
    w_sup=1.0,
    w_use=0.5,
    mode="adaptive_retrieval",
    closed=False,
):
    
    results = {}
    detailed_init_score = {"logproba_retrieval_thresh": 0, "proba_retrieval_thresh": 0,
        "pred_retrieval_decision": "", "do_retrieve": None, "proba_r": "", "logr": "",
        "lognr": "", "proba_nr": "", "seq_logprob": ""}
    if mode != "always_retrieve":
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda") # size 22
        preds = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            top_p=1.0,
            stopping_criteria=stopping_criteria,
        )
        
        pred_text = tokenizer.batch_decode(
            preds.sequences[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True, 
        )[0]
        pred = tokenizer.batch_decode(
            preds.sequences[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=False, 
        )[0]
        pred_token_ids = preds.sequences[:, inputs.input_ids.shape[1]:].squeeze(dim=0).tolist()
        pred_log_probs = []
        for score in preds.scores: 
            log_prob = torch.nn.functional.log_softmax(score[0], dim=0)
            log_prob = log_prob.tolist()
            pred_log_probs.append({idx: scr for idx, scr in enumerate(log_prob)})
        
        transition_scores = model.compute_transition_scores(
                preds.sequences, preds.scores, normalize_logits=True)          
            
        seq_score_no_penalty = transition_scores.sum(axis=1).cpu()
        seq_score =  seq_score_no_penalty / len(pred_token_ids)
        seq_score = seq_score.numpy().item()
        results["no_retrieval"] = pred_text

    # save relevance token scores
    if mode == "always_retrieve":
        do_retrieve = True
    elif mode == "no_retrieval":
        do_retrieve = False
    else:
        if threshold is not None:
            score_dict = {}
            score_dict_old = {}
            score_dict_new = {}
            for tok, id in ret_tokens.items():
                if id not in pred_log_probs[0]:
                    score_dict[tok] = -100
                prob = pred_log_probs[0][id]
                score_dict_old[tok] = float(prob)
                score_dict_new[tok] = float(np.exp(float(prob)))
                score_dict[tok] = float(np.exp(float(prob)))
            do_retrieve = (
                (score_dict["[Retrieval]"]
                / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) )
                > threshold
            )
            do_retrieve = bool(do_retrieve) # numpy to python bool !!!
            print("threshold", threshold, "do_retrieve", do_retrieve)
            proba_retrieval_thresh = score_dict_new["[Retrieval]"] \
                / (score_dict_new["[Retrieval]"] + score_dict_new["[No Retrieval]"])
            print("new r: {}, nr: {}, r/(r+nr): {}".format(score_dict_new["[Retrieval]"], 
                    score_dict_new["[No Retrieval]"], proba_retrieval_thresh))
            logproba_retrieval_thresh = score_dict_old["[Retrieval]"] \
                / (score_dict_old["[Retrieval]"] + score_dict_old["[No Retrieval]"])
            print("old r: {}, nr: {}, r/(r+nr): {}".format(score_dict_old["[Retrieval]"], 
                    score_dict_old["[No Retrieval]"], logproba_retrieval_thresh))
            # stt()
            detailed_init_score.update({
                "logproba_retrieval_thresh": float(logproba_retrieval_thresh), 
                "proba_retrieval_thresh": float(proba_retrieval_thresh),
                "pred_retrieval_decision": pred, 
                "proba_r": score_dict_new["[Retrieval]"], 
                "logr": score_dict_old["[Retrieval]"],
                "lognr": score_dict_old["[No Retrieval]"], 
                "proba_nr": score_dict_new["[No Retrieval]"],
                "seq_logprob": float(seq_score),
            })

        else:
            do_retrieve = "[Retrieval]" in pred
            print("pred", pred)

    detailed_init_score.update({"do_retrieve": do_retrieve})
    overall_scores = {}
    if do_retrieve is True:
        evidence_augmented_inputs = [
            prompt
            + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
                para["title"], para["text"]
            )
            for para in evidences
        ]
         
        inputs = [
            tokenizer(evidence_augmented_input, return_tensors="pt").to("cuda")
            for evidence_augmented_input in evidence_augmented_inputs
        ]

        preds = [
            model.generate(
            **_input,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            # temperature=0.0,
            do_sample=False,
            top_p=1.0,
            stopping_criteria=stopping_criteria,
            
        ) for _input in inputs]
        
    

        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        for p_idx, pred in enumerate(preds):
            pred_text = tokenizer.batch_decode(
                pred.sequences[:, inputs[p_idx].input_ids.shape[1]:], 
                skip_special_tokens=True, 
            )[0]
            pred_token_ids = pred.sequences[:, inputs[p_idx].input_ids.shape[1]:].squeeze(dim=0).tolist()
            pred_log_probs = []
            
            for score in pred.scores:
                log_prob = torch.nn.functional.log_softmax(score[0], dim=0)                
                log_prob, idxs = log_prob.topk(5000)
                log_prob = log_prob.tolist()              
                pred_log_probs.append({idx.item(): scr for idx, scr in zip(idxs, log_prob)})
                
                
            transition_scores = model.compute_transition_scores(
                pred.sequences, pred.scores, normalize_logits=True)          
            
            seq_score_no_penalty = transition_scores.sum(axis=1).cpu()
            seq_score =  seq_score_no_penalty / len(pred_token_ids)
            seq_score = seq_score.numpy().item()

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
            for tok, id in rel_tokens.items():
                prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
                relevance_score_dict[p_idx][tok] = float(np.exp(float(prob)))

            if grd_tokens is not None:
                groundness_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(grd_tokens.values()):
                        groundness_token_appear_indices.append(tok_idx)
                        break
                if len(groundness_token_appear_indices) > 0:
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in grd_tokens.items():
                        prob = (
                            pred_log_probs[idx][token_id]
                            if token_id in pred_log_probs[idx]
                            else -100
                        )
                        grd_score_dict[p_idx][token] = float(np.exp(float(prob)))

            if ut_tokens is not None:
                utility_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in ut_tokens.items():
                        prob = (
                            pred_log_probs[idx][token_id]
                            if token_id in pred_log_probs[idx]
                            else -100
                        )
                        ut_score_dict[p_idx][token] = float(np.exp(float(prob)))

            relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                np.sum(list(relevance_score_dict[p_idx].values()))
            )

            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (
                    grd_score_dict[p_idx]["[Fully supported]"] / gt_sum
                ) + 0.5 * (grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
            else:
                ground_score = 0.0

            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum(
                    [
                        ut_scores[i]
                        * (ut_score_dict[p_idx]["[Utility:{}]".format(i + 1)] / ut_sum)
                        for i in range(len(ut_scores))
                    ]
                )
            else:
                utility_score = 0.0
                
            overall_scores[p_idx] = {}

            if use_seqscore is True:
                log_seq_score = float(np.exp(seq_score))
                log_relevance_score = w_rel * relevance_score
                log_ground_score = w_sup * ground_score
                log_utility_score = w_use * utility_score
                
                overall_scores.update({
                    "log_seq_score_penalty": float(seq_score),
                    "log_seq_score_no_penalty": float(seq_score_no_penalty),
                    "seq_score_penalty": float(np.exp(seq_score)),
                    "seq_score_no_penalty": float(np.exp(seq_score_no_penalty)),
                    "weighted_relevance_score": float(log_relevance_score),
                    "weighted_ground_score": float(log_ground_score),
                    "weighted_utility_score": float(log_utility_score),
                })
                               
                
                final_score = (
                    float(np.exp(seq_score))
                    + w_rel * relevance_score
                    + w_sup * ground_score
                    + w_use * utility_score
                )
                
                print("seq: {:0.4f}, rel: {:0.4f}, sup: {:0.4f}, use: {:0.4f}, final: {:0.4f}, do_retrieve: {}".format(
                    log_seq_score, log_relevance_score, log_ground_score, log_utility_score, final_score, do_retrieve)
                )
                
            else:
                final_score = (
                    w_rel * relevance_score
                    + w_sup * ground_score
                    + w_use * utility_score
                )

            overall_scores[p_idx].update({
                "final_score": float(final_score),
                "relevance_score": float(relevance_score),
                "ground_score": float(ground_score),
                "utility_score": float(utility_score),
                "relevance_score_dict": relevance_score_dict,
                "grd_score_dict": grd_score_dict,
                "ut_score_dict": ut_score_dict,
            })
            results["retrieval_{}".format(p_idx)] = {
                "pred": pred_text,
                "score": final_score if pred_text != "</s>" else -1,
                "ctx": evidences[p_idx],
            }

    else:
        prompt += "[No Retrieval]"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda") # size 22
        preds = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            top_p=1.0,
            stopping_criteria=stopping_criteria,
        )
        
        pred_text = tokenizer.batch_decode(
            preds.sequences[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True, 
        )[0]
        
        pred = pred_text

    if len(results) == 1:
        postprocessed_pred = postprocess_answer_option_conditioned(pred)
        return postprocessed_pred, results, do_retrieve, detailed_init_score, overall_scores
    else:
        answer2score = {}
        if closed is True:
            for key, result in results.items():
                if key == "no_retrieval":
                    continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(
                answer2score.items(), key=lambda x: x[1], reverse=True
            )
            best_option = sorted_answers[0][0]
        else:
            path2score = {
                key: item["score"]
                for key, item in results.items()
                if key != "no_retrieval"
            }
            best_path = sorted(path2score.items(), key=lambda x: x[1], reverse=True)[0][
                0
            ]
            best_option = results[best_path]["pred"]
        return best_option, results, do_retrieve, detailed_init_score, overall_scores


def process_data_evidences(demonstration, top_n):
    ctx_key = "ctxs" if "ctxs" in demonstration else "top_contexts"
    prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
    evidences = demonstration[ctx_key][:top_n]
    return prompt, evidences


def preprocess_input_data(dataset, task=None):
    new_data = []
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
    for item in dataset:
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"],
                answer_labels["B"],
                answer_labels["C"],
                answer_labels["D"],
            )
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = (
                instruction + "\n\n### Input:\n" + item["question"] + choices
            )
            item["answers"] = [item["answerKey"]]
        else:
            prompt = (
                instruction + "\n\n## Input:\n\n" + item["question"]
                if instruction is not None
                else item["question"]
            )
            item["instruction"] = prompt
        new_data.append(item)

    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=15)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument(
        "--download_dir",
        type=str,
        help="specify vllm model download dir",
        default=".cache",
    )
    parser.add_argument(
        "--ndocs",
        type=int,
        default=10,
        help="Number of documents to retrieve per questions",
    )
    parser.add_argument(
        "--world_size", type=int, default=1, help="world size to use multiple GPUs."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="half",
        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.",
    )
    # Decoding hyperparams
    parser.add_argument(
        "--threshold", type=float, default=None, help="Adaptive threshold."
    )
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument(
        "--use_groundness", action="store_true", help="use ground score"
    )
    parser.add_argument("--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width", type=int, default=2, help="beam search width")
    parser.add_argument("--max_depth", type=int, default=2, help="tree depth width")
    parser.add_argument(
        "--w_rel", type=float, default=1.0, help="reward weight for document relevance"
    )
    parser.add_argument(
        "--w_sup",
        type=float,
        default=1.0,
        help="reward weight for generation support (attribution)",
    )
    parser.add_argument(
        "--w_use",
        type=float,
        default=1.0,
        help="reward weight for overall completeness / utility.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="mode to control retrieval.",
        default="default",
        choices=["adaptive_retrieval", "no_retrieval", "always_retrieve"],
    )
    parser.add_argument(
        "--metric", type=str, help="metric to be used during evaluation"
    )

    parser.add_argument(
        "--start_idx", type=int, help="metric to be used during evaluation", default=None,
    )
    
    parser.add_argument(
        "--end_idx", type=int, help="metric to be used during evaluation", default=None,
    )

    args = parser.parse_args()
    gpt = args.model_name
    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    input_data = preprocess_input_data(input_data, task=args.task)
    tokenizer = AutoTokenizer.from_pretrained(
        gpt,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        gpt, 
        device_map="cuda:0", 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility
    )
    model.generation_config.temperature = None
    model.generation_config.do_sample = False
    
    # model = model
    
    stop_words = ["</s>"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() 
                    for stop_word in stop_words]
    stopping_criterion = StoppingCriteriaList([StoppingCriteriaSub(tokenizer, stops=stop_words_ids)])

    def generate(stopping_criterion, tokenizer, prompt, evidences, max_new_tokens):
        return call_model_rerank_w_scores_batch(
            stopping_criterion,
            tokenizer,
            prompt,
            evidences=evidences,
            model=model,
            max_new_tokens=max_new_tokens,
            rel_tokens=rel_tokens,
            ret_tokens=ret_tokens,
            grd_tokens=grd_tokens,
            ut_tokens=ut_tokens,
            threshold=args.threshold,
            use_seqscore=args.use_seqscore,
            w_rel=args.w_rel,
            w_sup=args.w_sup,
            w_use=args.w_use,
            mode=args.mode,
            closed=args.task in ["fever", "arc_c"],
        )

    preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    count = 0
    detailed_init_score_all = []
    overall_scores_all = []
    if args.start_idx is not None:
        if args.end_idx is not None:
            input_data = input_data[args.start_idx:args.end_idx]
        else:
            input_data = input_data[args.start_idx:]
        
    for i, row in tqdm(enumerate(input_data), total=len(input_data)):
        results = {}
        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        _, evidences = process_data_evidences(row, top_n=args.ndocs)
        pred, results, do_retrieve, detailed_init_score, overall_scores = generate(
            stopping_criterion,
            tokenizer,
            prompt,
            evidences,
            max_new_tokens=args.max_new_tokens,
        )
        
        detailed_init_score_all.append(detailed_init_score)
        overall_scores_all.append(overall_scores)
        
        if type(pred) is str and len(pred) > 1 and (pred[0] == "#" or pred[0] == ":"):
            pred = pred[1:]
            
        prompts.append(prompt)
        preds.append(pred)
        all_results.append(results)
        if do_retrieve is True:
            count += 1
        if "answers" not in row and "answer" in row:
            row["answers"] = (
                [row["answer"]] if type(row["answer"]) is str else row["answer"]
            )
        if args.metric == "accuracy":
            metric_result = accuracy(pred, row["output"])

        elif args.metric == "match":
            if "SUPPORTS" in pred:
                pred = "true"
            elif "REFUTES" in pred:
                pred = "false"
            metric_result = match(pred, row["answers"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)
        print("average: {}, last {}".format(np.mean(metric_results), metric_results[-1]) )
        log_freq = 2
        if i % log_freq == 0:
            final_results = {
                "preds": preds,
                "prompts": prompts,
                "metric_results": metric_results,
                "all_results": all_results,
                "golds": golds,
                "metric": args.metric,
                "metric_mean": np.mean(metric_results),
                "count": count,
                "total_data": len(preds),
                "retrieval_frac": count/len(preds),
            }
            with open(args.output_file + "_tmp", "w") as outfile:
                json.dump(final_results, outfile)

    final_results = {
        "preds": preds,
        "prompts": prompts,
        "metric_results": metric_results,
        "all_results": all_results,
        "golds": golds,
        "metric": args.metric,
        "metric_mean": np.mean(metric_results),
        "scores": scores,
        "count": count,
        "total_data": len(preds),
        "retrieval_frac": count/len(preds),
        "detailed_init_score": detailed_init_score_all, 
        "overall_scores": overall_scores_all,
        
    }
    with open(args.output_file, "w") as outfile:
        json.dump(final_results, outfile)

    print("Final result: {0}".format(np.mean(metric_results)))
    print("Retrieval Frequencies: {0}".format(count / len(final_results)))


if __name__ == "__main__":
    main()
