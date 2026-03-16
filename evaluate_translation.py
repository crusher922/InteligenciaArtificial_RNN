import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "translation_pairs.json"

MODEL_PATH = BASE_DIR / "models" / "translation_model.keras"
INPUT_TOKENIZER_PATH = BASE_DIR / "models" / "input_tokenizer.pkl"
TARGET_TOKENIZER_PATH = BASE_DIR / "models" / "target_tokenizer.pkl"
METADATA_PATH = BASE_DIR / "models" / "translation_metadata.json"

LOGS_DIR = BASE_DIR / "logs"
EVAL_SUMMARY_PATH = LOGS_DIR / "evaluation_summary.json"
EVAL_PREDICTIONS_PATH = LOGS_DIR / "evaluation_predictions.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data():
    pairs = load_json(DATASET_PATH)
    sources = [item["source"].strip().lower() for item in pairs]
    targets = [item["target"].strip().lower() for item in pairs]
    return sources, targets


def sequence_to_text(sequence, tokenizer):
    reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
    words = []
    for idx in sequence:
        if idx == 0:
            continue
        word = reverse_word_index.get(int(idx), "")
        if word in ("startseq", "endseq", "<OOV>") or word == "":
            continue
        words.append(word)
    return " ".join(words).strip()


def build_inference_models(model, latent_dim, target_vocab_size):
    encoder_inputs = model.input[0]
    encoder_embedding = model.get_layer("encoder_embedding")(encoder_inputs)
    _, encoder_state = model.get_layer("encoder_rnn")(encoder_embedding)
    encoder_model = Model(encoder_inputs, encoder_state)

    decoder_embedding_layer = model.get_layer("decoder_embedding")
    decoder_rnn_layer = model.get_layer("decoder_rnn")
    decoder_dense_layer = model.get_layer("decoder_dense")

    decoder_state_input = Input(shape=(latent_dim,), name="decoder_state_input")
    decoder_single_input = Input(shape=(1,), name="decoder_single_input")

    decoder_embedded = decoder_embedding_layer(decoder_single_input)
    decoder_outputs, decoder_state = decoder_rnn_layer(
        decoder_embedded,
        initial_state=decoder_state_input
    )
    decoder_outputs = decoder_dense_layer(decoder_outputs)

    decoder_model = Model(
        [decoder_single_input, decoder_state_input],
        [decoder_outputs, decoder_state]
    )

    return encoder_model, decoder_model


def decode_sequence(input_text, encoder_model, decoder_model, input_tokenizer, target_tokenizer, max_input_len, max_target_len):
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding="post")

    state = encoder_model.predict(input_seq, verbose=0)

    start_token = target_tokenizer.word_index.get("startseq")
    end_token = target_tokenizer.word_index.get("endseq")

    target_seq = np.array([[start_token]])
    decoded_tokens = []

    for _ in range(max_target_len):
        output_tokens, state = decoder_model.predict([target_seq, state], verbose=0)
        sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))

        if sampled_token_index == 0 or sampled_token_index == end_token:
            break

        sampled_word = target_tokenizer.index_word.get(sampled_token_index, "")
        if sampled_word and sampled_word not in ("startseq", "endseq", "<OOV>"):
            decoded_tokens.append(sampled_word)

        target_seq = np.array([[sampled_token_index]])

    return " ".join(decoded_tokens).strip()


def token_match_ratio(reference, prediction):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()

    if not ref_tokens:
        return 0.0

    matches = sum(1 for r, p in zip(ref_tokens, pred_tokens) if r == p)
    return matches / len(ref_tokens)


def main():
    metadata = load_json(METADATA_PATH)
    input_tokenizer = load_pickle(INPUT_TOKENIZER_PATH)
    target_tokenizer = load_pickle(TARGET_TOKENIZER_PATH)
    model = load_model(MODEL_PATH)

    sources, targets = load_data()

    _, test_indices = train_test_split(
        np.arange(len(sources)),
        test_size=metadata["test_size"],
        random_state=metadata["random_state"],
        shuffle=True
    )

    test_sources = [sources[i] for i in test_indices]
    test_targets = [targets[i] for i in test_indices]

    encoder_model, decoder_model = build_inference_models(
        model=model,
        latent_dim=metadata["latent_dim"],
        target_vocab_size=metadata["target_vocab_size"]
    )

    smoothie = SmoothingFunction().method1
    rows = []

    exact_matches = 0
    bleu_scores = []
    token_accuracies = []

    for source_text, target_text in zip(test_sources, test_targets):
        predicted_text = decode_sequence(
            source_text,
            encoder_model,
            decoder_model,
            input_tokenizer,
            target_tokenizer,
            metadata["max_input_len"],
            metadata["max_target_len"]
        )

        exact_match = int(predicted_text == target_text)
        bleu = sentence_bleu(
            [target_text.split()],
            predicted_text.split() if predicted_text else [""],
            smoothing_function=smoothie
        )
        token_acc = token_match_ratio(target_text, predicted_text)

        exact_matches += exact_match
        bleu_scores.append(float(bleu))
        token_accuracies.append(float(token_acc))

        rows.append({
            "source": source_text,
            "expected": target_text,
            "predicted": predicted_text,
            "exact_match": exact_match,
            "bleu_score": float(bleu),
            "token_accuracy": float(token_acc)
        })

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_PREDICTIONS_PATH, index=False, encoding="utf-8")

    summary = {
        "samples_evaluated": len(test_sources),
        "exact_match_accuracy": float(exact_matches / len(test_sources)) if test_sources else 0.0,
        "average_bleu_score": float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        "average_token_accuracy": float(np.mean(token_accuracies)) if token_accuracies else 0.0
    }

    with open(EVAL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("Evaluación finalizada.")
    print(json.dumps(summary, indent=4, ensure_ascii=False))
    print(f"Predicciones guardadas en: {EVAL_PREDICTIONS_PATH}")
    print(f"Resumen guardado en: {EVAL_SUMMARY_PATH}")


if __name__ == "__main__":
    main()