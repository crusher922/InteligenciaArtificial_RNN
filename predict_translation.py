import json
import pickle
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "translation_model.keras"
INPUT_TOKENIZER_PATH = BASE_DIR / "models" / "input_tokenizer.pkl"
TARGET_TOKENIZER_PATH = BASE_DIR / "models" / "target_tokenizer.pkl"
METADATA_PATH = BASE_DIR / "models" / "translation_metadata.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_inference_models(model, latent_dim):
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


def translate_text(text, encoder_model, decoder_model, input_tokenizer, target_tokenizer, max_input_len, max_target_len):
    text = text.strip().lower()
    input_seq = input_tokenizer.texts_to_sequences([text])
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


def main():
    metadata = load_json(METADATA_PATH)
    model = load_model(MODEL_PATH)
    input_tokenizer = load_pickle(INPUT_TOKENIZER_PATH)
    target_tokenizer = load_pickle(TARGET_TOKENIZER_PATH)

    encoder_model, decoder_model = build_inference_models(
        model=model,
        latent_dim=metadata["latent_dim"]
    )

    print("Traductor RNN listo. Escribe una frase en inglés.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        user_text = input("Texto: ").strip()
        if user_text.lower() == "salir":
            print("Programa finalizado.")
            break

        prediction = translate_text(
            user_text,
            encoder_model,
            decoder_model,
            input_tokenizer,
            target_tokenizer,
            metadata["max_input_len"],
            metadata["max_target_len"]
        )

        if prediction:
            print(f"Traducción: {prediction}\n")
        else:
            print("Traducción: [sin traducción generada]\n")


if __name__ == "__main__":
    main()