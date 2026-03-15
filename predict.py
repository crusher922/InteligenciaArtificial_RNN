import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================
# 1. Cargar recursos
# =========================
model = load_model("translation_model.keras")

with open("input_tokenizer.pkl", "rb") as f:
    input_tokenizer = pickle.load(f)

with open("target_tokenizer.pkl", "rb") as f:
    target_tokenizer = pickle.load(f)

with open("translation_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

max_encoder_seq_length = metadata["max_encoder_seq_length"]
max_decoder_seq_length = metadata["max_decoder_seq_length"]

reverse_target_word_index = {index: word for word, index in target_tokenizer.word_index.items()}

# =========================
# 2. Reconstruir encoder y decoder
# =========================
# Capas del modelo entrenado
encoder_inputs = model.input[0]
decoder_inputs = model.input[1]

encoder_embedding = model.get_layer(index=2)
encoder_lstm = model.get_layer("encoder_lstm")

decoder_embedding_layer = model.get_layer(index=3)
decoder_lstm = model.get_layer("decoder_lstm")
decoder_dense = model.get_layer("decoder_dense")

# Encoder de inferencia
encoder_embedded = encoder_embedding(encoder_inputs)
_, state_h_enc, state_c_enc = encoder_lstm(encoder_embedded)
encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

# Decoder de inferencia
decoder_state_input_h = Input(shape=(encoder_lstm.units,))
decoder_state_input_c = Input(shape=(encoder_lstm.units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedded = decoder_embedding_layer(decoder_inputs)
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_embedded,
    initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# =========================
# 3. Función de traducción
# =========================
def decode_sequence(input_text):
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')

    states_value = encoder_model.predict(input_seq, verbose=0)

    start_token_index = target_tokenizer.word_index["<start>"]
    end_token = "<end>"

    target_seq = np.array([[start_token_index]])

    decoded_sentence = []

    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, "")

        if sampled_word == end_token or sampled_word == "":
            break

        decoded_sentence.append(sampled_word)

        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return " ".join(decoded_sentence)

# =========================
# 4. Prueba interactiva
# =========================
print("Traductor RNN pequeño")
print("Escribe una frase en español. Ejemplo: hola")
print("Escribe 'salir' para terminar.\n")

while True:
    text = input("Entrada: ").strip().lower()

    if text == "salir":
        print("Programa finalizado.")
        break

    translation = decode_sequence(text)
    print("Traducción:", translation if translation else "[sin traducción aprendida]")
    print()