import json
import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# =========================
# 1. Dataset
# =========================
pairs = [
    ("hola", "hello"),
    ("adios", "goodbye"),
    ("gracias", "thank you"),
    ("por favor", "please"),
    ("buenos dias", "good morning"),
    ("buenas noches", "good night"),
    ("como estas", "how are you"),
    ("estoy bien", "i am fine"),
    ("te amo", "i love you"),
    ("hasta luego", "see you later"),
    ("que hora es", "what time is it"),
    ("donde estas", "where are you"),
    ("necesito ayuda", "i need help"),
    ("tengo hambre", "i am hungry"),
    ("tengo sueño", "i am sleepy"),
    ("quiero agua", "i want water"),
    ("me llamo ana", "my name is ana"),
    ("soy estudiante", "i am a student"),
    ("esto es facil", "this is easy"),
    ("esto es dificil", "this is difficult")
]

# Añadimos tokens especiales al texto de salida
input_texts = [pair[0] for pair in pairs]
target_texts = [f"<start> {pair[1]} <end>" for pair in pairs]

# =========================
# 2. Tokenización
# =========================
input_tokenizer = Tokenizer(filters='')
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
input_word_index = input_tokenizer.word_index

target_tokenizer = Tokenizer(filters='')
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
target_word_index = target_tokenizer.word_index

num_encoder_tokens = len(input_word_index) + 1
num_decoder_tokens = len(target_word_index) + 1

max_encoder_seq_length = max(len(seq) for seq in input_sequences)
max_decoder_seq_length = max(len(seq) for seq in target_sequences)

encoder_input_data = pad_sequences(
    input_sequences,
    maxlen=max_encoder_seq_length,
    padding='post'
)

decoder_input_data = []
decoder_target_data = []

for seq in target_sequences:
    decoder_input = seq[:-1]
    decoder_target = seq[1:]

    decoder_input = pad_sequences([decoder_input], maxlen=max_decoder_seq_length - 1, padding='post')[0]
    decoder_target = pad_sequences([decoder_target], maxlen=max_decoder_seq_length - 1, padding='post')[0]

    decoder_input_data.append(decoder_input)
    decoder_target_data.append(decoder_target)

decoder_input_data = np.array(decoder_input_data)
decoder_target_data = np.array(decoder_target_data)

# One-hot para la salida
decoder_target_data_onehot = to_categorical(decoder_target_data, num_classes=num_decoder_tokens)

# =========================
# 3. Modelo Encoder-Decoder con LSTM
# =========================
latent_dim = 64
embedding_dim = 50

# Encoder
encoder_inputs = Input(shape=(None,), name="encoder_inputs")
encoder_embedding = Embedding(input_dim=num_encoder_tokens, output_dim=embedding_dim, mask_zero=True)(
    encoder_inputs
)
encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,), name="decoder_inputs")
decoder_embedding_layer = Embedding(input_dim=num_decoder_tokens, output_dim=embedding_dim, mask_zero=True)
decoder_embedding = decoder_embedding_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax', name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# =========================
# 4. Entrenamiento
# =========================
history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data_onehot,
    batch_size=4,
    epochs=300,
    verbose=1
)

# =========================
# 5. Guardar modelo y metadatos
# =========================
model.save("translation_model.keras")

with open("input_tokenizer.pkl", "wb") as f:
    pickle.dump(input_tokenizer, f)

with open("target_tokenizer.pkl", "wb") as f:
    pickle.dump(target_tokenizer, f)

metadata = {
    "max_encoder_seq_length": max_encoder_seq_length,
    "max_decoder_seq_length": max_decoder_seq_length,
    "latent_dim": latent_dim,
    "embedding_dim": embedding_dim,
    "num_encoder_tokens": num_encoder_tokens,
    "num_decoder_tokens": num_decoder_tokens
}

with open("translation_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print("\nEntrenamiento finalizado. Modelo guardado correctamente.")