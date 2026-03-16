import os
import json
import pickle
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)

# =========================
# CONFIGURACIÓN
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

DATASET_PATH = DATA_DIR / "translation_pairs.json"
TRAIN_LOG_PATH = LOGS_DIR / "training.log"
ERROR_LOG_PATH = LOGS_DIR / "errors.log"
HISTORY_CSV_PATH = LOGS_DIR / "training_history.csv"

MODEL_PATH = MODELS_DIR / "translation_model.keras"
INPUT_TOKENIZER_PATH = MODELS_DIR / "input_tokenizer.pkl"
TARGET_TOKENIZER_PATH = MODELS_DIR / "target_tokenizer.pkl"
METADATA_PATH = MODELS_DIR / "translation_metadata.json"
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_translation_model.keras"

EPOCHS = 200
BATCH_SIZE = 8
EMBEDDING_DIM = 64
LATENT_DIM = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42


def setup_directories():
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    (LOGS_DIR / "plots").mkdir(exist_ok=True)


def setup_logging():
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    train_handler = logging.FileHandler(TRAIN_LOG_PATH, encoding="utf-8")
    train_handler.setLevel(logging.INFO)
    train_handler.setFormatter(formatter)

    error_handler = logging.FileHandler(ERROR_LOG_PATH, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(train_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    sources = [item["source"].strip().lower() for item in pairs]
    targets = [f"startseq {item['target'].strip().lower()} endseq" for item in pairs]

    return sources, targets


def fit_tokenizers(input_texts, target_texts):
    input_tokenizer = Tokenizer(filters="", lower=True, oov_token="<OOV>")
    target_tokenizer = Tokenizer(filters="", lower=True, oov_token="<OOV>")

    input_tokenizer.fit_on_texts(input_texts)
    target_tokenizer.fit_on_texts(target_texts)

    return input_tokenizer, target_tokenizer


def prepare_sequences(input_texts, target_texts, input_tokenizer, target_tokenizer):
    input_sequences = input_tokenizer.texts_to_sequences(input_texts)
    target_sequences = target_tokenizer.texts_to_sequences(target_texts)

    max_input_len = max(len(seq) for seq in input_sequences)
    max_target_len = max(len(seq) for seq in target_sequences)

    encoder_input_data = pad_sequences(input_sequences, maxlen=max_input_len, padding="post")
    decoder_full_data = pad_sequences(target_sequences, maxlen=max_target_len, padding="post")

    decoder_input_data = decoder_full_data[:, :-1]
    decoder_target_data = decoder_full_data[:, 1:]
    decoder_target_data = np.expand_dims(decoder_target_data, -1)

    return encoder_input_data, decoder_input_data, decoder_target_data, max_input_len, max_target_len


def build_model(input_vocab_size, target_vocab_size, max_input_len, max_target_len):
    encoder_inputs = Input(shape=(max_input_len,), name="encoder_inputs")
    encoder_embedding = Embedding(
        input_dim=input_vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="encoder_embedding"
    )(encoder_inputs)

    encoder_rnn = SimpleRNN(
        LATENT_DIM,
        return_state=True,
        name="encoder_rnn"
    )
    _, encoder_state = encoder_rnn(encoder_embedding)

    decoder_inputs = Input(shape=(max_target_len - 1,), name="decoder_inputs")
    decoder_embedding_layer = Embedding(
        input_dim=target_vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="decoder_embedding"
    )
    decoder_embedding = decoder_embedding_layer(decoder_inputs)

    decoder_rnn_layer = SimpleRNN(
        LATENT_DIM,
        return_sequences=True,
        return_state=True,
        name="decoder_rnn"
    )
    decoder_outputs, _ = decoder_rnn_layer(decoder_embedding, initial_state=encoder_state)

    decoder_dense = Dense(target_vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_tokenizer(tokenizer, path):
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def save_metadata(metadata, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def main():
    setup_directories()
    logger = setup_logging()

    try:
        logger.info("Iniciando entrenamiento del traductor RNN...")
        logger.info(f"Dataset: {DATASET_PATH}")

        input_texts, target_texts = load_data(DATASET_PATH)
        logger.info(f"Total de pares cargados: {len(input_texts)}")

        train_indices, test_indices = train_test_split(
            np.arange(len(input_texts)),
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        train_input_texts = [input_texts[i] for i in train_indices]
        train_target_texts = [target_texts[i] for i in train_indices]

        logger.info(f"Cantidad para entrenamiento: {len(train_input_texts)}")
        logger.info(f"Cantidad reservada para prueba: {len(test_indices)}")

        input_tokenizer, target_tokenizer = fit_tokenizers(train_input_texts, train_target_texts)

        encoder_input_data, decoder_input_data, decoder_target_data, max_input_len, max_target_len = prepare_sequences(
            train_input_texts,
            train_target_texts,
            input_tokenizer,
            target_tokenizer
        )

        input_vocab_size = len(input_tokenizer.word_index) + 1
        target_vocab_size = len(target_tokenizer.word_index) + 1

        logger.info(f"Vocabulario entrada: {input_vocab_size}")
        logger.info(f"Vocabulario salida: {target_vocab_size}")
        logger.info(f"Longitud máxima entrada: {max_input_len}")
        logger.info(f"Longitud máxima salida: {max_target_len}")

        model = build_model(
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            max_input_len=max_input_len,
            max_target_len=max_target_len
        )

        callbacks = [
            ModelCheckpoint(
                filepath=BEST_MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                verbose=1
            ),
            CSVLogger(HISTORY_CSV_PATH, append=False)
        ]

        history = model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        model.save(MODEL_PATH)
        save_tokenizer(input_tokenizer, INPUT_TOKENIZER_PATH)
        save_tokenizer(target_tokenizer, TARGET_TOKENIZER_PATH)

        metadata = {
            "embedding_dim": EMBEDDING_DIM,
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "max_input_len": max_input_len,
            "max_target_len": max_target_len,
            "input_vocab_size": input_vocab_size,
            "target_vocab_size": target_vocab_size,
            "train_size": len(train_input_texts),
            "test_size_count": len(test_indices),
            "dataset_path": str(DATASET_PATH),
            "model_path": str(MODEL_PATH)
        }
        save_metadata(metadata, METADATA_PATH)

        logger.info("Entrenamiento finalizado correctamente.")
        logger.info(f"Modelo guardado en: {MODEL_PATH}")
        logger.info(f"Historial guardado en: {HISTORY_CSV_PATH}")

        final_metrics = {
            "final_train_loss": float(history.history["loss"][-1]),
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_val_accuracy": float(history.history["val_accuracy"][-1])
        }
        logger.info(f"Métricas finales: {final_metrics}")

    except Exception as e:
        logger.error("Error durante el entrenamiento.")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()