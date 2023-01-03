import streamlit as st

st.set_page_config(
    page_title="Photong",
    page_icon="🖼️",
    menu_items={
        "Get Help": "https://github.com/photong-ml/photong-v2-streamlit/blob/main/tutorial/README.md#using-photong",
        "Report a bug": "https://github.com/photong-ml/photong-v2-streamlit/issues/new",
        "About": """
        ## Photong
        Photong is an app that uses machine learning technology to generate a 16-bar melody from a photo.

        [View GitHub repository](https://github.com/photong-ml/photong-v2-streamlit)
        """
    }
)

st.title("Photong")
st.write(
    "Unsure what to do? Check out the [tutorial](https://github.com/photong-ml/photong-v2-streamlit/blob/main/tutorial/README.md#using-photong).")


with st.spinner("Loading required imports..."):
    from copy import deepcopy
    from io import BytesIO
    from math import floor
    from pathlib import Path

    import note_seq
    import numpy as np
    import tensorflow as tf
    from magenta.models import music_vae
    from scipy.io import wavfile


@st.cache(allow_output_mutation=True, show_spinner=False)
def init_encoder():
    inception_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
    )
    img_model = tf.keras.Model(
        inception_model.input,
        inception_model.layers[-1].output,
    )

    return img_model


@st.cache(allow_output_mutation=True, show_spinner=False)
def init_model():
    valence_model_path = "saved/valence_model_latest.h5"
    arousal_model_path = "saved/arousal_model_latest.h5"
    embedding_model_path = "saved/embedding_model_latest.h5"

    with st.spinner("Downloading models..."):
        if not Path(valence_model_path).exists():
            import gdown
            gdown.download(
                id="1N9MQpDROU3RfJ9msPdAQYBexUwEuxCRg",
                output=valence_model_path,
                quiet=True,
            )
        if not Path(arousal_model_path).exists():
            import gdown
            gdown.download(
                id="1MpQwvZsGr4VIQxYHO0quMLSJWw5mQZuu",
                output=arousal_model_path,
                quiet=True,
            )
        if not Path(embedding_model_path).exists():
            import gdown
            gdown.download(
                id="1Mypo6XOBS6uIjrY0xsq9m64JNr5krwK0",
                output=embedding_model_path,
                quiet=True,
            )

    valence_model = tf.keras.models.load_model(
        "saved/valence_model_latest.h5")
    arousal_model = tf.keras.models.load_model(
        "saved/arousal_model_latest.h5")
    embedding_model = tf.keras.models.load_model(
        "saved/embedding_model_latest.h5")

    return valence_model, arousal_model, embedding_model


@st.cache(allow_output_mutation=True, show_spinner=False)
def init_decoder():
    checkpoint_dir = f"saved/{decoder_config_name}"

    with st.spinner("Downloading model..."):
        if not Path(checkpoint_dir).exists():
            checkpoint_tar_path = f"saved/{decoder_config_name}.tar"
            import gdown
            gdown.download(
                url=f"https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/{decoder_config_name}.tar",
                output=checkpoint_tar_path,
                quiet=True,
            )
            gdown.extractall(checkpoint_tar_path, checkpoint_dir)
            Path(checkpoint_tar_path).unlink()

    decoder_model = music_vae.TrainedModel(
        decoder_config,
        batch_size=8,
        checkpoint_dir_or_path=f"{checkpoint_dir}/{decoder_config_name}.ckpt",
    )

    return decoder_model


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)


def number_to_note(number: int) -> tuple:
    octave = number // NOTES_IN_OCTAVE
    assert octave in OCTAVES, f"octave {octave} not in {OCTAVES}"
    assert 0 <= number <= 127, f"number {number} not in [0, 127]"
    note = NOTES[number % NOTES_IN_OCTAVE]

    return note, octave


def note_to_number(note: str, octave: int) -> int:
    assert note in NOTES, f"note {note} not in {NOTES}"
    assert octave in OCTAVES, f"octave {octave} not in {OCTAVES}"

    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)

    assert 0 <= note <= 127, f"note {note} not in [0, 127]"

    return note


THEORY_CONFIG = {
    "maj": {
        "diatonic": [0, 2, 4, 5, 7, 9, 11],
        "chords": {
            "default": "maj",
            "maj": [0, 5, 7],
            "min": [2, 4, 9],
            "dim": [11],
        }
    },
    "min": {
        "diatonic": [0, 2, 3, 5, 7, 8, 10],
        "chords": {
            "default": "min",
            "maj": [3, 8, 10],
            "min": [0, 5, 7],
            "dim": [2],
        },
    }
}

CHORD_OFFSETS = {
    "maj": [0, 4, 7, 12],
    "min": [0, 3, 7, 12],
    "dim": [0, 3, 6, 12],
}


def touch_up(aud_ns, arousal_res, tonality):
    config = THEORY_CONFIG[tonality]

    key = number_to_note(aud_ns.notes[0].pitch)[0]
    key_offset = NOTES.index(key)
    for note in aud_ns.notes:
        number = (note.pitch - key_offset) % NOTES_IN_OCTAVE
        if number not in config["diatonic"]:
            # make diatonic
            note.pitch = note.pitch + np.random.default_rng().choice([-1, 1])

    # append ending note on tonic
    end_note = deepcopy(aud_ns.notes[-1])
    end_note.pitch = aud_ns.notes[0].pitch
    end_note.start_time = 2 * round(end_note.end_time / 2)
    end_note.end_time = end_note.start_time + 2
    aud_ns.notes.append(end_note)

    chord_time_threshold = 0
    index = 0
    temp_sequence = deepcopy(aud_ns)
    while index < len(aud_ns.notes):
        note = aud_ns.notes[index]
        if note.start_time < chord_time_threshold:
            index += 1
            continue

        number = (note.pitch - key_offset) % NOTES_IN_OCTAVE

        chord_type = config["chords"]["default"]
        for chord in ["maj", "min", "dim"]:
            if number in config["chords"][chord]:
                chord_type = chord
                break

        chord_end_time = note.end_time
        while index + 1 < len(aud_ns.notes) and aud_ns.notes[index + 1].pitch == note.pitch:
            chord_end_time = aud_ns.notes[index + 1].end_time
            index += 1
        chord_end_time = 2 * floor((chord_end_time / 2) + 1)

        chord_notes = [deepcopy(aud_ns.notes[0]) for _ in range(4)]
        for (chord_note, chord_offset) in zip(chord_notes, CHORD_OFFSETS[chord_type]):
            note_name = number_to_note(note.pitch + chord_offset)
            # move down two octaves
            chord_note.pitch = note_to_number(
                *(note_name[0], note_name[1] - 2))
            chord_note.start_time = note.start_time
            chord_note.end_time = chord_end_time
            temp_sequence.notes.append(chord_note)

        chord_time_threshold = chord_end_time
        index += 1

    aud_ns = temp_sequence

    def change_tempo(note_sequence, new_tempo):
        new_sequence = deepcopy(note_sequence)
        ratio = note_sequence.tempos[0].qpm / new_tempo
        for note in new_sequence.notes:
            note.start_time *= ratio
            note.end_time *= ratio
        new_sequence.tempos[0].qpm = new_tempo
        return new_sequence

    aud_ns = change_tempo(aud_ns, new_tempo=float(arousal_res))

    return aud_ns


def load_image(img):
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def img_to_emb(img):
    img = load_image(img)
    img = tf.expand_dims(img, axis=0)
    img_features = img_model(img)
    return img_features.numpy()


def emb_to_aud(emb):
    return decoder_model.decode(
        length=decoder_config.hparams.max_seq_len,
        z=emb,
        temperature=0.5,
    )[0]


with st.spinner("Initialising... This will only happen once and may take a few seconds."):
    Path("saved").mkdir(exist_ok=True)
    img_model = init_encoder()
    valence_model, arousal_model, embedding_model = init_model()
    decoder_config_name = "hierdec-mel_16bar"
    decoder_config = music_vae.configs.CONFIG_MAP[decoder_config_name]

file = st.file_uploader(
    "Choose an image to get started.",
    type=["png", "jpg", "jpeg"]
)

if file is not None:
    with st.spinner("Got it! Let's see..."):
        img_emb = img_to_emb(file.getvalue())
        valence_res = valence_model.predict(img_emb, verbose=0).reshape(-1)[0]
        tonality = "maj" if valence_res >= 0.5 else "min"
        st.write(
            f"I think it's a {'happy' if tonality == 'maj' else 'sad'} image.")
        arousal_res = arousal_model.predict(img_emb, verbose=0).reshape(-1)[0]
        st.write(
            f"I think it's {'an exciting' if arousal_res > 0.5 else 'a peaceful'} image.")
        arousal_res = 160 * 1 / (1 + np.exp(-5 * (arousal_res - 0.5))) + 40

    with st.spinner("Generating a melody for your image..."):
        aud_res = embedding_model.predict(img_emb, verbose=0)
        with st.spinner("Loading decoder model... This will only happen once and may take a few seconds."):
            decoder_model = init_decoder()
        aud_ns = emb_to_aud(aud_res)

    with st.spinner("Adding some final touches..."):
        aud_ns = touch_up(aud_ns, arousal_res, tonality)

    with st.spinner("Synthesising the MIDI file..."):
        audio_data = note_seq.fluidsynth(aud_ns, sample_rate=44100.0)
        # Normalize for 16 bit audio
        audio_data = np.int16(
            audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
        )

        virtualfile = BytesIO()
        wavfile.write(virtualfile, 44100, audio_data)

    st.header("Result")
    st.image(file.getvalue())
    st.success("Here is your melody!")
    st.audio(virtualfile)
    virtualfile = BytesIO()
    note_seq.note_sequence_to_pretty_midi(aud_ns).write(virtualfile)
    st.download_button("Download MIDI", virtualfile, "output.mid")
