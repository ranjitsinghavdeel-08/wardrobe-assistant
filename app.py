# Wardrobe Assistant - Streamlit app
# Needs data.zip (made by Section 12 of the Colab notebook) in the same folder as this file.
import io
import json
import os
import random
import re
import zipfile

import numpy as np
import streamlit as st

DATA_ZIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.zip")


# ----------------------------------------------------------------------------
# Data loading (runs once, then cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_data():
    with zipfile.ZipFile(DATA_ZIP) as z:
        cfg = json.loads(z.read("config.json"))
        items = json.loads(z.read("index.json"))
        embs = np.load(io.BytesIO(z.read("embeddings.npy"))).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    data = {
        "cfg": cfg,
        "paths": [it["path"] for it in items["top"]],
        "folders": np.array([it["folder"] for it in items["top"]]),
        "colors": np.array([it["colors"] for it in items["top"]], dtype=np.float32),
        "embs": embs,
        "bottoms": items["bottom"],  # {folder: [paths]}
    }
    return data


@st.cache_data(max_entries=300, show_spinner=False)
def read_image(path):
    with zipfile.ZipFile(DATA_ZIP) as z:
        return z.read(path)


# ----------------------------------------------------------------------------
# Search logic (same as the Colab notebook)
# ----------------------------------------------------------------------------
def parse_description(text, cfg):
    t = text.lower().replace("t shirt", "t-shirt").replace("tshirt", "t-shirt")
    colour = None
    for word in sorted(cfg["COLOR_SYNONYMS"], key=len, reverse=True):
        if re.search(r"\b" + re.escape(word) + r"\b", t):
            colour = cfg["COLOR_SYNONYMS"][word]
            break
    garment_folders = None
    for word in sorted(cfg["GARMENT_KEYWORDS"], key=len, reverse=True):
        if re.search(r"\b" + re.escape(word) + r"\b", t):
            garment_folders = cfg["GARMENT_KEYWORDS"][word]
            break
    return colour, garment_folders


def dominant_colour(profile, color_names):
    return color_names[int(np.argmax(profile))]


def predict_top_wear(data, description, event, k=3):
    cfg = data["cfg"]
    color_names = cfg["COLOR_NAMES"]
    colour, garment_folders = parse_description(description, cfg)
    folders = garment_folders or cfg["EVENT_TOP_PREFERENCE"].get(event, cfg["TOP_FOLDERS"])
    folders = [f for f in folders if f in cfg["TOP_FOLDERS"]] or cfg["TOP_FOLDERS"]

    ids = np.where(np.isin(data["folders"], folders))[0]
    if len(ids) == 0:
        return [], {"colour": colour, "folders": folders}

    centroid = data["embs"][ids].mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    sim = data["embs"][ids] @ centroid
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-9)

    if colour and colour in color_names:
        ci = color_names.index(colour)
        score = 0.7 * data["colors"][ids, ci] + 0.3 * sim
    else:
        score = sim

    order = np.argsort(-score)[:k]
    results = [
        {
            "path": data["paths"][ids[o]],
            "score": float(score[o]),
            "colour": dominant_colour(data["colors"][ids[o]], color_names),
            "folder": str(data["folders"][ids[o]]),
        }
        for o in order
    ]
    return results, {"colour": colour, "folders": folders}


def suggest_bottom_wear(data, event, exclude=None):
    folders = data["cfg"]["EVENT_BOTTOM_RULES"][event]
    pool = [(f, p) for f in folders for p in data["bottoms"].get(f, [])]
    if exclude and len(pool) > 1:
        pool = [x for x in pool if x[1] != exclude]
    if not pool:
        return None, None
    folder, path = random.choice(pool)
    return path, folder


# ----------------------------------------------------------------------------
# User interface
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Wardrobe Assistant", page_icon="👗", layout="centered")
    st.title("👗 Wardrobe Assistant")
    st.caption("Describe your top wear, pick an event, and get a full outfit.")

    if not os.path.exists(DATA_ZIP):
        st.error("data.zip not found. Run Section 12 in the Colab notebook and upload data.zip next to app.py.")
        st.stop()

    data = load_data()
    events = list(data["cfg"]["EVENT_BOTTOM_RULES"].keys())

    with st.form("search"):
        description = st.text_input("Top wear", placeholder="e.g. white t shirt, blue shirt")
        event = st.selectbox("Event type", events)
        submitted = st.form_submit_button("Get outfit", use_container_width=True)

    if submitted:
        if not description.strip():
            st.warning("Please describe your top wear first.")
        else:
            results, info = predict_top_wear(data, description, event)
            bottom_path, bottom_folder = suggest_bottom_wear(data, event)
            st.session_state["outfit"] = {
                "description": description,
                "event": event,
                "results": results,
                "info": info,
                "bottom_path": bottom_path,
                "bottom_folder": bottom_folder,
            }

    outfit = st.session_state.get("outfit")
    if not outfit:
        return

    if not outfit["results"]:
        st.error("No matching top wear found in the collection.")
        return

    info = outfit["info"]
    if info["colour"] is None:
        st.info("No colour recognised in your description, showing the most typical match.")

    best = outfit["results"][0]
    st.subheader(f"{outfit['event'].title()} outfit")
    c1, c2 = st.columns(2)
    with c1:
        st.image(read_image(best["path"]), use_container_width=True)
        st.markdown(f"**Top wear**  \n{best['folder'].title()} · {best['colour']}")
    with c2:
        if outfit["bottom_path"]:
            st.image(read_image(outfit["bottom_path"]), use_container_width=True)
            st.markdown(f"**Bottom wear**  \n{outfit['bottom_folder'].title()}")
        else:
            st.warning("No bottom wear images found for this event.")

    if st.button("🔀 Shuffle bottom wear", use_container_width=True):
        p, f = suggest_bottom_wear(data, outfit["event"], exclude=outfit["bottom_path"])
        outfit["bottom_path"], outfit["bottom_folder"] = p, f
        st.rerun()

    if len(outfit["results"]) > 1:
        with st.expander("Other top wear matches"):
            cols = st.columns(len(outfit["results"]) - 1)
            for col, r in zip(cols, outfit["results"][1:]):
                with col:
                    st.image(read_image(r["path"]), use_container_width=True)
                    st.caption(f"{r['colour']} · score {r['score']:.2f}")


if __name__ == "__main__":
    main()
