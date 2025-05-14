import streamlit as st
import pandas as pd
import logging
import re
from PIL import Image
from ultralytics import YOLO
try:
    from rapidfuzz import process, fuzz
    USE_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    USE_RAPIDFUZZ = False

# — Silence the torch file‐watcher errors —
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# — Page config —
st.set_page_config(page_title="Smart Cashier", layout="wide")

# — Helpers —
def normalize(name: str) -> str:
    return re.sub(r"\W+", "", name).lower()

# — Load & cache YOLO detection model —
@st.cache_resource
def load_model():
    m = YOLO("best.pt")
    try:
        m.fuse()
    except Exception:
        pass
    return m

# — Load & cache inventory —
@st.cache_data
def load_inventory(path="database.csv"):
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", engine="python")
    cols = df.columns.tolist()
    prod_col = "product_name" if "product_name" in cols else next(c for c in cols if "name" in c.lower())
    price_col = "price" if "price" in cols else next(c for c in cols if "price" in c.lower())
    stock_col = "stock" if "stock" in cols else next(c for c in cols if "avail" in c.lower())
    df = df.rename(columns={prod_col: "product_name", price_col: "price", stock_col: "stock"})
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).astype(float)
    df["normalized"] = df["product_name"].apply(normalize)
    return df

# — Save inventory —
def save_inventory(df, path="database.csv"):
    df.drop(columns="normalized", errors="ignore").to_csv(path, index=False)

# — Initialize —
model = load_model()
inventory = load_inventory()
choices = inventory["product_name"].tolist()

# — UI —
st.title("🛒 Smart Cashier")
mode = st.sidebar.radio("Input mode:", ["Upload Images", "Camera"])

# Acquire images
images = []
if mode == "Upload Images":
    ups = st.file_uploader("Upload product images:", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if ups:
        images = [Image.open(u).convert("RGB") for u in ups]
else:
    sn = st.camera_input("Snap a photo of your products")
    if sn:
        images = [Image.open(sn).convert("RGB")]

if images:
    col_img, col_inv = st.columns([3, 2])
    detected_items = []
    with col_img:
        for idx, img in enumerate(images, 1):
            st.subheader(f"Detections #{idx}")
            st.image(img, caption=f"Image #{idx}", use_container_width=True)
            with st.spinner(f"Detecting items in image {idx}..."):
                res = model(img, conf=0.2, iou=0.3, imgsz=1280)[0]
            ann = res.plot()
            st.image(ann, caption="Annotated", use_container_width=True)
            confs = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            names = res.names
            for cls_i, conf in zip(classes, confs):
                prod = names[int(cls_i)]
                detected_items.append((prod, float(conf)))

    st.subheader("🛠️ Confirm Detections")
    final_items = []
    for i, (prod, conf) in enumerate(detected_items):
        key = f"det_{i}"
        label = f"{prod} ({conf:.2f})"
        checked = st.checkbox(label, value=True, key=key)
        if checked:
            final_items.append(prod)

    # Aggregate counts
    counts = {}
    for prod in final_items:
        counts[prod] = counts.get(prod, 0) + 1

    # Build invoice
    invoice = []
    grand_total = 0.0
    for name, qty in counts.items():
        mask = inventory["normalized"] == normalize(name)
        row = inventory[mask]
        matched = name
        score = 0
        if row.empty:
            if USE_RAPIDFUZZ:
                m = process.extractOne(name, choices, scorer=fuzz.WRatio)
                if m:
                    matched, score, _ = m
            else:
                nn = normalize(name)
                best = (None, 0)
                for choice in choices:
                    s = SequenceMatcher(None, nn, normalize(choice)).ratio()
                    if s > best[1]:
                        best = (choice, s)
                matched, score = best
            threshold = 60 if USE_RAPIDFUZZ else 0.5
            if score < threshold:
                st.warning(f"No good match for '{name}'. Assuming SAR 0.00.")
                invoice.append({"Product": name, "Unit Price": "SAR 0.00", "Qty": qty, "Total": "SAR 0.00"})
                continue
            row = inventory[inventory["product_name"] == matched]

        price = float(row.iloc[0]["price"])
        line_total = price * qty
        grand_total += line_total
        invoice.append({
            "Product": matched,
            "Unit Price": f"SAR {price:.2f}",
            "Qty": qty,
            "Total": f"SAR {line_total:.2f}"
        })

        # Display invoice
    with col_inv:
        st.subheader("🧾 Invoice")
        # 1-based numbering for invoice rows
        df_inv = pd.DataFrame(invoice)
        df_inv.index = df_inv.index + 1  # start at 1
        df_inv.index.name = "No."
        st.table(df_inv)
        st.metric("Grand Total", f"SAR {grand_total:.2f}")
        if st.button("✅ Confirm & Update Inventory"):
            for it in invoice:
                idx = inventory.index[inventory["product_name"] == it["Product"]]
                if not idx.empty:
                    inventory.at[idx[0], "stock"] -= it["Qty"]
            save_inventory(inventory)
            st.success("Inventory updated!")
        with st.expander("Remaining Stock"):
            st.dataframe(inventory.drop(columns="normalized"))
