import io
import time

import cv2
import fitz  # PyMuPDF
import numpy as np
import streamlit as st

st.set_page_config(page_title="PDF Noise Remover", layout="wide")

# =====================================================================
# CORE CLEANING ALGORITHM
# =====================================================================
def clean_page_image(
    img,
    block_size=31,
    c_val=15,
    bilateral_d=9,
    margin_top_pct=0.000,
    margin_bottom_pct=0.000,
    margin_left_pct=0.010,
    margin_right_pct=0.026,
    smart_top_pct=0.020,
    smart_bottom_pct=0.020,
    smart_left_pct=0.050,
    smart_right_pct=0.065,
):
    """Cleans a page image with independent margin controls for each edge."""
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if bilateral_d > 0:
        smoothed = cv2.bilateralFilter(gray, d=bilateral_d, sigmaColor=75, sigmaSpace=75)
    else:
        smoothed = gray

    thresh = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_val
    )

    my_top = int(h * margin_top_pct)
    my_bottom = int(h * margin_bottom_pct)
    mx_left = int(w * margin_left_pct)
    mx_right = int(w * margin_right_pct)

    if my_top > 0:
        thresh[0:my_top, :] = 255
    if my_bottom > 0:
        thresh[h - my_bottom:h, :] = 255
    if mx_left > 0:
        thresh[:, 0:mx_left] = 255
    if mx_right > 0:
        thresh[:, w - mx_right:w] = 255

    inverted = cv2.bitwise_not(thresh)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    bx_left = int(w * smart_left_pct)
    bx_right = int(w * smart_right_pct)
    by_top = int(h * smart_top_pct)
    by_bottom = int(h * smart_bottom_pct)

    for label in range(1, num_labels):
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        right = left + width
        bottom = top + height

        is_in_left_margin = right < bx_left
        is_in_right_margin = left > (w - bx_right)
        is_in_top_margin = bottom < by_top
        is_in_bottom_margin = top > (h - by_bottom)

        if is_in_left_margin or is_in_right_margin or is_in_top_margin or is_in_bottom_margin:
            if area < (w * h) * 0.02 or (width < w * 0.08 and height < h * 0.08):
                thresh[labels == label] = 255
                continue

        near_left = left < bx_left
        near_right = right > (w - bx_right)
        near_top = top < by_top
        near_bottom = bottom > (h - by_bottom)

        if near_left or near_right or near_top or near_bottom:
            touches_edge = (left <= 2) or (right >= w - 2) or (top <= 2) or (bottom >= h - 2)
            is_noise = False
            if touches_edge:
                if width > w * 0.03 or height > h * 0.03 or area > 300:
                    is_noise = True
            else:
                if area > 1000 or width > w * 0.06 or height > h * 0.06:
                    is_noise = True
            if is_noise:
                thresh[labels == label] = 255

    return thresh


def render_page_bgr(page, matrix):
    pix = page.get_pixmap(matrix=matrix)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def clean_pdf_bytes(input_bytes, start_page, end_page, dpi, params, progress_cb=None):
    doc = fitz.open(stream=input_bytes, filetype="pdf")
    total_pages = len(doc)
    start_idx = max(0, start_page - 1)
    end_idx = min(total_pages, end_page if end_page else total_pages)

    out_doc = fitz.open()
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    n = max(1, end_idx - start_idx)

    first_before, first_after = None, None

    for i, idx in enumerate(range(start_idx, end_idx)):
        page = doc[idx]
        img = render_page_bgr(page, matrix)
        cleaned = clean_page_image(img, **params)

        if first_before is None:
            first_before, first_after = img, cleaned

        ok, buf = cv2.imencode(".png", cleaned)
        if not ok:
            continue

        rect = page.rect
        out_page = out_doc.new_page(width=rect.width, height=rect.height)
        out_page.insert_image(rect, stream=buf.tobytes())

        if progress_cb:
            progress_cb((i + 1) / n)

    out_bytes = out_doc.write(garbage=3, deflate=True)
    doc.close()
    out_doc.close()
    return out_bytes, total_pages, first_before, first_after


def build_comparison_pdf(orig_bytes, clean_bytes, start_page, end_page, dpi, progress_cb=None):
    doc_orig = fitz.open(stream=orig_bytes, filetype="pdf")
    doc_clean = fitz.open(stream=clean_bytes, filetype="pdf")

    total_pages = min(len(doc_orig), len(doc_clean))
    start_idx = max(0, start_page - 1)
    end_idx = min(total_pages, end_page if end_page else total_pages)
    n = max(1, end_idx - start_idx)

    doc_compare = fitz.open()
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for i, idx in enumerate(range(start_idx, end_idx)):
        img_o = render_page_bgr(doc_orig[idx], matrix)
        img_c = render_page_bgr(doc_clean[idx], matrix)

        h_o, w_o = img_o.shape[:2]
        h_c, w_c = img_c.shape[:2]
        if h_o != h_c:
            new_w = int(w_c * h_o / h_c)
            img_c = cv2.resize(img_c, (new_w, h_o), interpolation=cv2.INTER_CUBIC)

        combined = np.hstack((img_o, img_c))
        ok, buf = cv2.imencode(".png", combined)
        if not ok:
            continue

        ch, cw = combined.shape[:2]
        page_w = cw * 72.0 / dpi
        page_h = ch * 72.0 / dpi

        cp = doc_compare.new_page(width=page_w, height=page_h)
        cp.insert_image(fitz.Rect(0, 0, page_w, page_h), stream=buf.tobytes())

        if progress_cb:
            progress_cb((i + 1) / n)

    out_bytes = doc_compare.write(garbage=3, deflate=True)
    doc_orig.close()
    doc_clean.close()
    doc_compare.close()
    return out_bytes


# =====================================================================
# STREAMLIT UI
# =====================================================================
st.title("📄 PDF Scan Noise Remover")
st.caption("Removes scanning noise, edge shadows and border artifacts from scanned PDFs.")

uploaded_file = st.file_uploader("Upload a scanned PDF", type=["pdf"])

with st.sidebar:
    st.header("Page range")
    start_page = st.number_input("Start page", min_value=1, value=1, step=1)
    process_all = st.checkbox("Process all pages", value=True)
    end_page = None
    if not process_all:
        end_page = st.number_input("End page", min_value=1, value=10, step=1)

    st.header("Quality")
    # Changed default DPI to 150 to prevent memory crashes on Streamlit Cloud
    dpi = st.slider("DPI", min_value=100, max_value=400, value=150, step=10) 

    with st.expander("Advanced cleaning parameters"):
        block_size = st.slider("Threshold block size (odd)", 3, 99, 31, step=2)
        c_val = st.slider("Threshold C value", 0, 40, 15)
        bilateral_d = st.slider("Bilateral filter diameter (0 = off)", 0, 20, 9)
        st.markdown("**Absolute border crop (%)**")
        margin_top_pct = st.slider("Top margin", 0.0, 0.10, 0.000, step=0.005)
        margin_bottom_pct = st.slider("Bottom margin", 0.0, 0.10, 0.000, step=0.005)
        margin_left_pct = st.slider("Left margin", 0.0, 0.10, 0.010, step=0.005)
        margin_right_pct = st.slider("Right margin", 0.0, 0.10, 0.026, step=0.005)
        st.markdown("**Smart shadow-detection zones (%)**")
        smart_top_pct = st.slider("Smart top zone", 0.0, 0.20, 0.020, step=0.005)
        smart_bottom_pct = st.slider("Smart bottom zone", 0.0, 0.20, 0.020, step=0.005)
        smart_left_pct = st.slider("Smart left zone", 0.0, 0.20, 0.050, step=0.005)
        smart_right_pct = st.slider("Smart right zone", 0.0, 0.20, 0.065, step=0.005)

    make_comparison = st.checkbox("Also generate side-by-side comparison PDF", value=False)

params = dict(
    block_size=block_size,
    c_val=c_val,
    bilateral_d=bilateral_d,
    margin_top_pct=margin_top_pct,
    margin_bottom_pct=margin_bottom_pct,
    margin_left_pct=margin_left_pct,
    margin_right_pct=margin_right_pct,
    smart_top_pct=smart_top_pct,
    smart_bottom_pct=smart_bottom_pct,
    smart_left_pct=smart_left_pct,
    smart_right_pct=smart_right_pct,
)

if uploaded_file is not None:
    input_bytes = uploaded_file.read()

    if st.button("🧹 Clean PDF", type="primary"):
        progress_bar = st.progress(0.0, text="Starting...")

        def cb(frac):
            progress_bar.progress(frac, text=f"Cleaning pages... {int(frac * 100)}%")

        t0 = time.time()
        cleaned_bytes, total_pages, before_img, after_img = clean_pdf_bytes(
            input_bytes, start_page, end_page, dpi, params, progress_cb=cb
        )
        elapsed = time.time() - t0
        progress_bar.progress(1.0, text="Done")

        st.success(f"Processed in {elapsed:.1f}s. Total pages in source PDF: {total_pages}")

        if before_img is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Before (page 1 of range)")
                # Updated parameter from use_container_width to width="stretch"
                st.image(cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB), width="stretch")
            with col2:
                st.subheader("After")
                # Updated parameter from use_container_width to width="stretch"
                st.image(after_img, width="stretch")

        st.download_button(
            "⬇️ Download cleaned PDF",
            data=cleaned_bytes,
            file_name="cleaned.pdf",
            mime="application/pdf",
        )

        if make_comparison:
            with st.spinner("Building comparison PDF..."):
                comp_bytes = build_comparison_pdf(
                    input_bytes, cleaned_bytes, start_page, end_page, dpi
                )
            st.download_button(
                "⬇️ Download side-by-side comparison PDF",
                data=comp_bytes,
                file_name="comparison.pdf",
                mime="application/pdf",
            )
else:
    st.info("Upload a scanned PDF to get started.")
