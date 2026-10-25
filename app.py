import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

# Page setup
st.set_page_config(page_title="Handy contact Saver", layout="wide")
st.title("📱 Handwritten Mobile Number Scanner & MNIST Explorer")
st.write("Write a 10-digit mobile number, tap any misclassified digit to teach the AI in real time, and save all participants into a single `.vcf` file!")

# Initialize dynamic keys in session_state so buttons can clear the canvas on demand
if 'multi_canvas_key' not in st.session_state:
    st.session_state['multi_canvas_key'] = 0
if 'single_canvas_key' not in st.session_state:
    st.session_state['single_canvas_key'] = 0
if 'participant_input_key' not in st.session_state:
    st.session_state['participant_input_key'] = 0

# 1. Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# 2. Create background image with 10 faint slot dividers
def create_10_slot_background(width=850, height=180):
    bg = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(bg)
    slot_w = width / 10.0
    for i in range(1, 10):
        x = int(i * slot_w)
        for y in range(15, height - 15, 12):
            draw.line([(x, y), (x, min(y + 6, height - 15))], fill=(45, 45, 45, 255), width=1)
    return bg

# 3. Convert a cropped 2D grayscale array into a centered 28x28 MNIST array
def crop_to_mnist_28x28(cropped_np):
    coords = np.argwhere(cropped_np > 60)
    if len(coords) < 5:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    tight = cropped_np[y0:y1, x0:x1]

    h, w = tight.shape
    max_dim = max(h, w)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    start_y = (max_dim - h) // 2
    start_x = (max_dim - w) // 2
    square[start_y:start_y + h, start_x:start_x + w] = tight

    resized_20 = Image.fromarray(square).resize((20, 20), Image.Resampling.BILINEAR)

    mnist_frame = np.zeros((28, 28), dtype=np.float32)
    mnist_frame[4:24, 4:24] = np.array(resized_20, dtype=np.float32)

    y_idx, x_idx = np.indices((28, 28))
    total_mass = np.sum(mnist_frame)
    if total_mass > 0:
        com_y = np.sum(y_idx * mnist_frame) / total_mass
        com_x = np.sum(x_idx * mnist_frame) / total_mass
        shift_y = int(np.clip(round(14.0 - com_y), -2, 2))
        shift_x = int(np.clip(round(14.0 - com_x), -2, 2))
        mnist_frame = np.roll(mnist_frame, shift_y, axis=0)
        mnist_frame = np.roll(mnist_frame, shift_x, axis=1)

    return mnist_frame / 255.0

# 4. Single-digit canvas preprocessor
def preprocess_drawn_image(canvas_rgba):
    img = Image.fromarray(canvas_rgba.astype('uint8'), 'RGBA').convert('L')
    img_np = np.array(img)
    if np.max(img_np) < 60:
        return None
    return crop_to_mnist_28x28(img_np)

# 5. Multi-Digit Segmentation with Valley-Splitting
def segment_multiple_digits(canvas_rgba):
    img = Image.fromarray(canvas_rgba.astype('uint8'), 'RGBA').convert('L')
    img_np = np.array(img)
    binary_ink = (img_np > 60).astype(np.uint8)
    if np.sum(binary_ink) == 0:
        return []

    col_ink_count = np.sum(binary_ink, axis=0)
    col_has_ink = col_ink_count > 1

    raw_segments = []
    in_digit = False
    start_col = 0

    for x, has_ink in enumerate(col_has_ink):
        if has_ink and not in_digit:
            in_digit = True
            start_col = x
        elif not has_ink and in_digit:
            in_digit = False
            if (x - start_col) >= 4:
                raw_segments.append((start_col, x))
    if in_digit and (len(col_has_ink) - start_col) >= 4:
        raw_segments.append((start_col, len(col_has_ink)))

    final_segments = []
    for (x0, x1) in raw_segments:
        slice_bin = binary_ink[:, x0:x1]
        rows_with_ink = np.where(np.sum(slice_bin, axis=1) > 0)[0]
        if len(rows_with_ink) == 0:
            continue
        h = rows_with_ink[-1] - rows_with_ink[0] + 1
        w = x1 - x0

        if w > max(38, int(h * 1.15)):
            est_digits = max(2, min(4, int(round(w / (h * 0.72)))))
            sub_w = w / est_digits
            curr_start = x0
            for d_idx in range(1, est_digits):
                target_cut = int(x0 + d_idx * sub_w)
                search_left = max(curr_start + 8, target_cut - int(sub_w * 0.35))
                search_right = min(x1 - 8, target_cut + int(sub_w * 0.35))
                if search_right > search_left:
                    valley_offset = np.argmin(col_ink_count[search_left:search_right])
                    cut_col = search_left + valley_offset
                else:
                    cut_col = target_cut
                final_segments.append((curr_start, cut_col))
                curr_start = cut_col
            final_segments.append((curr_start, x1))
        else:
            final_segments.append((x0, x1))

    digit_frames = []
    for (x0, x1) in final_segments:
        slice_np = img_np[:, x0:x1]
        frame_28 = crop_to_mnist_28x28(slice_np)
        if frame_28 is not None:
            digit_frames.append(frame_28)

    return digit_frames

# 6. Online Learning Function: Fine-tunes CNN on user correction + memory anchor batch
def learn_from_user_correction(cnn_model, X_anchor, y_anchor, sample_flat, correct_digit):
    cnn_model.train()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.0009)
    criterion = nn.CrossEntropyLoss()

    user_img = torch.tensor(sample_flat.reshape(1, 1, 28, 28), dtype=torch.float32)
    user_variations = []
    for sx in [-1, 0, 1]:
        for sy in [-1, 0, 1]:
            user_variations.append(torch.roll(user_img, shifts=(sy, sx), dims=(2, 3)))
    X_user_batch = torch.cat(user_variations, dim=0)
    y_user_batch = torch.tensor([correct_digit] * len(user_variations), dtype=torch.long)

    rand_idx = np.random.choice(len(y_anchor), size=64, replace=False)
    X_mem = torch.tensor(X_anchor[rand_idx].reshape(-1, 1, 28, 28), dtype=torch.float32)
    y_mem = torch.tensor(y_anchor[rand_idx], dtype=torch.long)

    X_combined = torch.cat([X_user_batch, X_mem], dim=0)
    y_combined = torch.cat([y_user_batch, y_mem], dim=0)

    for _ in range(7):
        optimizer.zero_grad()
        logits = cnn_model(X_combined)
        loss = criterion(logits, y_combined)
        loss.backward()
        optimizer.step()

    cnn_model.eval()
    torch.save(cnn_model.state_dict(), 'cnn_mnist.pth')

    row = np.concatenate([[correct_digit], (sample_flat * 255.0).astype(int)])
    pd.DataFrame([row]).to_csv(
        'user_feedback.csv', mode='a', header=not os.path.exists('user_feedback.csv'), index=False
    )

# 7. Helper to generate vCard (.vcf) text and parse existing contacts
def create_vcf_entry(name, phone, index_num):
    clean_name = name.strip() if name.strip() else f"Participant #{index_num}"
    return (
        "BEGIN:VCARD\n"
        "VERSION:3.0\n"
        f"FN:{clean_name}\n"
        f"TEL;TYPE=CELL:{phone}\n"
        "END:VCARD\n"
    )

def parse_saved_vcf(vcf_path='event_contacts.vcf'):
    contacts = []
    if not os.path.exists(vcf_path):
        return contacts
    with open(vcf_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    curr_name, curr_phone = "", ""
    for line in lines:
        line = line.strip()
        if line.startswith("FN:"):
            curr_name = line[3:]
        elif line.startswith("TEL"):
            curr_phone = line.split(":")[-1]
        elif line == "END:VCARD":
            contacts.append({"Name": curr_name, "Mobile Number": curr_phone})
            curr_name, curr_phone = "", ""
    return contacts

# 8. Load models and dataset
@st.cache_resource
def load_models_and_data():
    df = pd.read_csv('mnist_test.csv')
    y = df.iloc[:, 0].values.astype('int')
    X = df.iloc[:, 1:].values.astype('float32') / 255.0

    cnn = SimpleCNN()
    cnn.load_state_dict(torch.load('cnn_mnist.pth', map_location=torch.device('cpu')))
    cnn.eval()

    pca = joblib.load('pca_50.pkl')
    kmeans = joblib.load('kmeans_10_pca.pkl')
    return X, y, cnn, pca, kmeans

X, y, cnn_model, pca_model, kmeans_model = load_models_and_data()

# Sidebar Mode Switcher & Reset Controls
st.sidebar.header("🕹️ Demo Mode")
mode = st.sidebar.radio(
    "Choose Application Mode:",
    [
        "📱 Multi-Digit Mobile Number Scanner (.vcf)",
        "✍️ Single Digit Live Draw & Teach",
        "📂 Pick from MNIST Dataset"
    ]
)

if os.path.exists('user_feedback.csv'):
    n_learned = sum(1 for _ in open('user_feedback.csv'))
    st.sidebar.success(f"🧠 User Corrections Learned: **{n_learned}**")

saved_contacts = parse_saved_vcf('event_contacts.vcf')
st.sidebar.info(f"📇 Total Contacts in Master VCF: **{len(saved_contacts)}**")
if len(saved_contacts) > 0:
    with open('event_contacts.vcf', 'r', encoding='utf-8') as f:
        master_vcf_content = f.read()
    st.sidebar.download_button(
        label=f"⬇️ Download Master VCF ({len(saved_contacts)} Contacts)",
        data=master_vcf_content,
        file_name="event_contacts.vcf",
        mime="text/vcard",
        use_container_width=True
    )

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Reset & Clear Controls")
if st.sidebar.button("🧹 Clear Drawing Screen", use_container_width=True):
    st.session_state['multi_canvas_key'] += 1
    st.session_state['single_canvas_key'] += 1
    st.session_state['participant_input_key'] += 1
    st.session_state['selected_wrong_pos'] = 0
    st.rerun()

if st.sidebar.button("🗑️ Reset Saved VCF Contacts", use_container_width=True):
    if os.path.exists('event_contacts.vcf'):
        os.remove('event_contacts.vcf')
    st.session_state['multi_canvas_key'] += 1
    st.session_state['participant_input_key'] += 1
    st.toast("Cleared all saved VCF contacts!", icon="🗑️")
    st.rerun()

# ==============================================================================
# MODE 1: MULTI-DIGIT MOBILE NUMBER SCANNER & TOUCH-TO-TEACH
# ==============================================================================
if mode == "📱 Multi-Digit Mobile Number Scanner (.vcf)":
    top_col1, top_col2, top_col3 = st.columns([2.2, 1, 1])
    with top_col1:
        st.subheader("1. Write Your 10-Digit Mobile Number")
    with top_col2:
        if st.button("🧹 Clear Screen (Next Person)", type="secondary", use_container_width=True):
            st.session_state['multi_canvas_key'] += 1
            st.session_state['participant_input_key'] += 1
            st.session_state['selected_wrong_pos'] = 0
            st.toast("Screen cleared for the next participant!", icon="🧹")
            st.rerun()
    with top_col3:
        if st.button("🔄 Reset Session & Clear All", type="secondary", use_container_width=True):
            st.session_state['confirm_reset'] = True

    if st.session_state.get('confirm_reset', False):
        st.warning("⚠️ **What would you like to reset?**")
        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            if st.button("🧹 Only Clear Screen & Inputs", use_container_width=True):
                st.session_state['multi_canvas_key'] += 1
                st.session_state['participant_input_key'] += 1
                st.session_state['selected_wrong_pos'] = 0
                st.session_state['confirm_reset'] = False
                st.rerun()
        with r_col2:
            if st.button("🗑️ Wipe Saved VCF Contacts & Clear Screen", type="primary", use_container_width=True):
                if os.path.exists('event_contacts.vcf'):
                    os.remove('event_contacts.vcf')
                st.session_state['multi_canvas_key'] += 1
                st.session_state['participant_input_key'] += 1
                st.session_state['selected_wrong_pos'] = 0
                st.session_state['confirm_reset'] = False
                st.toast("All saved contacts wiped & screen reset!", icon="🔄")
                st.rerun()
        with r_col3:
            if st.button("✖️ Cancel", use_container_width=True):
                st.session_state['confirm_reset'] = False
                st.rerun()

    stroke_w = st.slider("Pen Thickness:", 7, 18, 11, key="multi_slider")

    multi_canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=stroke_w,
        stroke_color="#FFFFFF",
        background_color="#000000",
        background_image=create_10_slot_background(850, 180),
        height=180,
        width=850,
        drawing_mode="freedraw",
        return_image_data=True,
        key=f"multi_canvas_{st.session_state['multi_canvas_key']}"
    )

    if multi_canvas.image_data is not None:
        digit_imgs = segment_multiple_digits(multi_canvas.image_data)

        if len(digit_imgs) > 0:
            batch_tensor = torch.tensor(np.array(digit_imgs).reshape(-1, 1, 28, 28), dtype=torch.float32)
            with torch.no_grad():
                logits = cnn_model(batch_tensor)
                probs = torch.softmax(logits, dim=1).numpy()
                pred_digits = np.argmax(probs, axis=1)
                confidences = np.max(probs, axis=1) * 100

            flat_batch = np.array(digit_imgs).reshape(len(digit_imgs), -1)
            clusters = kmeans_model.predict(pca_model.transform(flat_batch))

            # --- NEW FILTER: IF > 10 DIGITS DETECTED, DROP THE LOWEST-CONFIDENCE SEGMENTS ---
            raw_detected_count = len(digit_imgs)
            dropped_info = []
            if raw_detected_count > 10:
                # Find indices sorted from lowest confidence to highest confidence
                sorted_by_conf = np.argsort(confidences)
                drop_indices = sorted_by_conf[:(raw_detected_count - 10)]
                for d_i in sorted(drop_indices):
                    dropped_info.append(f"Pos #{d_i+1} (Pred: {pred_digits[d_i]} at {confidences[d_i]:.1f}%)")

                # Keep only the top 10 highest-confidence segments in their original left-to-right order
                keep_indices = np.sort(sorted_by_conf[-10:])
                digit_imgs = [digit_imgs[i] for i in keep_indices]
                pred_digits = pred_digits[keep_indices]
                confidences = confidences[keep_indices]
                clusters = clusters[keep_indices]

            detected_number_str = "".join(str(d) for d in pred_digits)

            # Display Segmented Digits
            if raw_detected_count > 10:
                st.subheader(f"2. Segmented Digits (10 Kept | Auto-Ignored {raw_detected_count - 10} Low-Confidence Segment(s))")
                st.caption(f"🛡️ **Auto-Filtered Out Lowest Confidence:** {', '.join(dropped_info)}")
            else:
                st.subheader(f"2. Segmented Digits ({len(digit_imgs)} detected)")

            n_show = len(digit_imgs)
            cols = st.columns(n_show)
            for i in range(n_show):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(1.5, 1.5))
                    ax.imshow(digit_imgs[i], cmap='gray')
                    ax.set_title(f"#{i+1} Pred: {pred_digits[i]}\n({confidences[i]:.0f}% | C#{clusters[i]})", fontsize=8.5)
                    ax.axis('off')
                    st.pyplot(fig)

            st.markdown("---")

            # --- STEP 3: HUMAN-IN-THE-LOOP VERIFICATION & TOUCH-TO-CORRECT ---
            col_left, col_right = st.columns([1.3, 1.2])

            with col_left:
                st.subheader("3. Recognized Telephone Number")
                if len(detected_number_str) == 10:
                    st.success(f"## 📞 `{detected_number_str}` (10 Digits Complete!)")
                else:
                    st.warning(f"## 📞 `{detected_number_str}` ({len(detected_number_str)}/10 digits detected)")

                is_satisfied = st.radio(
                    "🤔 Is this recognized mobile number 100% correct?",
                    ["👍 Yes, it is correct!", "👎 No, tap a wrong digit to fix & teach AI"],
                    horizontal=True,
                    key=f"multi_satisfaction_{st.session_state['multi_canvas_key']}"
                )

            if is_satisfied == "👎 No, tap a wrong digit to fix & teach AI":
                st.info("👆 **Step A: Tap the misidentified digit below** (for example, tap `#9` if the 9th digit was wrong):")
                btn_cols = st.columns(n_show)

                if 'selected_wrong_pos' not in st.session_state or st.session_state['selected_wrong_pos'] >= n_show:
                    st.session_state['selected_wrong_pos'] = 0

                for i in range(n_show):
                    with btn_cols[i]:
                        is_selected = (st.session_state['selected_wrong_pos'] == i)
                        btn_type = "primary" if is_selected else "secondary"
                        if st.button(f"#{i+1}: {pred_digits[i]}", key=f"tap_digit_{i}", type=btn_type, use_container_width=True):
                            st.session_state['selected_wrong_pos'] = i
                            st.rerun()

                sel_idx = st.session_state['selected_wrong_pos']
                wrong_pred = int(pred_digits[sel_idx])

                fix_col1, fix_col2 = st.columns([1, 3])
                with fix_col1:
                    fig_z, ax_z = plt.subplots(figsize=(2, 2))
                    ax_z.imshow(digit_imgs[sel_idx], cmap='gray')
                    ax_z.set_title(f"Selected Digit #{sel_idx+1}\n(Currently Pred: {wrong_pred})", fontsize=9, color='crimson')
                    ax_z.axis('off')
                    st.pyplot(fig_z)

                with fix_col2:
                    st.markdown(f"#### 🎯 **Step B: What is the true number for Digit #{sel_idx+1}?**")
                    true_digit_choice = st.radio(
                        "Tap the correct digit (0–9):",
                        options=list(range(10)),
                        index=(8 if wrong_pred != 8 else 0),
                        horizontal=True,
                        key="touch_correct_digit"
                    )

                    if st.button(f"🚀 Teach AI that Digit #{sel_idx+1} is '{true_digit_choice}' & Update Number", type="primary"):
                        with st.spinner(f"Training CNN on your handwritten '{true_digit_choice}'..."):
                            learn_from_user_correction(
                                cnn_model, X, y, digit_imgs[sel_idx].flatten(), int(true_digit_choice)
                            )
                        st.toast(f"AI learned Digit #{sel_idx+1} is '{true_digit_choice}'! Updating prediction...", icon="✅")
                        st.rerun()

            with col_right:
                st.subheader("4. Save to Master VCF (`event_contacts.vcf`)")
                participant_name = st.text_input(
                    "Participant Name (Optional):",
                    placeholder=f"Participant #{len(saved_contacts) + 1}",
                    key=f"part_name_{st.session_state['participant_input_key']}"
                )

                save_col1, save_col2 = st.columns([1.4, 1])
                with save_col1:
                    if st.button("💾 Save Contact & Clear for Next", type="primary", use_container_width=True):
                        next_num = len(saved_contacts) + 1
                        vcf_entry = create_vcf_entry(participant_name, detected_number_str, next_num)
                        with open("event_contacts.vcf", "a", encoding="utf-8") as f:
                            f.write(vcf_entry)

                        st.session_state['multi_canvas_key'] += 1
                        st.session_state['participant_input_key'] += 1
                        st.session_state['selected_wrong_pos'] = 0
                        st.toast(f"Saved {detected_number_str} & cleared screen for next person!", icon="✅")
                        st.rerun()

                with save_col2:
                    if st.button("💾 Save Only (Keep Screen)", use_container_width=True):
                        next_num = len(saved_contacts) + 1
                        vcf_entry = create_vcf_entry(participant_name, detected_number_str, next_num)
                        with open("event_contacts.vcf", "a", encoding="utf-8") as f:
                            f.write(vcf_entry)
                        st.toast(f"Saved {detected_number_str} to Master VCF!", icon="✅")
                        st.rerun()
        else:
            st.info("👆 Write a 10-digit mobile number across the 10 slots above!")

    # Master VCF Table & Combined Download Button at bottom
    if len(saved_contacts) > 0:
        st.markdown("---")
        st.subheader(f"📇 Master Contact List (`event_contacts.vcf` — {len(saved_contacts)} Contacts Saved)")
        col_tbl, col_dl = st.columns([2, 1])
        with col_tbl:
            st.dataframe(pd.DataFrame(saved_contacts), use_container_width=True)
        with col_dl:
            with open('event_contacts.vcf', 'r', encoding='utf-8') as f:
                all_vcf_bytes = f.read()
            st.download_button(
                label=f"📲 Download ALL {len(saved_contacts)} Contacts (.vcf)",
                data=all_vcf_bytes,
                file_name="event_contacts.vcf",
                mime="text/vcard",
                type="primary",
                use_container_width=True
            )

# ==============================================================================
# MODE 2 & 3: SINGLE DIGIT DRAW & DATASET EXPLORER
# ==============================================================================
else:
    sample_flat = None
    true_label = None

    if mode == "✍️ Single Digit Live Draw & Teach":
        col_draw, col_prep, col_pred, col_clust = st.columns([1.2, 1, 1.5, 1.2])
        with col_draw:
            st.subheader("1. Draw Here (0–9)")
            if st.button("🧹 Clear Drawing Pad", use_container_width=True):
                st.session_state['single_canvas_key'] += 1
                st.rerun()
            stroke_width = st.slider("Brush Thickness:", 10, 28, 18)
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 1)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=240,
                width=240,
                drawing_mode="freedraw",
                return_image_data=True,
                key=f"single_canvas_{st.session_state['single_canvas_key']}"
            )
        if canvas_result.image_data is not None:
            processed_28x28 = preprocess_drawn_image(canvas_result.image_data)
            if processed_28x28 is not None:
                sample_flat = processed_28x28.flatten()
                with col_prep:
                    st.subheader("2. 28×28 Input")
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(processed_28x28, cmap='gray')
                    ax.set_title("Auto-Centered 28×28", fontsize=10)
                    ax.axis('off')
                    st.pyplot(fig)
            else:
                st.info("👆 Draw a digit inside the black box on the left!")

    else:
        filter_digit = st.sidebar.selectbox("Filter by True Digit Label:", ["Any Digit"] + list(range(10)))
        available_indices = np.arange(len(y)) if filter_digit == "Any Digit" else np.where(y == int(filter_digit))[0]

        if st.sidebar.button("🎲 Pick a Random Sample"):
            st.session_state['sample_idx'] = int(np.random.choice(available_indices))

        if 'sample_idx' not in st.session_state or st.session_state['sample_idx'] not in available_indices:
            st.session_state['sample_idx'] = int(available_indices[0])

        selected_idx = st.sidebar.number_input("Sample Index:", 0, len(y)-1, st.session_state['sample_idx'])
        sample_flat = X[selected_idx]
        true_label = y[selected_idx]

        col_prep, col_pred, col_clust = st.columns([1, 1.5, 1.2])
        with col_prep:
            st.subheader("1. Dataset Image")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(sample_flat.reshape(28, 28), cmap='gray')
            ax.set_title(f"Sample #{selected_idx} | True: {true_label}", fontsize=11)
            ax.axis('off')
            st.pyplot(fig)

    if sample_flat is not None:
        sample_tensor = torch.tensor(sample_flat.reshape(1, 1, 28, 28), dtype=torch.float32)
        with torch.no_grad():
            logits = cnn_model(sample_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_digit = int(np.argmax(probs))
            confidence = probs[pred_digit] * 100

        sample_pca = pca_model.transform(sample_flat.reshape(1, -1))
        assigned_cluster = int(kmeans_model.predict(sample_pca)[0])
        centroid_784 = pca_model.inverse_transform(kmeans_model.cluster_centers_[assigned_cluster].reshape(1, -1))

        with col_pred:
            st.subheader("Supervised (CNN)")
            if true_label is None:
                st.success(f"### 🎯 Predicted Digit: **{pred_digit}**\n**Confidence:** {confidence:.2f}%")
            elif pred_digit == true_label:
                st.success(f"### 🎯 Predicted: **{pred_digit}** ({confidence:.1f}%) — ✅")
            else:
                st.error(f"### ❌ Predicted: **{pred_digit}** ({confidence:.1f}%) | True: {true_label}")

            fig_prob, ax_prob = plt.subplots(figsize=(5, 2.3))
            bars = ax_prob.bar(range(10), probs * 100, color='steelblue')
            bars[pred_digit].set_color('forestgreen')
            ax_prob.set_xticks(range(10))
            ax_prob.set_xlabel("Digit Class (0–9)")
            ax_prob.set_ylabel("Confidence (%)")
            ax_prob.set_ylim(0, 105)
            st.pyplot(fig_prob)

            st.markdown("---")
            st.markdown("#### 🤔 Are you satisfied with this prediction?")
            feedback = st.radio(
                "Select an option:",
                ["👍 Yes, it's correct!", "👎 No, teach the system the right number"],
                horizontal=True,
                key="feedback_radio"
            )

            if feedback == "👎 No, teach the system the right number":
                correct_num = st.number_input(
                    "Type the correct digit you drew (0–9):",
                    min_value=0, max_value=9, value=(5 if pred_digit != 5 else 0), step=1
                )
                if st.button("🚀 Train & Update Model Now"):
                    with st.spinner(f"Teaching the CNN that this image is a '{correct_num}'..."):
                        learn_from_user_correction(cnn_model, X, y, sample_flat, int(correct_num))
                    st.toast("Model updated! Re-evaluating your drawing...", icon="✅")
                    st.rerun()

        with col_clust:
            st.subheader("Unsupervised (k-Means)")
            st.info(f"**Assigned Group:** Cluster #{assigned_cluster}")
            fig_c, ax_c = plt.subplots(figsize=(2.8, 2.8))
            ax_c.imshow(centroid_784.reshape(28, 28), cmap='magma')
            ax_c.set_title(f"Cluster #{assigned_cluster} Centroid", fontsize=10)
            ax_c.axis('off')
            st.pyplot(fig_c)