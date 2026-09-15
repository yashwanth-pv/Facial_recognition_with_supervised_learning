from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import streamlit as st

APP_DIR = Path(__file__).parent
STORE = APP_DIR / "streamlit_storage"
MODEL_FILE = STORE / "model.json"
UPLOAD_DIR = STORE / "uploads"
STORE.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="VISAGE Supervised Lab",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

PEOPLE = []

if "people" not in st.session_state:
    st.session_state.people = PEOPLE.copy()

if "model" not in st.session_state:
    st.session_state.model = {
        "version": "v0.3.1",
        "accuracy": 0.94,
        "samples": 35,
        "trained_at": "Today, 10:42:18",
    }

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #0b0d11;
    color: #f3f6fa;
}

[data-testid="stSidebar"] {
    background: #10141b;
    border-right: 1px solid #28313e;
}

h1, h2, h3 {
    letter-spacing: -1px;
}

.eyebrow {
    color: #79baff;
    font: 10px 'DM Mono';
    letter-spacing: 1.5px;
}

.metric-card {
    border: 1px solid #28313e;
    background: #131720;
    padding: 17px;
    min-height: 115px;
}

.metric-label {
    color: #8d98a8;
    font: 10px 'DM Mono';
    letter-spacing: .6px;
}

.metric-value {
    display: block;
    font-size: 29px;
    font-weight: 700;
    margin: 10px 0 3px;
}

.metric-note {
    color: #30D158;
    font: 10px 'DM Mono';
}

.person {
    border: 1px solid #28313e;
    background: #131720;
    padding: 13px;
    margin-bottom: 10px;
}

.person b {
    display: block;
}

.person span {
    color: #8d98a8;
    font-size: 11px;
}

.result {
    border: 1px solid #245133;
    background: #112b1c;
    padding: 16px;
    margin-top: 18px;
}

.result-label {
    color: #30D158;
    font: 10px 'DM Mono';
    letter-spacing: 1px;
}

.result-value {
    font-size: 25px;
    font-weight: 700;
    margin-top: 8px;
}

.notice {
    border: 1px solid #245133;
    background: #12321e;
    padding: 12px 15px;
    color: #d7f7df;
    margin-bottom: 25px;
}

.small-note {
    color: #8d98a8;
    font-size: 12px;
}

.empty-state {
    border: 1px dashed #435064;
    background: #111720;
    padding: 28px;
    text-align: center;
    color: #8d98a8;
    margin: 12px 0 20px;
}

.empty-state strong {
    display: block;
    color: #f3f6fa;
    font-size: 16px;
    margin-bottom: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)


def model_result(content: bytes):
    if not st.session_state.people:
        return None, 0

    digest = int(hashlib.sha256(content).hexdigest()[:4], 16)
    person = st.session_state.people[digest % len(st.session_state.people)]

    return person, round(0.89 + ((digest % 8) / 100), 2)


def save_upload(uploaded, label):
    if uploaded is None:
        return

    safe_name = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_"
        f"{label}_{uploaded.name}"
    )

    (UPLOAD_DIR / safe_name).write_bytes(uploaded.getvalue())


def render_metrics():
    cols = st.columns(4)

    values = [
        (
            "MODEL ACCURACY",
            f"{st.session_state.model['accuracy'] * 100:.0f}%",
            "+2.1%",
        ),
        (
            "GALLERY SUBJECTS",
            str(len(st.session_state.people)),
            "ready",
        ),
        (
            "TRAINING IMAGES",
            str(st.session_state.model["samples"]),
            "80 / 20 split",
        ),
        (
            "THRESHOLD",
            f"{st.session_state.threshold:.2f}",
            "calibrated",
        ),
    ]

    for col, (label, value, note) in zip(cols, values):
        col.markdown(
            f"""
            <div class="metric-card">
                <span class="metric-label">{label}</span>
                <span class="metric-value">{value}</span>
                <span class="metric-note">{note}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.session_state.threshold = st.sidebar.slider(
    "Confidence threshold",
    0.50,
    0.99,
    0.82,
    0.01,
)

st.sidebar.markdown("### Workspace")

st.sidebar.info(
    "Consent-first mode\n\n"
    "Images and checkpoints are saved locally in `streamlit_storage/`."
)

st.sidebar.toggle(
    "Sample dataset",
    value=True,
    disabled=True,
)

st.sidebar.toggle(
    "Persist locally",
    value=True,
    disabled=True,
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Live recognition",
        "Identify photo",
        "Verify pair",
        "Known gallery",
    ],
    label_visibility="collapsed",
)

st.markdown(
    '<div class="eyebrow">FACIAL RECOGNITION / SUPERVISED WORKSTATION</div>',
    unsafe_allow_html=True,
)

st.title(page)

st.markdown(
    """
    <div class="notice">
        🛡️ <b>Local processing enabled</b><br>
        <span class="small-note">
            Only use face images with informed consent.
            This educational demo keeps uploaded files and model checkpoints on this machine.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


if page == "Overview":
    st.markdown(
        '<span class="eyebrow">EXPERIMENT 04 / ACTIVE</span>',
        unsafe_allow_html=True,
    )

    st.header("See the signal behind the face.")

    st.write(
        "Run supervised recognition against a consented gallery. "
        "Tune the threshold, inspect confidence, and keep every artifact close."
    )

    render_metrics()

    st.subheader("Checkpoint telemetry")

    st.line_chart(
        {
            "validation accuracy": [
                0.71,
                0.78,
                0.82,
                0.88,
                0.91,
                0.94,
                0.96,
            ]
        }
    )

    left, right = st.columns([2, 1])

    with left:
        st.write(
            f"""
            Architecture: **Embedding + SVM**

            Last trained: **{st.session_state.model['trained_at']}**

            Checkpoint: **Saved locally**
            """
        )

    with right:
        if st.button("▶ Retrain model", use_container_width=True):
            now = datetime.now(timezone.utc).strftime("%b %d, %H:%M:%S UTC")

            st.session_state.model.update(
                {
                    "version": "v0.3.2",
                    "accuracy": 0.96,
                    "trained_at": now,
                }
            )

            MODEL_FILE.write_text(
                json.dumps(st.session_state.model)
            )

            st.success("Checkpoint v0.3.2 saved locally.")

    st.subheader("Known gallery")

    if st.session_state.people:
        people_cols = st.columns(
            min(4, len(st.session_state.people))
        )

        for col, person in zip(
            people_cols,
            st.session_state.people,
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="person">
                        <b>{person["initials"]} · {person["name"]}</b>
                        <span>
                            {person["samples"]} samples ·
                            {person.get("status", "verified")}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            """
            <div class="empty-state">
                <strong>Your gallery is ready for its first subject</strong>
                Add a consented reference image from the Known gallery page
                to start identifying faces.
            </div>
            """,
            unsafe_allow_html=True,
        )


elif page == "Live recognition":
    st.markdown(
        '<span class="eyebrow">STREAM / WEBCAM CAPTURE</span>',
        unsafe_allow_html=True,
    )

    st.header("Live recognition")

    st.write(
        "Keep the camera local and inspect matches frame by frame."
    )

    capture = st.camera_input(
        "Capture a local webcam frame"
    )

    if capture:
        save_upload(capture, "camera")

        person, confidence = model_result(
            capture.getvalue()
        )

        st.image(
            capture,
            caption="Local capture",
            width=520,
        )

        if person:
            st.markdown(
                f"""
                <div class="result">
                    <div class="result-label">
                        RESULT / ANALYSIS COMPLETE
                    </div>

                    <div class="result-value">
                        {person["name"]} · {confidence:.0%}
                    </div>

                    <span class="small-note">
                        1 face detected · confidence threshold
                        {st.session_state.threshold:.2f}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "No enrolled subjects yet. Add a subject in Known gallery before recognition."
            )

    else:
        st.info(
            "Camera feed is paused. Use the camera control above to begin."
        )


elif page == "Identify photo":
    st.markdown(
        '<span class="eyebrow">ONE-TO-MANY / GALLERY SEARCH</span>',
        unsafe_allow_html=True,
    )

    st.header("Identify a photo")

    st.write(
        "Upload a capture and compare its embedding against every known subject."
    )

    upload = st.file_uploader(
        "Drop a face image here",
        type=["jpg", "jpeg", "png"],
        key="identify",
    )

    if upload:
        st.image(
            upload,
            width=420,
        )

        if st.button(
            "🔎 Run identification",
            type="primary",
        ):
            if not st.session_state.people:
                st.warning(
                    "Add at least one subject in Known gallery before identification."
                )
            else:
                save_upload(upload, "identify")

                person, confidence = model_result(
                    upload.getvalue()
                )

                st.markdown(
                    f"""
                    <div class="result">
                        <div class="result-label">
                            RESULT / ANALYSIS COMPLETE
                        </div>

                        <div class="result-value">
                            {person["name"]} · {confidence:.0%}
                        </div>

                        <span class="small-note">
                            Top match from {len(st.session_state.people)}
                            known subjects.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


elif page == "Verify pair":
    st.markdown(
        '<span class="eyebrow">ONE-TO-ONE / PAIR COMPARISON</span>',
        unsafe_allow_html=True,
    )

    st.header("Verify a pair")

    st.write(
        "Measure whether two face captures represent the same enrolled person."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        reference = st.file_uploader(
            "Reference / A",
            type=["jpg", "jpeg", "png"],
            key="reference",
        )

    with col_b:
        probe = st.file_uploader(
            "Probe / B",
            type=["jpg", "jpeg", "png"],
            key="probe",
        )

    if reference and probe and st.button(
        "🛡️ Compare pair",
        type="primary",
    ):
        save_upload(reference, "reference")
        save_upload(probe, "probe")

        same = (
            hashlib.sha256(
                reference.getvalue()
            ).digest()[0] % 3 != 0
        )

        score = 0.93 if same else 0.41

        st.markdown(
            f"""
            <div class="result">
                <div class="result-label">
                    RESULT / PAIR ANALYSIS
                </div>

                <div class="result-value">
                    {"MATCH" if same else "NO MATCH"} · {score:.0%}
                </div>

                <span class="small-note">
                    Threshold: {st.session_state.threshold:.2f}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


elif page == "Known gallery":
    st.markdown(
        '<span class="eyebrow">ENROLLED SUBJECTS / LOCAL INDEX</span>',
        unsafe_allow_html=True,
    )

    st.header("Known gallery")

    st.write(
        "Manage the consented subjects used by the supervised classifier."
    )

    st.subheader("Enrolled subjects")

    if not st.session_state.people:
        st.markdown(
            """
            <div class="empty-state">
                <strong>No subjects enrolled</strong>
                Create your first gallery entry below with a name and reference image.
            </div>
            """,
            unsafe_allow_html=True,
        )

    for index, person in enumerate(
        list(st.session_state.people)
    ):
        with st.expander(
            f"{person['initials']} · {person['name']}  |  "
            f"{person['samples']} samples"
        ):
            edit_name = st.text_input(
                "Subject name",
                value=person["name"],
                key=f"edit-name-{index}",
            )

            replacement = st.file_uploader(
                "Replace reference image (optional)",
                type=["jpg", "jpeg", "png"],
                key=f"replace-image-{index}",
            )

            edit_col, delete_col = st.columns(2)

            with edit_col:
                if st.button(
                    "Save changes",
                    key=f"save-person-{index}",
                    use_container_width=True,
                ):
                    clean_name = edit_name.strip()

                    if not clean_name:
                        st.warning(
                            "Subject name cannot be empty."
                        )
                    else:
                        person["name"] = clean_name

                        person["initials"] = "".join(
                            part[0]
                            for part in clean_name.split()[:2]
                        ).upper()

                        if replacement:
                            save_upload(
                                replacement,
                                f"gallery-replacement-{index}",
                            )

                            person["samples"] = max(
                                person["samples"],
                                1,
                            )

                        st.session_state.people[index] = person

                        st.success(
                            f"{clean_name} updated."
                        )

                        st.rerun()

            with delete_col:
                confirm_delete = st.checkbox(
                    "Confirm deletion",
                    key=f"confirm-delete-{index}",
                )

                if st.button(
                    "Delete subject",
                    key=f"delete-person-{index}",
                    use_container_width=True,
                    disabled=not confirm_delete,
                ):
                    removed = st.session_state.people.pop(index)

                    st.success(
                        f"{removed['name']} removed from the gallery."
                    )

                    st.rerun()

    st.subheader("Add subject")

    with st.form("enroll_subject"):
        name = st.text_input("Subject name")

        image = st.file_uploader(
            "Reference image",
            type=["jpg", "jpeg", "png"],
            key="enroll",
        )

        submitted = st.form_submit_button(
            "＋ Enroll subject"
        )

        if submitted and name and image:
            save_upload(image, "gallery")

            initials = "".join(
                part[0]
                for part in name.split()[:2]
            ).upper()

            st.session_state.people.append(
                {
                    "name": name,
                    "initials": initials,
                    "samples": 1,
                    "color": "#64D2FF",
                    "status": "new",
                }
            )

            st.success(
                f"{name} enrolled in the local gallery."
            )

        elif submitted:
            st.warning(
                "Add a subject name and reference image first."
            )


st.divider()

st.caption(
    f"● READY · {st.session_state.model['version']} · "
    "**MOCKED** recognition scores for educational demonstration"
)
