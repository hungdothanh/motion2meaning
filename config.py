#------------------config.py------------------

PRETRAINED_MODEL_PATH = 'weights/saved_weight.pth'      
SEGMENT_LENGTH = 1000  # 10 seconds at 100Hz 
NUM_CLASSES = 4  # Healthy, Stage 2, Stage 2.5, Stage 3
CLASS_NAMES = ["Healthy","Stage 2","Stage 2.5","Stage 3"]
SENSOR_NAMES = [f"Left VGRF-{i}" for i in range(1,9)] + [f"Right VGRF-{i}" for i in range(1,9)]



css = """
.container {
    max-width: 900px;
    margin: auto;
}

/* bump up the base font size of everything */
.gradio-container, .gradio-container * {
  font-size: 1.125rem !important;  /* 18px if root is 16px */
}

/* if you want even larger labels or dropdown items: */
.gradio-container label,
.gradio-container .block-label,
.gradio-container .gr-dropdown .dropdown,
.gradio-container .gr-radio .radio,
.gradio-container .gr-checkbox .checkbox,
.gradio-container .gr-button {
  font-size: 1.25rem !important;   /* 20px */
}

# -------------------
/* Reduce vertical spacing between rows */
.gr-row {
    margin-top: 0px;
    margin-bottom: 0px;
}

h1 {
    text-align: center;
    display:block;
    font-size: 48px !important;
    font-weight: 900;
}


label, .block-label{
    font-weight: 790;
    font-size: 13px;
}

select, option{
    font-size: 16px;
    font-weight: 800;
}

textarea {
    font-size: 14px;
    font-weight: 500;
}

/* tab text */
button {
    font-size: 14px;
    font-weight: 550;
}

.gr-button, .gr-textbox, .gr-dropdown {
    font-size: 16px;
    font-weight: 800;
}

/* Blue-themed container for gait parameter boxes */
.gait-param-box {
    background: #fff;
    border: 1px solid #bfdbfe;  /* soft blue border */
    border-radius: 12px;
    padding: 12px;
    margin-top: -8px;
    margin-left: 0px;
    margin-right: 6px;
    mArgin-bottom: -4px;
    width: 100%;
    box-shadow: 0 4px 10px rgba(161, 196, 208, 0.2);  /* subtle blue shadow */
}

.icon-button {
    background-color: rgba(60, 130, 200, 0.1);
    border: 2px solid #3C82C8;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    padding: 0;
    font-size: 18px;
    line-height: 28px;
    text-align: center;
    transition: background-color 0.2s ease, transform 0.1s ease;
}
.icon-button:hover {
    background-color: rgba(60, 130, 200, 0.2);
    transform: scale(1.1);
}
.icon-button:active {
    background-color: rgba(60, 130, 200, 0.3);
    transform: scale(0.95);
}
"""


js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""
