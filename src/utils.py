import base64
import streamlit as st
from pathlib import Path


def check_maintenance_status():
    if st.secrets.under_maintenance:
        maintenance_img = html_img_with_href(
            "images/under_maintenance.jpg",
            "",
            width="100%",
            height="100%",
        )
        st.write(maintenance_img, unsafe_allow_html=True)
        st.stop()


def footnote():
    st.write("&nbsp;")
    st.write("&copy; 2026 Enrique Amat [enriam.codes@gmail.com]")


def html_img_with_href(img_file, url, width, height):
    if url == "":
        url = None

    try:
        with Path(img_file).open("rb") as f:
            data = f.read()
            img = base64.b64encode(data).decode()
    except FileNotFoundError:
        html_code = "" if url is None else f'<a href="{url}">{url}</a>'

    else:
        if url is None:
            html_code = f"""<img src="data:image/png;base64,{img}"
                style="width:{width}; height:{height}"/>"""
        else:
            html_code = f"""
            <a href="{url}">
                <img src="data:image/png;base64,{img}"
                style="width:{width}; height:{height}"/>
            </a>"""

    return html_code
