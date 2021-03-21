import base64
from math import radians

import numpy as np
import streamlit as st
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset

import tempfile
import datetime


def make_square(im):
    x, y = im.size
    maximum = max(x, y)
    size = int(np.ceil(np.sqrt(2 * maximum ** 2)))
    new_im = Image.new('L', (size, size))
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def angles_to_coords(image, angles):
    angles_radians = [radians(angle) for angle in (angles - 90)]
    center = (np.array(image.shape) / 2)[0]
    r = image.shape[0] // 2
    x = r * np.cos(angles_radians) + center
    y = r * np.sin(angles_radians) + center
    coords = np.floor(np.array(list(zip(x, y)))).astype(int)
    return coords


def calculate_coords(image, angle, detector_number, span, detector=False):
    angles = (np.linspace(0, span, detector_number) + angle - 1 / 2 * span) % 360
    if detector:
        angles = ((angles + 180) % 360)[::-1]

    coords = angles_to_coords(image, angles)
    return coords


def bresenham(x1, y1, x2, y2):
    swap = False
    if abs(y2 - y1) > abs(x2 - x1):
        swap = True
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    dx = x2 - x1
    dy = y2 - y1
    m = dy / dx if dx != 0 else 1
    q = y1 - m * x1
    if x1 < x2:
        xs = np.arange(np.floor(x1), np.ceil(x2) + 1, 1, dtype=int)
    else:
        xs = np.arange(np.ceil(x1), np.floor(x2) - 1, -1, dtype=int)
    ys = np.round(m * xs + q).astype(int)
    if swap:
        return (ys, xs)
    else:
        return (xs, ys)


def normalize(array):
    array = np.array(array)
    array = array.astype('float32')
    array -= np.min(array)
    array /= np.max(array)
    return array


def radon_transform(image, alpha, detector_number, span):
    scan_number = int(180 / alpha)
    angles = np.linspace(0, 180, scan_number)
    sinogram = np.zeros((scan_number, detector_number))
    for i, angle in enumerate(angles):
        emiter_coords = calculate_coords(image, angle, detector_number, span, detector=False)
        detector_coords = calculate_coords(image, angle, detector_number, span, detector=True)
        bresenham_lines = [bresenham(x1, y1, x2, y2) for (x1, y1), (x2, y2) in zip(emiter_coords, detector_coords)]
        results = []
        for line in bresenham_lines:
            results.append(np.sum(image[tuple(line)]))
        sinogram[i] = normalize(results)

    return np.swapaxes(sinogram, 0, 1)


def reshape_to_original(image, size):
    shape = image.shape[0]
    x = np.floor((shape / 2) - (size[1] / 2)).astype(int)
    y = np.floor((shape / 2) - (size[0] / 2)).astype(int)
    reshaped = image[y:y + size[0], x:x + size[1]]
    return reshaped


def inverse_radon_transform(sinogram, size, span):
    detector_number, scan_number = sinogram.shape
    sinogram = np.swapaxes(sinogram, 0, 1)
    pil_image = Image.new('L', size)
    pil_image = make_square(pil_image)
    image = np.array(pil_image).astype('float64')
    count = image.copy()
    angles = np.linspace(0, 180, scan_number)
    for i, angle in enumerate(angles):
        emiter_coords = calculate_coords(image, angle, detector_number, span, detector=False)
        detector_coords = calculate_coords(image, angle, detector_number, span, detector=True)
        bresenham_lines = [np.array(bresenham(x1, y1, x2, y2)) for (x1, y1), (x2, y2) in
                           zip(emiter_coords, detector_coords)]
        for j, line in enumerate(bresenham_lines):
            image[tuple(line)] += sinogram[i][j]
            count[tuple(line)] += 1
    count[count == 0] = 1
    image = normalize(image / count)
    image = reshape_to_original(image, size)
    return image


def download_link(object_to_download, download_filename, download_link_text):
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


if __name__ == '__main__':
    title = 'Tomography simulator'
    st.set_page_config(page_title=title, layout='centered')
    st.title(title)
    detectors = int(st.sidebar.text_input("Number of detectors", 30))  # detector_number
    alpha_step = float(st.sidebar.text_input("\u0394\u03B1 step", 15))  # alpha
    detectors_range = int(st.sidebar.text_input("Detectors range", 90))  # span
    file = st.file_uploader('Upload an image')
    regex_pattern = r'[A-Z][a-z]+'

    if file is not None:
        img = Image.open(file).convert('L')
        st.image(img, "Initial image in grayscale")
        image = make_square(img)
        st.image(img, "image in square")
        img_array = np.array(img)

        sinogram = radon_transform(img_array, alpha_step, detectors, detectors_range)

        PIL_image = Image.fromarray(np.uint8(sinogram * 255), 'L')

        st.image(PIL_image, "Sinogram")

        image_after = inverse_radon_transform(sinogram, (1024, 1024), detectors_range)

        PIL_image = Image.fromarray(np.uint8(image_after * 255), 'L')

        st.image(PIL_image, "Inverse radon")

        first_name = st.text_input("Patient's first name")
        last_name = st.text_input("Patient's last name")
        commentary = st.text_input("Medical commentary")

        suffix = '.dcm'
        filename_little_endian = tempfile.NamedTemporaryFile(suffix=suffix).name

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        file_meta.ImplementationClassUID = "1.2.3.4"

        ds = FileDataset(filename_little_endian, {},
                         file_meta=file_meta, preamble=b"\0" * 128)

        ds.PatientName = "Test^Firstname"
        ds.PatientID = "123456"

        ds.is_little_endian = True
        ds.is_implicit_VR = True

        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
        ds.ContentTime = timeStr

        ds.save_as(filename_little_endian)

        if st.button('Download'):
            tmp_download_link = download_link(filename_little_endian, "PLIK.dcm", "POBIERZ")
            st.markdown(tmp_download_link, unsafe_allow_html=True)

        st.text("zapisano")
