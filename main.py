import base64
import datetime
import os
import tempfile
import time
from math import radians, pi

import numpy as np
import pydicom
import streamlit as st
from PIL import Image
from pydicom._storage_sopclass_uids import MRImageStorage
from pydicom.dataset import Dataset, FileDataset, validate_file_meta
from pydicom.uid import generate_uid


# image padding - so that emitters can go in circle around it
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


def radon_transform(image, alpha, detector_number, span, img_arr, steps):
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
        if steps and i % 5 == 0:
            img_arr.append(Image.fromarray(np.uint8(sinogram.copy() * 255), 'L'))
    return sinogram


def reshape_to_original(image, size):
    shape = image.shape[0]
    x = np.floor((shape / 2) - (size[1] / 2)).astype(int)
    y = np.floor((shape / 2) - (size[0] / 2)).astype(int)
    reshaped = image[y:y + size[0], x:x + size[1]]
    return reshaped


def tresh(array, min, max):
    array[array < min] = min
    array[array > max] = max
    return array


def create_mask(size):
    mask = np.zeros(size)
    center = int(size / 2)
    mask[center] = 1.0
    for i in range(center + 1, len(mask)):
        dist = i - center
        if dist % 2 == 0:
            mask[i] = 0.0
            mask[center - dist] = 0.0
        else:
            mask[i] = (-4 / pow(pi, 2)) / pow(dist, 2)
            mask[center - dist] = mask[i]
    return mask


def inverse_radon_transform(sinogram, size, span, arr, steps, filter=True):
    scan_number, detector_number = sinogram.shape
    if filter:
        mask = create_mask(detector_number)
        for i in range(scan_number):
            sinogram[i, :] = np.convolve(sinogram[i, :], mask, 'same')
        sinogram = tresh(np.real(sinogram), 0, 1)

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
        if steps and i % 5 == 0:
            count_copy = count.copy()
            count_copy[count_copy == 0] = 1
            image_copy = image.copy()
            image_copy = image_copy / count_copy
            image_copy = normalize(image_copy)
            image_copy = reshape_to_original(image_copy, size)
            arr.append(Image.fromarray(np.uint8(image_copy * 255), 'L'))
    count[count == 0] = 1
    image = image / count
    image = normalize(image)
    image = reshape_to_original(image, size)
    return image


def download_link(object_to_download, download_filename, download_link_text):
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


def rmsdiff(im1, im2):
    return np.sqrt(np.mean((im1 - im2) ** 2))


def create_dicom(path, image, meta):
    ds = Dataset()
    ds.MediaStorageSOPClassUID = MRImageStorage
    ds.MediaStorageSOPInstanceUID = generate_uid()
    ds.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    fd = FileDataset(path, {}, file_meta=ds, preamble=b'\0' * 128)
    fd.is_little_endian = True
    fd.is_implicit_VR = False

    fd.SOPClassUID = MRImageStorage
    fd.PatientName = 'Test^Firstname'
    fd.PatientID = '123456'
    now = datetime.datetime.now()
    fd.StudyDate = now.strftime('%Y%m%d')

    fd.Modality = 'MR'
    fd.SeriesInstanceUID = generate_uid()
    fd.StudyInstanceUID = generate_uid()
    fd.FrameOfReferenceUID = generate_uid()

    fd.BitsStored = 16
    fd.BitsAllocated = 16
    fd.SamplesPerPixel = 1
    fd.HighBit = 15

    fd.ImagesInAcquisition = '1'
    fd.Rows = image.shape[0]
    fd.Columns = image.shape[1]
    fd.InstanceNumber = 1

    fd.ImagePositionPatient = r'0\0\1'
    fd.ImageOrientationPatient = r'1\0\0\0\-1\0'
    fd.ImageType = r'ORIGINAL\PRIMARY\AXIAL'

    fd.RescaleIntercept = '0'
    fd.RescaleSlope = '1'
    fd.PixelSpacing = r'1\1'
    fd.PhotometricInterpretation = 'MONOCHROME2'
    fd.PixelRepresentation = 1

    for key, value in meta.items():
        setattr(fd, key, value)

    validate_file_meta(fd.file_meta, enforce_standard=True)

    fd.PixelData = (image * 255).astype(np.uint16).tobytes()

    fd.save_as(path)


def write_dicom(pixel_array, filename):
    """
    INPUTS:
    pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
    filename: string name for the output file.
    """

    # This code block was taken from the output of a MATLAB secondary
    # capture.  I do not know what the long dotted UIDs mean, but
    # this code works.
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
    file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble="\0" * 128)
    ds.Modality = 'WSD'
    ds.ContentDate = str(datetime.date.today()).replace('-', '')
    ds.ContentTime = str(time.time())  # milliseconds since the epoch
    ds.StudyInstanceUID = '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
    ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
    ds.SOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    ds.SOPClassUID = 'Secondary Capture Image Storage'
    ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'

    ## These are the necessary imaging components of the FileDataset object.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SmallestImagePixelValue = '\\x00\\x00'
    ds.LargestImagePixelValue = '\\xff\\xff'
    ds.Columns = pixel_array.shape[0]
    ds.Rows = pixel_array.shape[1]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()

    ds.save_as(filename)


def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploadedfile.name))


def array_last_element(arr):
    return len(arr) - 1


if __name__ == '__main__':
    title = 'Tomography simulator'
    st.set_page_config(page_title=title, layout='centered')
    st.title(title)
    detectors = int(st.sidebar.text_input("Number of detectors", 30))  # detector_number
    alpha_step = float(st.sidebar.text_input("\u0394\u03B1 step", 15))  # alpha
    detectors_range = int(st.sidebar.text_input("Detectors range", 90))  # span
    if_filter = st.sidebar.checkbox("Do you want filtering?", True)
    if_radon = st.sidebar.checkbox("Do you want steps in Radon Transformation?", True)
    if_inv_radon = st.sidebar.checkbox("Do you want steps in Inverse Radon Transformation", True)
    file = st.file_uploader('Upload an image')
    regex_pattern = r'[A-Z][a-z]+'

    if file is not None:
        if file.name.lower().endswith('.dcm'):
            ds = pydicom.read_file(file, force=True)
            img = Image.fromarray(ds.pixel_array).convert('L')
        else:
            img = Image.open(file).convert('L')
        image_shape = np.array(img).shape
        print(image_shape)
        st.image(img, "Initial image in grayscale")
        image = make_square(img)
        st.image(img, "image in square")
        img_array = np.array(img)
        img_during_radon = []
        img_during_inv_radon = []

        sinogram = radon_transform(img_array, alpha_step, detectors, detectors_range, img_during_radon, if_radon)

        PIL_image = Image.fromarray(np.uint8(sinogram * 255), 'L')

        if if_radon:
            img_during_radon_last_el = array_last_element(img_during_radon)
            img_during_radon[img_during_radon_last_el] = PIL_image
            to_radon_number = st.slider("Choose radon transform stage", 0,
                                        img_during_radon_last_el, value=img_during_radon_last_el)

            st.image(img_during_radon[to_radon_number],
                     "Sinogram in {num} iteration of Radon transformation".format(num=to_radon_number * 5))
        else:
            st.image(PIL_image, "Radon transformation")

        image_after = inverse_radon_transform(sinogram, image_shape, detectors_range,
                                              img_during_inv_radon, if_inv_radon, if_filter)
        PIL_image = Image.fromarray(np.uint8(image_after * 255), 'L')
        if if_inv_radon:
            img_during_inv_radon_lat_el = array_last_element(img_during_inv_radon)
            img_during_inv_radon[img_during_inv_radon_lat_el] = PIL_image
            inverse_radon_number = st.slider("Choose inverse Radon transform stage", 0,
                                             img_during_inv_radon_lat_el, value=img_during_radon_last_el)
            st.image(img_during_inv_radon[inverse_radon_number],
                     "Sinogram in {num} iteration of inverse Radon transformation".format(num=inverse_radon_number * 5))
        else:
            st.image(PIL_image, "Inverse Radon transformation")

        st.markdown("Root-mean-square deviation equals: **{diff}**".format(diff=rmsdiff(img_array, image_after)))

        first_name = st.text_input("Patient's first name")
        last_name = st.text_input("Patient's last name")
        patient_id = st.text_input("Patient's ID")
        date = st.date_input("Creating date")
        commentary = st.text_input("Medical commentary")
        #
        suffix = '.dcm'
        filename_little_endian = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
        #
        # file_meta = FileMetaDataset()
        # file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        # file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        # file_meta.ImplementationClassUID = "1.2.3.4"
        #
        # ds = FileDataset(filename_little_endian, {},
        #                  file_meta=file_meta, preamble=b"\0" * 128)
        #
        # ds.PatientName = "Test^Firstname"
        # ds.PatientID = "123456"
        #
        # ds.is_little_endian = True
        # ds.is_implicit_VR = True
        #
        # dt = datetime.datetime.now()
        # ds.ContentDate = dt.strftime('%Y%m%d')
        # timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
        # ds.ContentTime = timeStr
        #
        # ds.save_as(filename_little_endian)
        # write_dicom(img_as_float(start_img), filename_little_endian)
        #
        # x = np.arange(16).reshape(16, 1)
        # pixel_array = (x + x.T) * 32
        # pixel_array = np.tile(pixel_array, (16, 16))
        #
        # image2d = pixel_array.astype(np.uint16)
        #
        # meta = pydicom.Dataset()
        # meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
        # meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        # meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        #
        # ds = Dataset()
        # ds.file_meta = meta
        #
        # ds.is_little_endian = True
        # ds.is_implicit_VR = False
        #
        # ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
        # ds.PatientName = "Test^Firstname"
        # ds.PatientID = "123456"
        #
        # ds.Modality = "MR"
        # ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        # ds.StudyInstanceUID = pydicom.uid.generate_uid()
        # ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
        #
        # ds.BitsStored = 16
        # ds.BitsAllocated = 16
        # ds.SamplesPerPixel = 1
        # ds.HighBit = 15
        #
        # ds.ImagesInAcquisition = "1"
        #
        # ds.Rows = image2d.shape[0]
        # ds.Columns = image2d.shape[1]
        # ds.InstanceNumber = 1
        #
        # ds.ImagePositionPatient = r"0\0\1"
        # ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
        # ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
        #
        # ds.RescaleIntercept = "0"
        # ds.RescaleSlope = "1"
        # ds.PixelSpacing = r"1\1"
        # ds.PhotometricInterpretation = "MONOCHROME2"
        # ds.PixelRepresentation = 1
        #
        # pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
        #
        # print("Setting pixel data...")
        # ds.PixelData = image2d.tobytes()
        #
        # ds.save_as(filename_little_endian)

        # save_uploadedfile()

        # if st.button('Download'):

        create_dicom(filename_little_endian, np.array(image_after), dict(
            PatientName=last_name + '^' + first_name,
            PatientID=patient_id,
            ImageComments=commentary,
            StudyDate=date,
        ))
        tmp_download_link = get_binary_file_downloader_html(filename_little_endian, "DICOM")
        st.markdown(tmp_download_link, unsafe_allow_html=True)
