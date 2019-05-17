import datetime
import logging
import os
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from typing import Tuple, List, Iterator

import apache_beam as beam
import h5py
import numpy as np
import pydicom
from apache_beam import Pipeline
from google.cloud import storage

# import matplotlib
# matplotlib.use('Agg')  # Need this to write images from the GSA servers.  Order matters:
# import matplotlib.pyplot as plt  # First import matplotlib, then use Agg, then import plt
# from PIL import Image, ImageDraw  # Polygon to mask
from google.cloud.storage import Blob
from scipy.ndimage.morphology import binary_closing  # Morphological operator

from ml4cvd.defines import IMAGE_EXT, TENSOR_EXT, DICOM_EXT, CONCAT_CHAR, HD5_GROUP_CHAR, GCS_BUCKET, JOIN_CHAR
from ml4cvd.defines import MRI_DATE, MRI_FRAMES, MRI_SEGMENTED, MRI_TO_SEGMENT, MRI_ZOOM_INPUT, MRI_ZOOM_MASK

MRI_PIXEL_WIDTH = 'mri_pixel_width'
MRI_PIXEL_HEIGHT = 'mri_pixel_height'
MRI_SERIES_TO_WRITE = ['cine_segmented_lax_2ch', 'cine_segmented_lax_3ch', 'cine_segmented_lax_4ch', 'cine_segmented_sax_b1', 'cine_segmented_sax_b2',
                       'cine_segmented_sax_b3', 'cine_segmented_sax_b4', 'cine_segmented_sax_b5', 'cine_segmented_sax_b6', 'cine_segmented_sax_b7',
                       'cine_segmented_sax_b8', 'cine_segmented_sax_b9', 'cine_segmented_sax_b10', 'cine_segmented_sax_b11',
                       'cine_segmented_sax_inlinevf']
ALLOWED_MRI_FIELD_IDS = ['20208', '20209']

TEMP_ZIPPED = 'temp_zipped'
TEMP_DICOMS = 'temp_dicoms'
OUTPUT_TENSORS = 'output_tensors'


def tensorize_mri(pipeline: Pipeline, output_path: str):
    # blobs = bucket.list_blobs(prefix="projects/pbatra/mri_test/")
    mri_zip_blobs = bucket.list_blobs(prefix="data/mris/cardiac/2")
    zips_with_allowed_field_ids = _filter_by_field_id(mri_zip_blobs, ALLOWED_MRI_FIELD_IDS)

    # blobs = ["projects/pbatra/mri_test/2345032_20208_2_0.zip", "projects/pbatra/mri_test/2345032_20209_2_0.zip"]
    # blobs = ["projects/pbatra/mri_test/2345032_20208_2_0.zip", "projects/pbatra/mri_test/2345370_20208_2_0.zip"]
    # blob_list = [blob.name for blob in blobs]
    all_files = (
        pipeline
        # create a list of files to read
        # | 'create_file_path_tuple' >> beam.Create([blob for blob in blobs])
        | 'CreateMriZipTuple' >> beam.Create(zips_with_allowed_field_ids)

        | 'ProcessMriZip' >> beam.Map(_write_tensors_from_zipped_dicoms, output_path)
    )

    result = pipeline.run()
    result.wait_until_finish()


try:
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(GCS_BUCKET)
except OSError:
    bucket = 'whoops'
    logging.warning("GCS storage client could not be instantiated!")


def _write_tensors_from_zipped_dicoms(zip_info: Tuple[str, str, str],
                                      output_path: str,
                                      x=256,
                                      y=256,
                                      z=48,
                                      include_heart_zoom=True,
                                      zoom_x=50,
                                      zoom_y=35,
                                      zoom_width=96,
                                      zoom_height=96,
                                      write_pngs=False
                                      ) -> None:
    try:
        zip_blob_path, sample_id, field_id = zip_info

        with tempfile.TemporaryDirectory() as root_folder:
            zip_folder = os.path.join(root_folder, TEMP_ZIPPED)
            dicom_folder = os.path.join(root_folder, TEMP_DICOMS)
            tensors_folder = os.path.join(root_folder, OUTPUT_TENSORS)

            zip_file_name = f"{sample_id}_{field_id}.zip"
            zip_file_path = os.path.join(zip_folder, zip_file_name)
            tensor_file_name = f"{sample_id}_{field_id}{TENSOR_EXT}"
            tensor_path = os.path.join(tensors_folder, tensor_file_name)
            output_blob = bucket.blob(f"{output_path}/{tensor_file_name}")

            logging.info(f"Writing tensor {tensor_file_name} to {output_blob.public_url} ...")

            for folder in [zip_folder, tensors_folder, dicom_folder]:
                if not os.path.exists(folder):
                    os.makedirs(folder)

            bucket.blob(zip_blob_path).download_to_filename(zip_file_path)

            with h5py.File(tensor_path, 'w') as hd5:
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(dicom_folder)
                    _write_tensors_from_dicoms(x, y, z, include_heart_zoom, zoom_x, zoom_y, zoom_width, zoom_height,
                                               write_pngs, tensors_folder, dicom_folder, hd5, Counter())

            output_blob.upload_from_filename(tensor_path)

            shutil.rmtree(zip_folder)
            shutil.rmtree(dicom_folder)
    except:
        logging.exception(f"Problem with mri tensorization for sample id '{sample_id}'")


def _write_tensors_from_dicoms(x,
                               y,
                               z,
                               include_heart_zoom,
                               zoom_x,
                               zoom_y,
                               zoom_width,
                               zoom_height,
                               write_pngs,
                               tensors_folder,
                               dicom_folder,
                               hd5,
                               stats) -> None:
    """Convert a folder of DICOMs from a sample into tensors for each series

    Segmented dicoms require special processing and are written to tensor per-slice

    Arguments
        x: Width of the tensors (actual MRI width will be padded with 0s or cropped to this number)
        y: Height of the tensors (actual MRI width will be padded with 0s or cropped to this number)
        dicom_folder: Folder with all dicoms associated with one sample.
        hd5: Tensor file in which to create datasets for each series and each segmented slice
        stats: Counter to keep track of summary statistics

    """
    views = defaultdict(list)
    for dicom in os.listdir(dicom_folder):
        if os.path.splitext(dicom)[-1] != DICOM_EXT:
            continue
        d = pydicom.read_file(os.path.join(dicom_folder, dicom))
        if d.SeriesDescription.lower() in MRI_SERIES_TO_WRITE:
            views[d.SeriesDescription.lower()].append(d)
            stats[d.SeriesDescription.lower()] += 1

    diastoles = {}
    diastoles_pix = {}
    systoles = {}
    systoles_pix = {}

    for v in views:
        mri_shape = (views[v][0].Rows, views[v][0].Columns, len(views[v]))
        stats['mri shape:' + str(mri_shape)] += 1

        if v != MRI_TO_SEGMENT:
            mri_data = np.zeros((x, y, max(z, len(views[v]))), dtype=np.float32)
        else:
            full_slice = np.zeros((x, y), dtype=np.float32)
            full_mask = np.zeros((x, y), dtype=np.float32)

        for slicer in views[v]:
            if MRI_PIXEL_WIDTH not in hd5:
                hd5.create_dataset(MRI_PIXEL_WIDTH, data=float(slicer.PixelSpacing[0]))
            if MRI_PIXEL_HEIGHT not in hd5:
                hd5.create_dataset(MRI_PIXEL_HEIGHT, data=float(slicer.PixelSpacing[1]))
            if slicer.pixel_array.shape[0] == mri_shape[0] and slicer.pixel_array.shape[1] == mri_shape[1]:
                sx = min(slicer.Rows, x)
                sy = min(slicer.Columns, y)
                slice_data = slicer.pixel_array.astype(np.float32)[:sx, :sy]
                if v != MRI_TO_SEGMENT:
                    mri_data[:sx, :sy, slicer.InstanceNumber - 1] = slice_data
                elif v == MRI_TO_SEGMENT and _has_overlay(slicer):
                    overlay, mask = _get_overlay_from_dicom(slicer)
                    ventricle_pixels = np.count_nonzero(mask == 1)
                    if ventricle_pixels == 0:
                        continue
                    cur_angle = (slicer.InstanceNumber - 1) // MRI_FRAMES  # dicom InstanceNumber is 1-based
                    if not cur_angle in diastoles:
                        diastoles[cur_angle] = slicer
                        diastoles_pix[cur_angle] = ventricle_pixels
                        systoles[cur_angle] = slicer
                        systoles_pix[cur_angle] = ventricle_pixels
                    else:
                        if ventricle_pixels > diastoles_pix[cur_angle]:
                            diastoles[cur_angle] = slicer
                            diastoles_pix[cur_angle] = ventricle_pixels
                        if ventricle_pixels < systoles_pix[cur_angle]:
                            systoles[cur_angle] = slicer
                            systoles_pix[cur_angle] = ventricle_pixels

                    full_slice[:sx, :sy] = slice_data
                    full_mask[:sx, :sy] = mask
                    hd5.create_dataset(MRI_TO_SEGMENT + HD5_GROUP_CHAR + str(slicer.InstanceNumber), data=full_slice, compression='gzip')
                    hd5.create_dataset(MRI_SEGMENTED + HD5_GROUP_CHAR + str(slicer.InstanceNumber), data=full_mask, compression='gzip')
                    if MRI_DATE not in hd5:
                        hd5.create_dataset(MRI_DATE, (1,), data=_date_from_dicom(slicer), dtype=h5py.special_dtype(vlen=str))
                    if include_heart_zoom:
                        zoom_slice = full_slice[zoom_x: zoom_x + zoom_width, zoom_y: zoom_y + zoom_height]
                        zoom_mask = full_mask[zoom_x: zoom_x + zoom_width, zoom_y: zoom_y + zoom_height]
                        hd5.create_dataset(MRI_ZOOM_INPUT + HD5_GROUP_CHAR + str(slicer.InstanceNumber), data=zoom_slice, compression='gzip')
                        hd5.create_dataset(MRI_ZOOM_MASK + HD5_GROUP_CHAR + str(slicer.InstanceNumber), data=zoom_mask, compression='gzip')
                # Commenting out the plotting code because it requires matplotlib, which has been difficult to install on Mac
                #     if write_pngs:
                #         overlay = np.ma.masked_where(overlay != 0, slicer.pixel_array)
                #         # Note that plt.imsave renders the first dimension (our x) as vertical and our y as horizontal
                #         plt.imsave(tensors_folder + v + '_{0:3d}'.format(slicer.InstanceNumber) + '_mask' + IMAGE_EXT, mask)
                #         plt.imsave(tensors_folder + v + '_{0:3d}'.format(slicer.InstanceNumber) + '_overlay' + IMAGE_EXT, overlay)
                #         if include_heart_zoom:
                #             plt.imsave(tensors_folder + v + '_{}'.format(slicer.InstanceNumber) + '_zslice' + IMAGE_EXT, zoom_slice)
                #             plt.imsave(tensors_folder + v + '_{}'.format(slicer.InstanceNumber) + '_zmask' + IMAGE_EXT, zoom_mask)
                # if write_pngs:
                #     plt.imsave(tensors_folder + v + '_' + str(slicer.InstanceNumber) + IMAGE_EXT, slicer.pixel_array)

        if v == MRI_TO_SEGMENT:
            # if len(diastoles) == 0:
            #     raise ValueError('Error not writing tensors if diastole could not be found.')
            for angle in diastoles:
                sx = min(diastoles[angle].Rows, x)
                sy = min(diastoles[angle].Columns, y)
                full_slice[:sx, :sy] = diastoles[angle].pixel_array.astype(np.float32)[:sx, :sy]
                overlay, full_mask[:sx, :sy] = _get_overlay_from_dicom(diastoles[angle])
                hd5.create_dataset('diastole_frame_b' + str(angle), data=full_slice, compression='gzip')
                hd5.create_dataset('diastole_mask_b' + str(angle), data=full_mask, compression='gzip')
                # if write_pngs:
                #     plt.imsave(tensors_folder + 'diastole_frame_b' + str(angle) + IMAGE_EXT, full_slice)
                #     plt.imsave(tensors_folder + 'diastole_mask_b' + str(angle) + IMAGE_EXT, full_mask)

                sx = min(systoles[angle].Rows, x)
                sy = min(systoles[angle].Columns, y)
                full_slice[:sx, :sy] = systoles[angle].pixel_array.astype(np.float32)[:sx, :sy]
                overlay, full_mask[:sx, :sy] = _get_overlay_from_dicom(systoles[angle])
                hd5.create_dataset('systole_frame_b' + str(angle), data=full_slice, compression='gzip')
                hd5.create_dataset('systole_mask_b' + str(angle), data=full_mask, compression='gzip')
                # if write_pngs:
                #     plt.imsave(tensors_folder + 'systole_frame_b' + str(angle) + IMAGE_EXT, full_slice)
                #     plt.imsave(tensors_folder + 'systole_mask_b' + str(angle) + IMAGE_EXT, full_mask)
        else:
            hd5.create_dataset(v, data=mri_data, compression='gzip')


def _has_overlay(d) -> bool:
    try:
        _ = d[0x6000, 0x3000].value
        return True
    except KeyError:
        return False


def _get_overlay_from_dicom(d) -> Tuple[np.ndarray, np.ndarray]:
    """Get an overlay from a DICOM file

    Morphological operators are used to transform the pixel outline of the myocardium
    to the labeled pixel masks for myocardium and left ventricle

    Arguments
        d: the dicom file
        stats: Counter to keep track of summary statistics

    Returns
        Tuple of two numpy arrays.
        The first is the raw overlay array with myocardium outline,
        The second is a pixel mask with 0 for background 1 for myocardium and 2 for ventricle
    """
    i_overlay = 0
    dicom_tag = 0x6000 + 2 * i_overlay
    overlay_raw = d[dicom_tag, 0x3000].value
    rows = d[dicom_tag, 0x0010].value  # rows = 512
    cols = d[dicom_tag, 0x0011].value  # cols = 512
    overlay_frames = d[dicom_tag, 0x0015].value
    bits_allocated = d[dicom_tag, 0x0100].value

    np_dtype = np.dtype('uint8')
    length_of_pixel_array = len(overlay_raw)
    expected_length = rows * cols
    if bits_allocated == 1:
        expected_bit_length = expected_length
        bit = 0
        arr = np.ndarray(shape=(length_of_pixel_array * 8), dtype=np_dtype)
        for byte in overlay_raw:
            for bit in range(bit, bit + 8):
                arr[bit] = byte & 0b1
                byte >>= 1
            bit += 1
        arr = arr[:expected_bit_length]
    if overlay_frames == 1:
        arr = arr.reshape(rows, cols)
        idx = np.where(arr == 1)
        min_pos = (np.min(idx[0]), np.min(idx[1]))
        max_pos = (np.max(idx[0]), np.max(idx[1]))
        short_side = min((max_pos[0] - min_pos[0]), (max_pos[1] - min_pos[1]))
        small_radius = max(2, short_side * 0.185)
        big_radius = max(2, short_side * 0.9)
        myocardium_structure = _unit_disk(small_radius)
        m1 = binary_closing(arr, myocardium_structure).astype(np.int)
        ventricle_structure = _unit_disk(big_radius)
        m2 = binary_closing(arr, ventricle_structure).astype(np.int)
        return arr, m1 + m2


def _unit_disk(r) -> np.ndarray:
    y, x = np.ogrid[-r: r + 1, -r: r + 1]
    return (x ** 2 + y ** 2 <= r ** 2).astype(np.int)


def _str2date(d) -> datetime.date:
    parts = d.split('-')
    if len(parts) < 2:
        return datetime.datetime.now().date()
    return datetime.date(int(parts[0]), int(parts[1]), int(parts[2]))


def _date_from_dicom(d) -> str:
    return d.AcquisitionDate[0:4] + CONCAT_CHAR + d.AcquisitionDate[4:6] + CONCAT_CHAR + d.AcquisitionDate[6:]


def _filter_by_field_id(blobs: Iterator[Blob], allowed_mri_field_ids: List[str]) -> List[Tuple[str, str, str]]:
    # Parse field_id and sample_id from a zipped mri path
    # Assume the zipped path is in the format <sample_id>_<field_id>_*.zip
    zips_with_allowed_field_ids = []
    for blob in blobs:
        blob_path = blob.name
        parsed_file_name = os.path.basename(blob_path).split(JOIN_CHAR)
        sample_id, field_id = parsed_file_name[0], parsed_file_name[1]

        if field_id in allowed_mri_field_ids:
            logging.debug(f"Including MRI zip path {blob}, because field_id {field_id} is not one of '{ALLOWED_MRI_FIELD_IDS}'")
            zips_with_allowed_field_ids.append((blob_path, sample_id, field_id))
        else:
            logging.debug(f"Skipping MRI zip path {blob}, because field_id {field_id} is not one of '{ALLOWED_MRI_FIELD_IDS}'")

    return zips_with_allowed_field_ids
