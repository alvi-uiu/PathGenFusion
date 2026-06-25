import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


def preprocess_metadata(file_path):
    try:
        metadata = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: Metadata file {file_path} not found.")
        return pd.DataFrame()

    if 'TCGA Subtype' not in metadata.columns or 'Sample ID' not in metadata.columns:
        print("Error: Metadata file must contain 'TCGA Subtype' and 'Sample ID' columns.")
        return pd.DataFrame()

    metadata = metadata[metadata['TCGA Subtype'] != 'BRCA.Normal']
    metadata['TCGA Subtype'] = metadata['TCGA Subtype'].replace({
        'BRCA.LumA': 0, 'BRCA.LumB': 0,
        'BRCA.Basal': 1, 'BRCA.Her2': 1
    })
    metadata = metadata[pd.to_numeric(metadata['TCGA Subtype'], errors='coerce').notnull()]
    metadata['TCGA Subtype'] = metadata['TCGA Subtype'].astype(int)
    return metadata.reset_index(drop=True)


def preprocess_images(image_dir, metadata, max_patches=50, img_size=(256, 256)):
    patch_data = {}
    valid_sample_ids = []
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    if not os.path.isdir(image_dir):
        print(f"Error: Image directory '{image_dir}' not found or is not a directory.")
        return {}, metadata

    for patient_id in metadata['Sample ID'].values:
        try:
            sample_id_matches = [
                item for item in os.listdir(image_dir)
                if patient_id in item and os.path.isdir(os.path.join(image_dir, item))
            ]
            if not sample_id_matches:
                continue

            patient_path = os.path.join(image_dir, sample_id_matches[0])
            img_files = [
                f for f in os.listdir(patient_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
            ]
            if len(img_files) == 0:
                continue

            selected_files = random.sample(img_files, max_patches) if len(img_files) > max_patches else img_files

            patches = []
            for img_file in selected_files:
                img_path = os.path.join(patient_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        patch = img.resize(img_size)
                        patch_tensor = transforms.ToTensor()(patch)
                        patch_tensor = color_jitter(patch_tensor)
                        patch_array = patch_tensor.numpy().transpose(1, 2, 0)
                        patch_array = np.clip(patch_array, 0, 1)
                        patches.append(patch_array)
                except Exception:
                    continue

            if len(patches) > 0:
                while len(patches) < max_patches:
                    patches.append(patches[random.randint(0, len(patches) - 1)])
                if len(patches) == max_patches:
                    patch_data[patient_id] = np.array(patches)
                    valid_sample_ids.append(patient_id)

        except Exception as e:
            print(f"General error processing patient {patient_id}: {e}")
            continue

    updated_metadata = metadata[metadata['Sample ID'].isin(valid_sample_ids)].copy().reset_index(drop=True)
    final_patch_data = {sid: patch_data[sid] for sid in updated_metadata['Sample ID'] if sid in patch_data}
    return final_patch_data, updated_metadata


def filter_data(gex_file_path, metadata):
    if metadata.empty:
        print("Metadata is empty, cannot filter GEX data.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='int')

    sample_ids_metadata = metadata['Sample ID'].tolist()

    try:
        gex_data = pd.read_csv(gex_file_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: Gene expression file '{gex_file_path}' not found.")
        return pd.DataFrame(), metadata, pd.Series(dtype='int')
    except (pd.errors.EmptyDataError, ValueError) as e:
        print(f"Error reading GEX file: {e}")
        return pd.DataFrame(), metadata, pd.Series(dtype='int')

    if gex_data.empty:
        return pd.DataFrame(), metadata, pd.Series(dtype='int')

    gex_data = gex_data.transpose()
    common_ids = list(set(sample_ids_metadata) & set(gex_data.index))

    if not common_ids:
        print("No common sample IDs found between metadata and GEX data.")
        return pd.DataFrame(), metadata, pd.Series(dtype='int')

    metadata_filtered = metadata[metadata['Sample ID'].isin(common_ids)].copy()
    metadata_filtered = metadata_filtered.sort_values(by='Sample ID').reset_index(drop=True)

    ordered_ids = metadata_filtered['Sample ID'].tolist()
    filtered_gex = gex_data.loc[ordered_ids]
    labels = metadata_filtered['TCGA Subtype']

    return filtered_gex.reset_index(drop=True), metadata_filtered, labels.reset_index(drop=True)


def preprocess_gene_expression(gex_data, nan_strategy='mean', normalization='zscore'):
    if gex_data.empty:
        print("Warning: GEX data is empty before preprocessing.")
        return gex_data

    gex_data = gex_data.apply(pd.to_numeric, errors='coerce')
    min_val = gex_data.min().min()

    if pd.isna(min_val):
        return gex_data.fillna(0)

    if min_val <= 0:
        gex_data = gex_data - min_val + 1e-9

    gex_data = np.log1p(gex_data)

    if normalization == 'zscore':
        scaler = StandardScaler()
    elif normalization == 'minmax':
        scaler = MinMaxScaler()
    else:
        return gex_data.copy()

    scaled_np = scaler.fit_transform(gex_data)
    scaled = pd.DataFrame(scaled_np, index=gex_data.index, columns=gex_data.columns)

    if scaled.isnull().values.any():
        if nan_strategy == 'mean':
            scaled = scaled.fillna(scaled.mean())
        elif nan_strategy == 'median':
            scaled = scaled.fillna(scaled.median())
        elif nan_strategy == 'drop_genes':
            scaled = scaled.dropna(axis=1)
        elif nan_strategy == 'drop_samples':
            scaled = scaled.dropna(axis=0)
        scaled = scaled.fillna(0)

    return scaled


def extract_image_features(image_data_list):
    if not image_data_list:
        print("No image data provided to extract_image_features.")
        return np.array([])

    image_data_list = [
        item for item in image_data_list
        if isinstance(item, np.ndarray) and item.ndim == 4 and item.shape[0] > 0
    ]
    if not image_data_list:
        return np.array([])

    input_shape_cnn = image_data_list[0].shape[1:]
    num_patches = image_data_list[0].shape[0]

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape_cnn)
    x = GlobalAveragePooling2D()(base_model.output)
    feature_extractor = Model(inputs=base_model.input, outputs=x)

    patient_data = []
    for patches in image_data_list:
        if patches.shape[0] != num_patches or patches.shape[1:] != input_shape_cnn:
            patient_data.append(np.zeros((num_patches, 2048)))
            continue

        features = np.zeros((num_patches, 2048))
        for i in range(0, num_patches, 16):
            batch = patches[i:i + 16]
            if batch.shape[0] > 0:
                features[i:i + 16] = feature_extractor.predict(batch, verbose=0)
        patient_data.append(features)

    return np.array(patient_data) if patient_data else np.array([])
