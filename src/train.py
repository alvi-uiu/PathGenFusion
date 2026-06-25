import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, precision_recall_curve, auc
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

from .preprocessing import (
    preprocess_metadata,
    preprocess_images,
    filter_data,
    preprocess_gene_expression,
    extract_image_features,
)
from .model import build_transformer_model
from .xai import MultimodalExplainer, create_publication_visualizations

# Default data paths (override by passing arguments to main())
GEX_FILE = "/content/gene_expression.csv"
METADATA_FILE = "/content/TCGA-BRCA-A2-target_variable.xlsx"
IMAGE_DIR = "/content/BLOCKS_NORM_MACENKO"


def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    return lr * float(tf.math.exp(-0.1))


def plot_training_curves(all_histories):
    if not all_histories:
        print("No history objects to plot.")
        return

    epochs_per_fold = [len(h.history['loss']) for h in all_histories if h and 'loss' in h.history]
    if not epochs_per_fold:
        return

    min_ep = min(epochs_per_fold)
    tr_loss = [h.history['loss'][:min_ep] for h in all_histories if 'loss' in h.history]
    vl_loss = [h.history['val_loss'][:min_ep] for h in all_histories if 'val_loss' in h.history]
    tr_acc  = [h.history['accuracy'][:min_ep] for h in all_histories if 'accuracy' in h.history]
    vl_acc  = [h.history['val_accuracy'][:min_ep] for h in all_histories if 'val_accuracy' in h.history]

    if not (tr_loss and vl_loss and tr_acc and vl_acc):
        print("Insufficient history data for plotting.")
        return

    ep = range(1, min_ep + 1)
    avg_tl, std_tl = np.mean(tr_loss, axis=0), np.std(tr_loss, axis=0)
    avg_vl, std_vl = np.mean(vl_loss, axis=0), np.std(vl_loss, axis=0)
    avg_ta, std_ta = np.mean(tr_acc, axis=0), np.std(tr_acc, axis=0)
    avg_va, std_va = np.mean(vl_acc, axis=0), np.std(vl_acc, axis=0)

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a0a')
    train_c, val_c = '#00D2FF', '#FF6B6B'

    for ax, avg_t, std_t, avg_v, std_v, ylabel, title in [
        (ax1, avg_tl, std_tl, avg_vl, std_vl, 'Loss', 'Training & Validation Loss'),
        (ax2, avg_ta, std_ta, avg_va, std_va, 'Accuracy', 'Training & Validation Accuracy'),
    ]:
        ax.set_facecolor('#111111')
        ax.grid(True, alpha=0.15, linewidth=0.5)
        for i in range(len(ep) - 1):
            alpha = 0.8 + 0.2 * (i / len(ep))
            ax.plot(list(ep)[i:i+2], avg_t[i:i+2], color=train_c, linewidth=3, alpha=alpha)
            ax.plot(list(ep)[i:i+2], avg_v[i:i+2], color=val_c, linewidth=3, alpha=alpha)
        ax.fill_between(ep, avg_t - std_t, avg_t + std_t, color=train_c, alpha=0.2, linewidth=0)
        ax.fill_between(ep, avg_v - std_v, avg_v + std_v, color=val_c, alpha=0.2, linewidth=0)
        ax.set_xlabel('Epochs', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color=train_c, linewidth=3, label='Training'),
                plt.Line2D([0], [0], color=val_c, linewidth=3, label='Validation'),
            ],
            frameon=True, fancybox=True, framealpha=0.3, facecolor='#222222',
        )

    plt.suptitle('Model Training Performance', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    print("Saved training curves to 'training_curves.png'")
    plt.show()
    plt.style.use('default')


def validate_data_consistency(image_features, gex_numpy, labels_numpy, metadata_df, gene_names):
    print("\n--- Validating Data Consistency ---")
    n_img = image_features.shape[0] if hasattr(image_features, 'shape') and image_features.ndim > 0 else 0
    n_gex = gex_numpy.shape[0] if hasattr(gex_numpy, 'shape') and gex_numpy.ndim > 0 else 0
    n_lbl = labels_numpy.shape[0] if hasattr(labels_numpy, 'shape') and labels_numpy.ndim > 0 else 0
    n_met = len(metadata_df)

    valid = True
    if not (n_img == n_gex == n_lbl == n_met):
        print(f"Error: Sample count mismatch — images:{n_img}, gex:{n_gex}, labels:{n_lbl}, meta:{n_met}")
        valid = False
    if n_img == 0:
        print("Error: No samples found.")
        valid = False
    if valid and labels_numpy.size > 0:
        unique, counts = np.unique(labels_numpy, return_counts=True)
        if len(unique) < 2:
            print(f"Error: Need at least 2 classes, found {unique}.")
            valid = False
    if valid and hasattr(gex_numpy, 'shape') and gex_numpy.ndim > 1:
        if gex_numpy.shape[1] != len(gene_names):
            print(f"Error: GEX columns {gex_numpy.shape[1]} != gene names {len(gene_names)}")
            valid = False
    print(f"{'Validation passed' if valid else 'Validation failed'}: {n_img} samples.")
    return valid


def main(
    gex_file=GEX_FILE,
    metadata_file=METADATA_FILE,
    image_dir=IMAGE_DIR,
    num_heads=8,
    num_patches=50,
    epochs=35,
    batch_size=8,
    n_splits=5,
):
    random_seed = 39
    import random
    random.seed(random_seed)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    metadata = preprocess_metadata(metadata_file)
    print(f"Metadata: {len(metadata)} samples")
    if metadata.empty:
        return

    image_data_dict, metadata = preprocess_images(image_dir, metadata, max_patches=num_patches)
    print(f"After image preprocessing: {len(metadata)} samples")
    if metadata.empty or not image_data_dict:
        return

    gex_df, metadata, labels = filter_data(gex_file, metadata)
    print(f"After filtering: {len(metadata)} samples, GEX shape: {gex_df.shape}")
    if metadata.empty or gex_df.empty:
        return

    gene_names = gex_df.columns.tolist()
    gex_scaled = preprocess_gene_expression(gex_df)
    if gex_scaled.empty:
        return

    default_h, default_w, default_c = 256, 256, 3
    if image_data_dict:
        first = next(iter(image_data_dict.values()))
        if first.ndim == 4:
            _, default_h, default_w, default_c = first.shape

    ordered_images = []
    for sid in metadata['Sample ID'].tolist():
        if sid in image_data_dict:
            ordered_images.append(image_data_dict[sid])
        else:
            ordered_images.append(np.zeros((num_patches, default_h, default_w, default_c)))

    print("Extracting image features...")
    image_features = extract_image_features(ordered_images)
    print(f"Image features shape: {image_features.shape}")
    if image_features.ndim == 0 or image_features.shape[0] == 0:
        return

    gex_numpy = gex_scaled.to_numpy()
    labels_numpy = labels.to_numpy()

    if not validate_data_consistency(image_features, gex_numpy, labels_numpy, metadata, gene_names):
        return
    labels_numpy = labels_numpy.astype(int)

    unique_l, counts_l = np.unique(labels_numpy, return_counts=True)
    min_class = int(np.min(counts_l)) if len(unique_l) > 1 else 2
    n_splits = min(n_splits, min_class) if min_class > 1 else 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_f1, all_mcc, all_pr_auc, all_histories = [], [], [], []
    best_score = -1.0
    xai_data = {}
    weights_dir = "temp_model_weights"
    os.makedirs(weights_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_features, labels_numpy), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        x1_tr, x1_vl = image_features[train_idx], image_features[val_idx]
        x2_tr, x2_vl = gex_numpy[train_idx], gex_numpy[val_idx]
        y_tr, y_vl = labels_numpy[train_idx], labels_numpy[val_idx]

        tf.keras.backend.clear_session(); gc.collect()

        model = build_transformer_model(
            gene_dims=x2_tr.shape[1],
            vision_feature_dim=x1_tr.shape[2],
            num_patches=x1_tr.shape[1],
            num_heads=num_heads,
        )
        if fold == 1:
            model.summary(line_length=150)

        history = model.fit(
            [x1_tr, x2_tr], y_tr,
            validation_data=([x1_vl, x2_vl], y_vl),
            epochs=epochs, batch_size=batch_size,
            callbacks=[
                LearningRateScheduler(lr_scheduler),
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ],
            verbose=1,
        )
        all_histories.append(history)

        y_pred_prob = model.predict([x1_vl, x2_vl], verbose=0)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_vl, y_pred)
        f1 = f1_score(y_vl, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_vl, y_pred)
        pr_auc = 0.0
        if len(np.unique(y_vl)) > 1:
            prec, rec, _ = precision_recall_curve(y_vl, y_pred_prob)
            pr_auc = auc(rec, prec)

        print(f"Fold {fold}: F1={f1:.4f}, MCC={mcc:.4f}, PR-AUC={pr_auc:.4f}\nCM:\n{cm}")
        all_f1.append(f1); all_mcc.append(mcc); all_pr_auc.append(pr_auc)

        if f1 > best_score:
            best_score = f1
            w_path = os.path.join(weights_dir, f"best_fold_{fold}.weights.h5")
            model.save_weights(w_path)
            xai_data = {
                'construction_args': {
                    'gene_dims': x2_tr.shape[1],
                    'vision_feature_dim': x1_tr.shape[2],
                    'num_patches': x1_tr.shape[1],
                    'num_heads': num_heads,
                },
                'weights_path': w_path,
                'vision_to_explain': x1_vl.copy(),
                'gene_to_explain_np': x2_vl.copy(),
                'labels_to_explain': y_vl.copy(),
            }

        del model; gc.collect()

    if not all_f1:
        print("No folds trained.")
        return

    print(f"\nMean F1:     {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Mean MCC:    {np.mean(all_mcc):.4f} ± {np.std(all_mcc):.4f}")
    print(f"Mean PR-AUC: {np.mean(all_pr_auc):.4f} ± {np.std(all_pr_auc):.4f}")

    plot_training_curves(all_histories)

    if xai_data and 'weights_path' in xai_data:
        print(f"\n--- XAI for best model (F1={best_score:.4f}) ---")
        tf.keras.backend.clear_session(); gc.collect()
        best_model = build_transformer_model(**xai_data['construction_args'])
        try:
            best_model.load_weights(xai_data['weights_path'])
        except Exception as e:
            print(f"Error loading best model weights: {e}")
            return

        explainer = MultimodalExplainer(best_model, gene_names)
        create_publication_visualizations(
            explainer,
            xai_data['vision_to_explain'],
            xai_data['gene_to_explain_np'],
            xai_data['labels_to_explain'],
            gene_feature_names_list=gene_names,
            sample_idx=0, top_genes=15,
            max_genes_for_permutation_viz=200,
        )

    import shutil
    if os.path.exists(weights_dir):
        shutil.rmtree(weights_dir)
        print(f"Cleaned up {weights_dir}")


if __name__ == "__main__":
    main()
