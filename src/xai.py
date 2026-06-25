import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


class MultimodalExplainer:
    def __init__(self, model, gene_feature_names):
        self.model = model
        self.gene_feature_names = gene_feature_names if gene_feature_names is not None else []

    def integrated_gradients(self, inputs, baseline=None, steps=50):
        if not isinstance(inputs, list):
            inputs = [inputs]
        inputs = [tf.cast(inp, tf.float32) for inp in inputs]

        if baseline is None:
            baseline = [tf.zeros_like(inp, dtype=tf.float32) for inp in inputs]
        elif not isinstance(baseline, list):
            baseline = [tf.cast(baseline, tf.float32)] * len(inputs)
        baseline = [tf.cast(b, tf.float32) for b in baseline]

        alphas = tf.linspace(0.0, 1.0, steps + 1)
        accumulated = [tf.zeros_like(inp, dtype=tf.float32) for inp in inputs]

        for alpha in alphas:
            interpolated = [baseline[i] + alpha * (inputs[i] - baseline[i]) for i in range(len(inputs))]
            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                preds = self.model(interpolated, training=False)
            grads = tape.gradient(preds, interpolated)
            if grads is not None and all(g is not None for g in grads):
                for i in range(len(inputs)):
                    accumulated[i] += grads[i]

        integrated = []
        for i in range(len(inputs)):
            avg_grad = accumulated[i] / float(steps + 1)
            integrated.append((inputs[i] - baseline[i]) * avg_grad)
        return integrated

    def extract_attention_weights(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        inputs = [tf.cast(inp, tf.float32) for inp in inputs]

        attn_outputs = []
        attn_names = []
        seen = set()
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.MultiHeadAttention) and layer.name not in seen:
                attn_outputs.append(layer.output)
                attn_names.append(layer.name)
                seen.add(layer.name)

        if not attn_outputs:
            for layer in self.model.layers:
                if 'attention' in layer.name.lower() and layer.name not in seen:
                    attn_outputs.append(layer.output)
                    attn_names.append(layer.name)
                    seen.add(layer.name)

        if not attn_outputs:
            print("No suitable attention layers found.")
            return None

        try:
            attn_model = tf.keras.Model(inputs=self.model.inputs, outputs=attn_outputs)
            print(f"Extracting from attention layers: {attn_names}")
            values = attn_model(inputs, training=False)
        except Exception as e:
            print(f"Error extracting attention weights: {e}")
            return None

        return values if isinstance(values, list) else [values]

    def compute_gene_importance(self, vision_data, gene_data, labels, method='permutation', max_genes_to_permute=None):
        vision_tf = tf.cast(vision_data, tf.float32)
        gene_tf = tf.cast(gene_data, tf.float32)
        labels_tf = tf.cast(labels, tf.int32)

        baseline_acc = self.model.evaluate([vision_tf, gene_tf], labels_tf, verbose=0)[1]
        num_genes = gene_tf.shape[1]
        importances = np.zeros(num_genes)
        gene_np = gene_tf.numpy()

        if max_genes_to_permute is not None and 0 < max_genes_to_permute < num_genes:
            indices = np.random.choice(num_genes, max_genes_to_permute, replace=False)
            print(f"Computing permutation importance for {max_genes_to_permute}/{num_genes} genes...")
        else:
            indices = np.arange(num_genes)
            print(f"Computing permutation importance for all {num_genes} genes...")

        for count, i in enumerate(indices, 1):
            corrupted = gene_np.copy()
            np.random.shuffle(corrupted[:, i])
            corrupted_acc = self.model.evaluate(
                [vision_tf, tf.convert_to_tensor(corrupted, dtype=tf.float32)], labels_tf, verbose=0
            )[1]
            importances[i] = baseline_acc - corrupted_acc
            if count % 50 == 0 or count == len(indices):
                print(f"  {count}/{len(indices)} genes processed")

        return importances

    def gradcam_patches(self, vision_data, gene_data, target_layer_name=None):
        vision_tf = tf.cast(vision_data, tf.float32)
        gene_tf = tf.cast(gene_data, tf.float32)

        if target_layer_name is None:
            for layer in reversed(self.model.layers):
                try:
                    shape = None
                    if hasattr(layer, 'output_shape'):
                        shape = layer.output_shape
                    elif hasattr(layer, 'output') and hasattr(layer.output, 'shape'):
                        shape = layer.output.shape.as_list()
                    if shape and len(shape) >= 3 and shape[1] == vision_tf.shape[1]:
                        name_l = layer.name.lower()
                        if any(k in name_l for k in ['vision', 'v2g', 'transformer', 'attention', 'layernorm']) \
                                and 'gene' not in name_l:
                            target_layer_name = layer.name
                            print(f"Auto-selected GradCAM target layer: {target_layer_name}")
                            break
                except Exception:
                    continue
            if target_layer_name is None:
                print("Could not automatically find a suitable layer for GradCAM. Please specify target_layer_name.")
                return None

        try:
            self.model.get_layer(target_layer_name)
        except ValueError:
            print(f"Layer '{target_layer_name}' not found in model.")
            return None

        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(target_layer_name).output, self.model.output],
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model([vision_tf, gene_tf], training=False)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            print("Gradients for GradCAM are None.")
            return None

        importance = tf.reduce_sum(tf.abs(grads), axis=-1)
        max_v = tf.reduce_max(importance, axis=1, keepdims=True)
        min_v = tf.reduce_min(importance, axis=1, keepdims=True)
        return ((importance - min_v) / (max_v - min_v + 1e-6)).numpy()


def create_publication_visualizations(
    explainer, vision_data, gene_data, labels,
    gene_feature_names_list, sample_idx=0, top_genes=20,
    max_genes_for_permutation_viz=500,
):
    plt.style.use('default')
    fig = plt.figure(figsize=(22, 14))

    if hasattr(vision_data, 'numpy'): vision_data = vision_data.numpy()
    if hasattr(gene_data, 'numpy'): gene_data = gene_data.numpy()
    if hasattr(labels, 'numpy'): labels = labels.numpy()

    sv = tf.convert_to_tensor(vision_data[sample_idx:sample_idx + 1], dtype=tf.float32)
    sg = tf.convert_to_tensor(gene_data[sample_idx:sample_idx + 1], dtype=tf.float32)

    # 1. Integrated Gradients
    print("Computing Integrated Gradients...")
    ig = explainer.integrated_gradients([sv, sg])
    top_gene_indices_ig = []
    if ig and len(ig) > 1 and ig[1] is not None:
        attrs = np.abs(ig[1][0].numpy())
        if attrs.ndim > 1:
            attrs = np.mean(attrs, axis=0)
        if len(attrs) == len(gene_feature_names_list):
            top_gene_indices_ig = np.argsort(attrs)[-top_genes:]
            ax1 = plt.subplot(2, 3, 1)
            plt.barh(range(top_genes), attrs[top_gene_indices_ig], color='steelblue', alpha=0.8)
            plt.yticks(range(top_genes), [gene_feature_names_list[i] for i in top_gene_indices_ig])
            plt.xlabel('Attribution Score (Integrated Gradients)')
            plt.title(f'Top {top_genes} Gene Features (IG) - Sample {sample_idx}', fontweight='bold')
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            ax1.invert_yaxis()
        else:
            ax1 = plt.subplot(2, 3, 1)
            ax1.text(0.5, 0.5, "IG Plot Skipped (Data Mismatch)", ha='center', va='center')
    else:
        ax1 = plt.subplot(2, 3, 1)
        ax1.text(0.5, 0.5, "IG Plot Skipped", ha='center', va='center')

    # 2. Permutation Importance
    print(f"Computing Permutation Importance ({max_genes_for_permutation_viz} genes)...")
    top_gene_indices_perm_plot = []
    subset = min(50, vision_data.shape[0])
    perm_imp = explainer.compute_gene_importance(
        vision_data[:subset], gene_data[:subset], labels[:subset],
        max_genes_to_permute=max_genes_for_permutation_viz,
    )
    if len(perm_imp) == len(gene_feature_names_list):
        nonzero = np.where(perm_imp != 0)[0]
        if len(nonzero) > 0:
            sorted_nz = nonzero[np.argsort(np.abs(perm_imp[nonzero]))]
            n = min(top_genes, len(sorted_nz))
            top_gene_indices_perm_plot = sorted_nz[-n:]
            scores = perm_imp[top_gene_indices_perm_plot]
            names = [gene_feature_names_list[i] for i in top_gene_indices_perm_plot]
            order = np.argsort(scores)
            scores, names = scores[order], [names[i] for i in order]
            ax2 = plt.subplot(2, 3, 2)
            plt.barh(range(n), scores, color=['firebrick' if x < 0 else 'seagreen' for x in scores], alpha=0.8)
            plt.yticks(range(n), names)
            plt.xlabel('Importance Score (Δ Accuracy)')
            plt.title(f'Top {n} Gene Features (Permutation)', fontweight='bold')
            plt.grid(axis='x', linestyle='--', alpha=0.5)
        else:
            ax2 = plt.subplot(2, 3, 2)
            ax2.text(0.5, 0.5, "Permutation Plot Skipped\n(No non-zero importance)", ha='center', va='center')
    else:
        ax2 = plt.subplot(2, 3, 2)
        ax2.text(0.5, 0.5, "Permutation Plot Skipped\n(Data Mismatch)", ha='center', va='center')

    # 3. GradCAM Patch Importance
    print("Computing GradCAM patch importance...")
    gc_out = explainer.gradcam_patches(sv, sg)
    ax3 = plt.subplot(2, 3, 3)
    if gc_out is not None and gc_out.shape[0] > 0:
        scores_gc = gc_out[0]
        n_p = len(scores_gc)
        ncols = int(np.ceil(np.sqrt(n_p)))
        nrows = int(np.ceil(n_p / ncols))
        padded = np.pad(scores_gc, (0, nrows * ncols - n_p), constant_values=np.nan)
        im = plt.imshow(padded.reshape(nrows, ncols), cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar(im, ax=ax3, label="Normalized Importance Score")
        plt.title(f'Patch Importance (GradCAM) - Sample {sample_idx}', fontweight='bold')
        ax3.set_xticks([]); ax3.set_yticks([])
    else:
        ax3.text(0.5, 0.5, "GradCAM Plot Skipped", ha='center', va='center')

    # 4. Attention Output
    print("Extracting attention outputs...")
    attn_vals = explainer.extract_attention_weights([sv, sg])
    ax4 = plt.subplot(2, 3, 4)
    if attn_vals is not None and len(attn_vals) > 0:
        data_plot = attn_vals[0][0].numpy()
        if data_plot.ndim == 2:
            im = plt.imshow(data_plot, cmap='Blues', aspect='auto', interpolation='nearest')
            plt.colorbar(im, ax=ax4, label="Feature Value")
            plt.title(f'Attention Layer Output - Sample {sample_idx}\n(Shape: {data_plot.shape})', fontweight='bold')
            plt.xlabel('Feature Index'); plt.ylabel('Token Index')
        elif data_plot.ndim == 1:
            im = plt.imshow(data_plot[np.newaxis, :], cmap='Blues', aspect='auto', interpolation='nearest')
            plt.colorbar(im, ax=ax4, label="Feature Value")
            plt.title(f'Attention Layer Output (1 Token) - Sample {sample_idx}', fontweight='bold')
            plt.xlabel('Feature Index'); ax4.set_yticks([])
        else:
            ax4.text(0.5, 0.5, f"Shape: {data_plot.shape}", ha='center', va='center')
    else:
        ax4.text(0.5, 0.5, "Attention Plot Skipped", ha='center', va='center')

    # 5. Feature Distribution
    ax5 = plt.subplot(2, 3, 5)
    top_idx = top_gene_indices_perm_plot[-1] if len(top_gene_indices_perm_plot) > 0 else \
              (top_gene_indices_ig[-1] if len(top_gene_indices_ig) > 0 else -1)
    src = "Permutation" if len(top_gene_indices_perm_plot) > 0 else "IG"
    if top_idx != -1:
        c0, c1 = labels == 0, labels == 1
        if np.sum(c0) > 0 and np.sum(c1) > 0:
            gene_name = gene_feature_names_list[top_idx]
            sns.kdeplot(gene_data[c0, top_idx], label='Class 0', color='dodgerblue', fill=True, ax=ax5, warn_singular=False)
            sns.kdeplot(gene_data[c1, top_idx], label='Class 1', color='tomato', fill=True, ax=ax5, warn_singular=False)
            plt.xlabel('Normalized Gene Expression'); plt.ylabel('Density')
            plt.title(f'Distribution: {gene_name}\n(Top by {src})', fontweight='bold')
            plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.5)
        else:
            ax5.text(0.5, 0.5, "Distribution Plot Skipped", ha='center', va='center')
    else:
        ax5.text(0.5, 0.5, "Distribution Plot Skipped\n(No top gene identified)", ha='center', va='center')

    # 6. Summary
    ax6 = plt.subplot(2, 3, 6); ax6.axis('off')
    pred_prob = explainer.model.predict([sv, sg], verbose=0)[0][0]
    pred_class = int(pred_prob > 0.5)
    lines = [
        f"XAI SUMMARY - Sample {sample_idx}",
        f"Predicted: {pred_class} (Prob: {pred_prob:.3f}), Actual: {labels[sample_idx]}",
        "",
    ]
    if len(top_gene_indices_ig) >= 3:
        lines += ["Top 3 Genes (IG):"] + [f"{i+1}. {gene_feature_names_list[top_gene_indices_ig[-(i+1)]]}" for i in range(3)] + [""]
    if len(top_gene_indices_perm_plot) >= 3:
        lines += ["Top 3 Genes (Permutation):"] + [f"{i+1}. {gene_feature_names_list[top_gene_indices_perm_plot[-(i+1)]]}" for i in range(3)] + [""]
    lines += ["Methods: Integrated Gradients, Permutation Importance, GradCAM, Attention Viz."]
    ax6.text(0.05, 0.95, "\n".join(lines), transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.8, edgecolor='gray'))

    fig.suptitle('Multimodal XAI Analysis for BRCA Classification', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('multimodal_xai_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved XAI visualization to 'multimodal_xai_analysis.png'")
    plt.show()

    results = {}
    if 'attrs' in dir() and attrs is not None: results['gene_attributions_ig'] = attrs
    if len(perm_imp) > 0: results['gene_importance_perm'] = perm_imp
    if gc_out is not None: results['patch_importance_gradcam'] = gc_out
    if attn_vals is not None: results['attention_layer_outputs'] = [
        a.numpy() if hasattr(a, 'numpy') else a for a in attn_vals
    ]
    return results
