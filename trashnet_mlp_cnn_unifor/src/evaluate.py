"""Avaliação e comparação dos modelos."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, f1_score)

from .config import CLASS_NAMES, EXPERIMENTS, FINAL_DIR, OUTPUT_DIR, BATCH_SIZE, experiment_dir
from .data_utils import load_splits
from .training_utils import make_dataset


def _plot_confusions(cm: np.ndarray, out_dir: Path, exp_id: str) -> None:
    "Desenha duas versões da matriz de confusão: uma com a quantidade de exemplos e outra com percentuais calculados por classe verdadeira."
    for normalize, suffix, title in [(False, "counts", "Matriz de confusão — contagens"), (True, "percent", "Matriz de confusão — percentual por classe real")]:
        values = cm.astype(float)
        if normalize:
            row_sums = values.sum(axis=1, keepdims=True)
            values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0) * 100
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")
        ax.set_title(f"{exp_id} — {title}")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                label = f"{values[i, j]:.1f}%" if normalize else str(int(values[i, j]))
                ax.text(j, i, label, ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"matriz_confusao_{suffix}.png", dpi=160)
        plt.close(fig)


def _plot_training_history(history_path: Path, out_dir: Path, exp_id: str) -> None:
    "Mostra como loss e acurácia mudaram durante o treino e a validação. As curvas ajudam a observar a evolução do aprendizado."
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    if history.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if {"loss", "val_loss"}.issubset(history.columns):
        axes[0].plot(history.epoch, history.loss, label="Treino")
        axes[0].plot(history.epoch, history.val_loss, label="Validação")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Época")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
    if {"accuracy", "val_accuracy"}.issubset(history.columns):
        axes[1].plot(history.epoch, history.accuracy, label="Treino")
        axes[1].plot(history.epoch, history.val_accuracy, label="Validação")
        axes[1].set_title("Acurácia")
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Acurácia")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
    fig.suptitle(f"{exp_id} — curvas de aprendizagem")
    fig.tight_layout()
    fig.savefig(out_dir / "curvas_aprendizado.png", dpi=160)
    plt.close(fig)


def evaluate_one(exp_id: str, splits: pd.DataFrame, dataset_root: Path) -> dict:
    "Carrega o melhor modelo salvo e calcula as métricas no conjunto de teste. Também salva previsões, métricas por classe, matrizes de confusão e exemplos de erros."
    cfg = EXPERIMENTS[exp_id]
    out_dir = experiment_dir(exp_id)
    model_path = out_dir / "best_model.keras"
    metadata_path = out_dir / "training_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Modelo/metadata de {exp_id} não encontrado. Execute o treinamento primeiro.")
    test_df = splits[splits.split == "test"].reset_index(drop=True)
    ds = make_dataset(test_df, cfg["image_size"], training=False)
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(ds, verbose=0)
    y_true = test_df.class_index.to_numpy(dtype=int)
    y_pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), average="macro", zero_division=0))
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / "metricas_por_classe.csv", index=True, index_label="classe")
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / "matriz_confusao_contagens.csv")
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
    pd.DataFrame(cm_pct, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / "matriz_confusao_percentual.csv", float_format="%.2f")
    _plot_confusions(cm, out_dir, exp_id)

    predictions = test_df[["relative_path", "class_name", "class_index"]].copy()
    predictions = predictions.rename(columns={"class_name": "classe_real", "class_index": "indice_real"})
    predictions["classe_prevista"] = [CLASS_NAMES[i] for i in y_pred]
    predictions["confianca_prevista"] = confidence
    predictions["acertou"] = y_true == y_pred
    for idx, class_name in enumerate(CLASS_NAMES):
        predictions[f"prob_{class_name}"] = probabilities[:, idx]
    predictions.to_csv(out_dir / "previsoes_teste.csv", index=False)

    errors = predictions[~predictions.acertou].copy().sort_values("confianca_prevista", ascending=False).head(6)
    errors.to_csv(out_dir / "analise_erros_top6.csv", index=False)
    if len(errors):
        fig, axes = plt.subplots(len(errors), 1, figsize=(8, max(3, len(errors) * 2.4)))
        if len(errors) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, errors.iterrows()):
            with Image.open(dataset_root / row.relative_path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"Real: {row.classe_real} | Prevista: {row.classe_prevista} | Confiança: {row.confianca_prevista:.1%}")
            ax.axis("off")
        fig.suptitle(f"{exp_id} — erros de maior confiança (até 6)")
        fig.tight_layout()
        fig.savefig(out_dir / "analise_erros_top6.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    history_path = out_dir / "historico_treinamento.csv"
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    _plot_training_history(history_path, out_dir, exp_id)
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    result = {
        "experimento": exp_id,
        "modelo": cfg["model_type"],
        "resolucao": f"{cfg['image_size'][1]}x{cfg['image_size'][0]}",
        "pesos_de_classe": bool(cfg["class_weights"]),
        "accuracy_teste": accuracy,
        "macro_precision_teste": float(np.mean(precision)),
        "macro_recall_teste": float(np.mean(recall)),
        "macro_f1_teste": macro_f1,
        "weighted_f1_teste": float(report["weighted avg"]["f1-score"]),
        "parametros": int(meta["model_parameters"]),
        "tamanho_modelo_mb": float(meta["model_file_size_mb"]),
        "epocas_executadas": int(meta["epochs_completed"]),
        "melhor_epoca_val_macro_f1": meta["best_epoch_by_validation_macro_f1"],
        "melhor_val_macro_f1": meta["best_validation_macro_f1"],
        "tempo_treinamento_segundos": float(meta["total_training_seconds"]),
        "tempo_medio_epoca_segundos": meta["average_epoch_seconds"],
        "ram_total_gb": meta["hardware"].get("ram_total_gb"),
        "ram_processo_pico_amostrado_mb": meta.get("process_ram_peak_mb_sampled"),
        "dispositivo": meta["hardware"].get("execution_device"),
        "observacao": "Métricas calculadas no conjunto de teste; seleção do checkpoint pela macro-F1 de validação.",
    }
    pd.DataFrame({"classe": CLASS_NAMES, "precision": precision, "recall": recall, "f1_score": f1, "support": support}).to_csv(out_dir / "metricas_detalhadas_por_classe.csv", index=False)
    (out_dir / "metricas_teste.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{exp_id}: accuracy={accuracy:.4f} | macro-F1={macro_f1:.4f}")
    return result


def _make_global_outputs(results: list[dict]) -> None:
    "Reúne os resultados dos experimentos que já foram avaliados em arquivos CSV e Excel, gráficos comparativos e um JSON que registra o modelo escolhido pela validação."
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = FINAL_DIR / "tabelas"
    charts_dir = FINAL_DIR / "graficos"
    confusion_dir = FINAL_DIR / "matrizes_confusao"
    errors_dir = FINAL_DIR / "analise_erros"
    for folder in (tables_dir, charts_dir, confusion_dir, errors_dir):
        folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).sort_values("experimento")
    df.to_csv(tables_dir / "resumo_experimentos.csv", index=False)
    df.to_csv(FINAL_DIR / "resumo_experimentos.csv", index=False)
    with pd.ExcelWriter(tables_dir / "resumo_experimentos.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Resumo", index=False)
        for row in results:
            exp_dir = experiment_dir(row["experimento"])
            per_class = pd.read_csv(exp_dir / "metricas_detalhadas_por_classe.csv")
            per_class.to_excel(writer, sheet_name=row["experimento"][:31], index=False)
            per_class.to_csv(tables_dir / f"metricas_por_classe_{row['experimento']}.csv", index=False)
            # Centraliza os artefatos dos seis experimentos para facilitar a correção.
            for filename in ("curvas_aprendizado.png",):
                source = exp_dir / filename
                if source.exists():
                    shutil.copy2(source, charts_dir / f"{row['experimento']}_{filename}")
            for filename in ("matriz_confusao_counts.png", "matriz_confusao_percent.png"):
                source = exp_dir / filename
                if source.exists():
                    shutil.copy2(source, confusion_dir / f"{row['experimento']}_{filename}")
    shutil.copy2(tables_dir / "resumo_experimentos.xlsx", FINAL_DIR / "resumo_experimentos.xlsx")
    best = max(results, key=lambda r: (r["melhor_val_macro_f1"] if r["melhor_val_macro_f1"] is not None else -1))
    # Copia a análise de erros somente do modelo escolhido pela validação.
    best_dir = experiment_dir(best["experimento"])
    for filename in ("analise_erros_top6.csv", "analise_erros_top6.png", "previsoes_teste.csv"):
        source = best_dir / filename
        if source.exists():
            shutil.copy2(source, errors_dir / f"{best['experimento']}_{filename}")
    (FINAL_DIR / "selecao_modelo.json").write_text(json.dumps({
        "criterio": "Maior macro-F1 no conjunto de validação",
        "experimento_selecionado": best["experimento"],
        "melhor_val_macro_f1": best["melhor_val_macro_f1"],
        "macro_f1_teste_do_selecionado": best["macro_f1_teste"],
        "nota": "O conjunto de teste não foi usado para selecionar o modelo; é apresentado apenas para avaliação final.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Gráfico comparando métricas finais.
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df.accuracy_teste, width, label="Accuracy")
    ax.bar(x, df.macro_f1_teste, width, label="Macro-F1")
    ax.bar(x + width, df.weighted_f1_teste, width, label="Weighted-F1")
    ax.set_xticks(x, df.experimento)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Métrica")
    ax.set_title("Comparação dos seis experimentos no conjunto de teste")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(charts_dir / "comparacao_metricas_teste.png", dpi=160)
    fig.savefig(FINAL_DIR / "comparacao_metricas_teste.png", dpi=160)
    plt.close(fig)

    # Tempo de treinamento e tamanho dos modelos.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df.experimento, df.tempo_treinamento_segundos / 60)
    ax.set_ylabel("Minutos")
    ax.set_title("Tempo total de treinamento por experimento")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(charts_dir / "comparacao_tempo_treinamento.png", dpi=160)
    fig.savefig(FINAL_DIR / "comparacao_tempo_treinamento.png", dpi=160)
    plt.close(fig)

    cnn = df[df.experimento.isin(["E3", "E5", "E6"])].copy()
    if len(cnn) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(cnn.resolucao, cnn.accuracy_teste, marker="o", label="Accuracy")
        ax.plot(cnn.resolucao, cnn.macro_f1_teste, marker="o", label="Macro-F1")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Resolução")
        ax.set_ylabel("Métrica")
        ax.set_title("CNN: efeito da resolução nas métricas de teste")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(charts_dir / "comparacao_resolucao_cnn.png", dpi=160)
        fig.savefig(FINAL_DIR / "comparacao_resolucao_cnn.png", dpi=160)
        plt.close(fig)


def main() -> None:
    "Define quais experimentos serão avaliados e, ao final, consolida os resultados disponíveis. Se um modelo ainda não foi treinado, ele é ignorado quando a opção é avaliar todos."
    parser = argparse.ArgumentParser(description="Avalia modelos treinados do TrashNet")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Avalia todos os experimentos com modelos disponíveis")
    group.add_argument("--experiment", choices=list(EXPERIMENTS), help="Avalia um experimento")
    args = parser.parse_args()
    splits, dataset_root = load_splits()
    selected = list(EXPERIMENTS) if args.all else [args.experiment]
    results = []
    for exp_id in selected:
        if (experiment_dir(exp_id) / "best_model.keras").exists():
            results.append(evaluate_one(exp_id, splits, dataset_root))
        elif args.all:
            print(f"{exp_id}: modelo não encontrado; pulando avaliação.")
        else:
            raise FileNotFoundError(f"Modelo de {exp_id} não encontrado.")
    if results:
        # Atualiza a visão global com todos os modelos que já foram avaliados.
        existing = []
        for exp_id in EXPERIMENTS:
            path = experiment_dir(exp_id) / "metricas_teste.json"
            if path.exists():
                existing.append(json.loads(path.read_text(encoding="utf-8")))
        _make_global_outputs(existing)
        print(f"\nResultados consolidados em: {FINAL_DIR}")


if __name__ == "__main__":
    main()
