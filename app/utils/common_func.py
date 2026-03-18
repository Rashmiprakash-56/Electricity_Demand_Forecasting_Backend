import numpy as np

def shap_global_importance(shap_values, feature_names):
    shap_array = np.array(shap_values)

    mean_abs = np.mean(np.abs(shap_array), axis=0)

    return sorted(
        [
            {
                "feature": feature_names[i],
                "mean_abs_shap": float(mean_abs[i])
            }
            for i in range(len(feature_names))
        ],
        key=lambda x: x["mean_abs_shap"],
        reverse=True
    )


def shap_waterfall_format(
    shap_values,
    base_values,
    predictions,
    feature_names,
    top_k=10
):
    output = []

    for i in range(len(shap_values)):
        contributions = [
            {
                "feature": feature_names[j],
                "value": float(shap_values[i][j])
            }
            for j in range(len(feature_names))
        ]

        contributions = sorted(
            contributions,
            key=lambda x: abs(x["value"]),
            reverse=True
        )[:top_k]

        output.append({
            "hour": i + 1,
            "base_value": float(base_values[i]),
            "prediction": float(predictions[i]),
            "contributions": contributions
        })

    return output
