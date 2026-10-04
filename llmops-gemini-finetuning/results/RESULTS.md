# LLMOps Evaluation Results

## Experimental Setup

- Dataset: Iris
- Training samples: 120 per representation
- Test samples: 30 per representation
- Random state: 42
- Base model: Gemini 2.5 Flash
- Epochs: 3
- Learning rate multiplier: 1.0

## Results

| Metric | V1 Raw | V2 Description |
|---|---:|---:|
| Accuracy | 56.67% | 36.67% |
| Format Compliance | 0.00% | 0.00% |

### V1 Per-Class Metrics

| Class | Precision | Recall |
|---|---:|---:|
| Setosa | 52.63% | 100.00% |
| Versicolor | 42.86% | 30.00% |
| Virginica | 100.00% | 40.00% |

### V2 Per-Class Metrics

| Class | Precision | Recall |
|---|---:|---:|
| Setosa | 35.71% | 100.00% |
| Versicolor | 0.00% | 0.00% |
| Virginica | 50.00% | 10.00% |

## Comparison

V1 (raw feature representation) performed better than V2
(natural-language description representation).

V1 achieved 56.67% accuracy, while V2 achieved 36.67%.
Therefore, V1 was better by 20 percentage points.

The raw representation directly exposes the four numerical measurements
to the model without adding linguistic transformation. The natural-language
representation introduces additional wording that does not provide useful
information for this structured classification task.

Both models achieved 0% strict format compliance. The models frequently
returned explanatory text instead of exactly one valid species name
(`setosa`, `versicolor`, or `virginica`). This illustrates an LLM-specific
output-format failure mode that traditional classifiers generally avoid.

## Conclusion

In this experiment, the raw feature representation produced better
classification performance than the natural-language representation.
The experiment demonstrates that LLMOps evaluation should consider not
only task accuracy but also output-format compliance and per-class behavior.
