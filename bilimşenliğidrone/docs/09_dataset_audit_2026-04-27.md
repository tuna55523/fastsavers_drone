# Dataset Audit - 2026-04-27

Dataset root:

- `bilimşenliğidrone/VisDrone.v1i.yolov8`

## 1) Split and file integrity

- Train images: `5180`
- Valid images: `1497`
- Test images: `740`
- Total images (actual): `7417`

- Train labels: `5180`
- Valid labels: `1497`
- Test labels: `740`

Integrity checks:

- Missing label file: `0`
- Orphan label file: `0`
- Empty label file: `0`
- Malformed label line: `0`
- Parse/token error: `0`
- Out-of-range bbox value: `0`
- Non-positive width/height bbox: `0`

## 2) Class consistency

- `nc: 1`
- `names: ['person']`
- Total box count: `146311`
- Class IDs found in labels: `{0: 146311}`

## 3) Split leakage checks

Raw name overlap across splits (after removing Roboflow hash suffix):

- train-valid: `0`
- train-test: `0`
- valid-test: `0`

Exact duplicate image overlap by MD5 hash:

- train-valid: `0`
- train-test: `0`
- valid-test: `0`

## 4) Resolution distribution

Most common resolutions:

- `1400x1050`: `2492`
- `1400x788`: `1585`
- `1360x765`: `1229`
- `2000x1500`: `735`
- `1916x1078`: `592`
- `960x540`: `411`
- `1920x1080`: `340`

Unreadable images: `0`

## 5) Object-size profile (normalized bbox area)

Train:

- mean: `0.000429`
- p50: `0.000201`
- p90: `0.000983`
- p99: `0.003400`
- tiny (`area < 0.001`): `90.27%`

Valid:

- mean: `0.000454`
- p50: `0.000211`
- p90: `0.001060`
- p99: `0.003762`
- tiny (`area < 0.001`): `89.07%`

Test:

- mean: `0.000466`
- p50: `0.000214`
- p90: `0.001148`
- p99: `0.003512`
- tiny (`area < 0.001`): `88.05%`

Interpretation:

- Dataset technically clean and training-ready.
- Strongly biased toward very small, distant people (aerial crowd style).

## 6) Practical conclusion for science fair follow task

Verdict: `Conditional approval`

Why conditional:

1. Detection quality for crowded aerial scenes should be good.
2. But nearest-person follow in school yard needs more medium/large person samples.

Recommended before final model lock:

1. Use this dataset for base training.
2. Add school-yard custom data (at least 300-800 labeled images) with medium/large persons.
3. Fine-tune final model on mixed data, giving higher weight to your custom set.
