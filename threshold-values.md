# LiteAttention Threshold Error Measurements

Measured on the **first LiteAttention instance, first forward call** during I2V-A14B generation.
Thresholds tested: 0, -1, -3, -10, -15, -20, -30.

- **skip**: fraction of tiles skipped
- **L1**: relative L1 error (normalized by reference output)
- **RMSE**: root mean squared error
- **Cossim**: cosine similarity (1.0 = identical)

## 480x832, 21 frames (seq_len=9180)

| Threshold | Skip % | L1       | RMSE     | Cossim     |
|-----------|--------|----------|----------|------------|
| 0.0       | 64.4%  | 0.399    | 0.223    | 0.9449     |
| -1.0      | 34.4%  | 0.131    | 0.093    | 0.9897     |
| -3.0      | 20.4%  | 0.035    | 0.031    | 0.9989     |
| -10.0     | 1.7%   | 0.0007   | 0.0008   | 0.99999    |
| -15.0     | 0.3%   | 0.0007   | 0.0008   | 0.99999    |
| -20.0     | 0.0%   | 0.0007   | 0.0008   | 0.99999    |
| -30.0     | 0.0%   | 0.0007   | 0.0008   | 0.99999    |

## 480x832, 41 frames (seq_len=16830)

| Threshold | Skip % | L1       | RMSE     | Cossim     |
|-----------|--------|----------|----------|------------|
| 0.0       | 74.3%  | 0.476    | 0.230    | 0.9375     |
| -1.0      | 43.0%  | 0.166    | 0.100    | 0.9877     |
| -3.0      | 27.1%  | 0.046    | 0.036    | 0.9983     |
| -10.0     | 3.4%   | 0.0008   | 0.0009   | 0.99999    |
| -15.0     | 0.8%   | 0.0008   | 0.0008   | 0.99999    |
| -20.0     | 0.2%   | 0.0008   | 0.0008   | 0.99999    |
| -30.0     | 0.0%   | 0.0008   | 0.0008   | 0.99999    |

## 1280x720, 21 frames (seq_len=21528)

| Threshold | Skip % | L1       | RMSE     | Cossim     |
|-----------|--------|----------|----------|------------|
| 0.0       | 74.9%  | 0.500    | 0.245    | 0.9317     |
| -1.0      | 40.9%  | 0.192    | 0.117    | 0.9838     |
| -3.0      | 24.2%  | 0.062    | 0.049    | 0.9970     |
| -10.0     | 2.1%   | 0.0007   | 0.0008   | 0.99999    |
| -15.0     | 0.4%   | 0.0006   | 0.0007   | 0.99999    |
| -20.0     | 0.0%   | 0.0006   | 0.0007   | 0.99999    |
| -30.0     | 0.0%   | 0.0006   | 0.0007   | 0.99999    |

## Key Observations

- **Noise floor**: Thresholds -10 and below converge to ~0.07% L1 error regardless of resolution/frames. This is likely quantization/numerical noise from the skip-list mechanism itself.
- **Sweet spot**: Threshold -3 gives 20-27% tile skipping with 3.5-6.2% L1 error depending on sequence length.
- **Sequence length effect**: Larger sequences (more frames or higher resolution) increase both skip rates and errors at the same threshold. At threshold=-3: 480p/21f skips 20.4% vs 720p/21f skips 24.2%, with L1 rising from 3.5% to 6.2%.
- **Threshold 0**: Extremely aggressive — skips 64-75% of tiles but with 40-50% L1 error.
