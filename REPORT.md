# Báo cáo tiến độ nghiên cứu
## Machine Learning-Based UAV Intrusion Detection: Feature Selection versus Feature Engineering

> Tài liệu này tổng hợp lại **toàn bộ pipeline, các quyết định kỹ thuật, và trạng thái thực tế** của
> dự án tại thời điểm hiện tại, để phục vụ việc review. Mọi số liệu trạng thái (đã chạy / chưa chạy,
> file nào tồn tại) đều được kiểm tra trực tiếp trên ổ đĩa, không suy diễn từ code.

---

## 1. Bối cảnh & Câu hỏi nghiên cứu

Dự án lấy cảm hứng từ bài báo *"Machine Learning-Based UAV Intrusion Detection: Feature
Selection versus Feature Extraction"*, nhưng mở rộng thành một **nghiên cứu so sánh có hệ thống**
giữa nhiều phương pháp Feature Selection (FS) và Feature Engineering/Extraction (FE) trên nhiều
họ classifier khác nhau, thay vì chỉ tái lập bài báo gốc.

**Research Question:**
> Với dataset UAV-NIDD, kỹ thuật FS hay FE nào là phù hợp nhất cho từng classifier
> (Decision Tree, Random Forest, XGBoost, LightGBM, KNN, MLP), khi xét cả hiệu năng phát hiện
> **và** chi phí tính toán (training time, inference time)?

Tài liệu tham khảo đã thu thập trong [paper/](paper/):
- *UAV-NIDD: A Dynamic Dataset for Cybersecurity and Intrusion Detection in UAV Networks*
- *Ensemble Machine Learning for UAV Network Intrusion Detection — Comprehensive Analysis*
- 2307.01570v1 (bài báo gốc truyền cảm hứng cho đề tài)

---

## 2. Tổng quan Pipeline (thiết kế)

```
Phase 1  Chuẩn bị dữ liệu & tiền xử lý ───────────────────────  [00_preprocessing.ipynb]
Phase 2  Tối ưu hyperparameter (chỉ trên raw features) ───────  [01_baseline.ipynb]
Phase 3  Feature Reduction
            ├─ FS: Correlation, Chi-Square, XGB Importance,
            │      SHAP, RFE, Consensus               ────────  [02_feature_selection/*]
            └─ FE: PCA, LDA, KernelPCA, AutoEncoder,
                   Statistical Features                ────────  [03_geature_extraction/*]
Phase 4  Classification (DT/RF/XGB/LGBM/KNN/MLP, hyperparameter cố định từ Phase 2)
Phase 5  Lặp lại 5 seeds (42, 52, 62, 72, 82)
Phase 6  Đánh giá: Accuracy, Precision, Recall, F1, PR-AUC, FPR/FNR, Training/Inference time
Phase 7  Phân tích so sánh: FS tốt nhất / FE tốt nhất / chiến lược tổng thể tốt nhất cho mỗi classifier
```

Toàn bộ logic dùng chung (resampling, tính metric, tạo model, vòng lặp thực nghiệm có thể
resume) được tách vào một module duy nhất: [notebook/common.py](notebook/common.py), để đảm bảo
**6 notebook FS + 5 notebook FE đều dùng đúng một implementation** — tránh sai lệch do code
viết lại nhiều lần.

---

## 3. Trạng thái thực hiện thực tế (đã kiểm tra trên ổ đĩa)

| Phase | Notebook | Trạng thái | Bằng chứng |
|---|---|---|---|
| 1. Tiền xử lý | [00_preprocessing.ipynb](notebook/00_preprocessing.ipynb) | ✅ **Đã chạy xong** | `processed/splits.npz` (146MB), `scaler.pkl`, `label_encoder.pkl`, `meta.json` đã tồn tại với số liệu thật |
| 2. Baseline + tuning | [01_baseline.ipynb](notebook/01_baseline.ipynb) | ⛔ **Code hoàn chỉnh, CHƯA chạy** | `models/best_params.json` chưa tồn tại, `results/` trống |
| 3a. FS rankings | [02_00_fs_rankings.ipynb](notebook/02_feature_selection/02_00_fs_rankings.ipynb) | ⛔ Code hoàn chỉnh, chưa chạy | `models/fs_rankings.pkl` chưa tồn tại |
| 3a. FS × 6 phương pháp | [02_01](notebook/02_feature_selection/02_01_fs_correlation.ipynb) … [02_06](notebook/02_feature_selection/02_06_fs_consensus.ipynb) | ⛔ Code hoàn chỉnh, chưa chạy | không có `results/fs_*_raw.csv` |
| 3a. FS merge | [02_99_fs_merge_results.ipynb](notebook/02_feature_selection/02_99_fs_merge_results.ipynb) | ⛔ Chưa chạy được (phụ thuộc 02_01-06) | — |
| 3b. FE × 5 phương pháp | [03_01](notebook/03_geature_extraction/03_01_fe_pca.ipynb) … [03_05](notebook/03_geature_extraction/03_05_fe_statfeatures.ipynb) | ⛔ Code hoàn chỉnh, chưa chạy | không có `results/fe_*_raw.csv` |
| 3b. FE merge + FS-vs-FE | [03_99_fe_merge_results.ipynb](notebook/03_geature_extraction/03_99_fe_merge_results.ipynb) | ⛔ Chưa chạy được | — |

**Kết luận về trạng thái**: Toàn bộ **kiến trúc pipeline đã được thiết kế và code đầy đủ** (16
notebook + 1 module chung), nhưng tính đến hiện tại **chỉ có Phase 1 (tiền xử lý) đã thực sự
chạy và tạo ra dữ liệu**. Phase 2 trở đi chưa được thực thi — `models/` và `results/` đang trống.
Báo cáo này do đó **chưa có số liệu định lượng (F1, PR-AUC, training time...) để trình bày** —
phần "kết quả" sẽ được bổ sung sau khi notebook 01 → 02 → 03 được chạy tuần tự.

---

## 4. Dataset — đã xử lý xong (Phase 1)

| Thuộc tính | Giá trị thực tế (sau xử lý) | Giá trị mô tả ban đầu trong đề cương |
|---|---|---|
| Số mẫu | **4,854,083** | ~803,000 |
| Số cột thô trong CSV | 135 | ~44 |
| Số feature giữ lại sau xử lý | **72** | ~44 |
| Số lớp (classes) | 10 | (không nêu cụ thể) |

> ⚠️ **Lưu ý cho reviewer**: dataset thực tế đã *lớn hơn nhiều* so với số liệu trong đề cương ban
> đầu (4.85M dòng / 135 cột thô so với ~803K / ~44). Theo ghi chú trong notebook, đây là do nhiều
> phiên capture đã được merge thêm vào `data/merged_cleaned.csv` theo thời gian. Phần "Dataset"
> trong báo cáo/luận văn cuối cùng cần cập nhật con số này.

10 lớp: `Brute-Force, DDoS, De-Authentication, DoS, Fake-Landing, GPS-Jamming, MITM, Normal,
Replay-Attack, Scanning` — **mất cân bằng nặng** (lớp lớn nhất `DDoS` ≈ 2.22M mẫu train, lớp nhỏ
nhất `Scanning` ≈ 6,321 mẫu train, tỉ lệ ~350:1).

### 4.1. Phát hiện quan trọng: Data Leakage qua timestamp

Trong quá trình tiền xử lý, một kiểm tra chẩn đoán phát hiện: **mỗi lớp tấn công được ghi lại
trong một khung thời gian tuyệt đối tách biệt** (mỗi loại attack = một session capture riêng).
Một Decision Tree huấn luyện **chỉ với cột `frame.time_epoch`** đạt macro F1 = **1.0000** — mô
hình học được "gói tin này thuộc session nào", không học hành vi tấn công thật.

**Cách xử lý:**
1. Loại bỏ 4 cột timing ở cấp frame: `frame.time_epoch`, `frame.time_relative`,
   `frame.time_delta`, `frame.time_delta_displayed` khỏi feature matrix (giữ `frame.time_epoch`
   nội bộ chỉ để làm khóa sắp xếp thời gian khi chia tách dữ liệu).
   - `tcp.time_delta` / `tcp.time_relative` được **giữ lại** vì mô tả hành vi timing theo từng
     kết nối (per-connection), không phải đặc trưng của session/file ghi.
2. Đổi chiến lược chia tách dữ liệu từ random stratified sang **time-based split theo từng lớp**
   (xem mục 4.3) để tránh các gói tin gần nhau về thời gian (rất giống nhau) bị rải ngẫu nhiên
   vào cả train và test.

Sau khi sửa: F1 giảm về mức thực tế hơn — **0.8494** (so với 1.0000 giả tạo trước đó).

> Đây là điểm **quan trọng nhất cần nêu rõ trong phần Methodology** của bài báo/luận văn — nó cho
> thấy nhóm đã chủ động kiểm tra leakage, không chỉ chạy mô hình và nhận kết quả "đẹp" một cách mù quáng.

### 4.2. Chính sách giữ lại feature (thay đổi so với cách làm ban đầu)

Một phiên bản tiền xử lý trước đó dùng ngưỡng missing-value (drop cột có >50% giá trị thiếu),
chỉ giữ được **16 feature**. Cách này được xác định là **sai** đối với dataset này: phần lớn giá
trị "thiếu" không phải là dữ liệu lỗi, mà do đặc trưng đó **chỉ áp dụng cho một loại protocol/
attack** (ví dụ `wlan.fixed.reason_code` chỉ có ở frame quản lý 802.11, `udp.srcport` chỉ có ở
gói UDP). Đây chính xác là loại tín hiệu mà nghiên cứu FS/FE muốn khai thác — drop theo ngưỡng
missing sẽ khiến K=16 trong các thí nghiệm FS/FE tương đương "giữ hết", làm mất ý nghĩa so sánh.

**Chính sách mới:** giữ mọi cột numeric thật (xác minh trên *toàn bộ file*, không chỉ 1 chunk),
fill missing = 0 (NaN = "field không áp dụng cho loại gói tin này"), chỉ loại bỏ:
- Cột định danh / free-text (MAC, domain name, IP dạng string, hex payload thô…)
- 4 cột leakage nêu ở mục 4.1
- Cột zero-variance (hằng số tuyệt đối sau khi fill)

Kết quả: **72 feature** (so với 16 theo cách cũ), trong đó có 2 cột hex được phục hồi thành số
(`wlan.fixed.reason_code`, `wlan.fc.ds` — parse từ dạng `'0x0007'`) vì chúng **liên quan trực
tiếp đến tấn công** (reason code là giá trị mà công cụ deauth như aireplay-ng thiết lập).

**Quyết định có chủ ý chưa làm**: các trường định danh thiết bị (MAC address, IP string) **không**
được label/one-hot encode, dù có thể mang tín hiệu tấn công (vd. địa chỉ broadcast trong
DDoS/deauth flood, MAC/IP không khớp trong MITM) — vì encode trực tiếp sẽ tái tạo đúng loại lỗi
leakage như `frame.time_epoch` (mô hình học "MAC của attacker trong session này" thay vì hành vi
tổng quát). Việc khai thác an toàn (vd. cờ is-broadcast, kiểm tra nhất quán IP↔MAC) được để lại
cho pha Feature Engineering (notebook 03) — **hiện chưa được triển khai**, là một hướng mở.

### 4.3. Chiến lược chia tách dữ liệu

- **Time-based split theo từng lớp** (không phải random stratified): trong mỗi lớp, sắp xếp các
  dòng theo `frame.time_epoch`, sau đó cắt 70% sớm nhất → train, 15% tiếp theo → validation, 15%
  muộn nhất → test. Index được xáo trộn (shuffle) trong từng split sau khi cắt mốc thời gian.
- Tỷ lệ: Train 70% / Val 15% / Test 15%, `random_state = 42`.
- Kích thước thực tế: Train = 3,397,857 / Val = 728,113 / Test = 728,113.
- **Hạn chế đã ghi nhận**: phần lớn các lớp chỉ có **một session capture duy nhất**, nên đây là
  một *temporal hold-out* (train trên "quá khứ" của session đó, test trên "tương lai" của cùng
  session), **không phải** true leave-session-out validation. Cần nêu rõ hạn chế này khi viết
  phần Threats to Validity / Limitations.

### 4.4. Các bước xử lý khác

| Quyết định | Giá trị | Lý do |
|---|---|---|
| Đọc CSV | Chunked, 50,000 dòng/chunk | Đọc 1 lần toàn bộ file 4.85M dòng từng đẩy RSS lên ~13.7GB và bị OOM-killer giết trên máy 15GB RAM — đây là nguyên nhân thật của lỗi `KeyboardInterrupt` gặp trước đó, không phải do dừng tay |
| Object→numeric | `ip.version`, `ip.proto`, `ip.ttl` | tshark ghi giá trị số dạng string, đôi khi nối nhiều giá trị bằng dấu phẩy (`'64,64'`) → coerce + fill 0 |
| Scaler | `StandardScaler`, **fit chỉ trên train**, transform val/test | tránh leakage thống kê từ val/test vào train |
| Class imbalance | **Không xử lý ở bước này** | xử lý riêng trong từng notebook thực nghiệm (RandomUnderSampler → BorderlineSMOTE), theo từng seed, chỉ trên tập train |

**Artifact đã lưu**: [processed/splits.npz](processed/splits.npz) (146MB), `scaler.pkl`,
`label_encoder.pkl`, `meta.json`.

---

## 5. Phase 2 — Baseline & Hyperparameter Tuning (đã viết code, chưa chạy)

Notebook [01_baseline.ipynb](notebook/01_baseline.ipynb) đã code đầy đủ quy trình:

1. **Resampling trên train** (theo từng seed): `RandomUnderSampler` giới hạn lớp lớn ở
   `CAP_MAJORITY = 20,000` mẫu → `BorderlineSMOTE` nâng lớp nhỏ lên `MIN_MINORITY = 10,000` mẫu.
2. **Grid search hyperparameter** cho 6 classifier, chọn theo **F1 macro trên validation set**
   (test set không được dùng để chọn mô hình — đúng yêu cầu của đề cương mục 11/12):
   - DT: `max_depth ∈ {10,20,None} × min_samples_leaf ∈ {1,5}`
   - RF: `n_estimators ∈ {100,200} × max_depth ∈ {10,20,None}`
   - XGB / LGBM: `n_estimators ∈ {100,200} × max_depth ∈ {6,10} × learning_rate ∈ {0.1,0.3}`
   - KNN: `n_neighbors ∈ {3,5,11}`
   - MLP: `hidden_layer_sizes ∈ {(128,), (256,), (128,64)}`, early stopping
3. Lưu hyperparameter tốt nhất → `models/best_params.json`, **tái sử dụng cố định** cho mọi thí
   nghiệm FS/FE ở Phase 3 (đảm bảo khác biệt hiệu năng đến từ FS/FE, không phải từ tuning lại).
4. Đánh giá baseline (raw features, không giảm chiều) trên 5 seeds, tính mean ± std cho:
   Accuracy, Precision, Recall, F1, PR-AUC, FPR, FNR, Training time, Inference time.
5. Per-class metrics + confusion matrix cho classifier tốt nhất + Wilcoxon signed-rank test giữa
   các classifier.

**Việc cần làm tiếp**: chạy notebook này (dự kiến tốn nhiều thời gian do grid search trên
3.4M dòng × nhiều cấu hình × 6 classifier) để sinh `models/best_params.json` — **đây là điều
kiện tiên quyết** để chạy bất kỳ notebook FS/FE nào (`common.py: load_data()` sẽ raise lỗi nếu
file này chưa tồn tại).

---

## 6. Phase 3a — Feature Selection (6 phương pháp, đã viết code, chưa chạy)

| Notebook | Phương pháp | Loại (theo đề cương §8) | Ghi chú kỹ thuật |
|---|---|---|---|
| [02_00](notebook/02_feature_selection/02_00_fs_rankings.ipynb) | Tính ranking 1 lần, dùng chung | — | Chạy trên **stratified subsample 100,000 mẫu** của train (không phải full 3.4M) — vì RFE/SHAP trên full set là nguyên nhân khiến notebook gốc bị treo/crash trước đây. Subsample chỉ ảnh hưởng đến *chọn feature nào*, không ảnh hưởng dữ liệu huấn luyện thật của classifier |
| [02_01](notebook/02_feature_selection/02_01_fs_correlation.ipynb) | Correlation | Filter | `|corrcoef(feature, label)|` |
| [02_02](notebook/02_feature_selection/02_02_fs_chisquare.ipynb) | Chi-Square | Filter | MinMax-scale trước khi tính chi2 (yêu cầu giá trị không âm) |
| [02_03](notebook/02_feature_selection/02_03_fs_xgb_importance.ipynb) | XGBoost Importance | Embedded | `feature_importances_` từ XGBClassifier |
| [02_04](notebook/02_feature_selection/02_04_fs_shap.ipynb) | SHAP | Explainable Embedded | `TreeExplainer` trên XGBoost, mean(|SHAP value|) trên 500 mẫu background |
| [02_05](notebook/02_feature_selection/02_05_fs_rfe.ipynb) | RFE | Wrapper | Decision Tree (`max_depth=10`) là estimator nền, loại 1 feature/bước |
| [02_06](notebook/02_feature_selection/02_06_fs_consensus.ipynb) | Consensus | Hybrid | Trung bình rank-position của 5 phương pháp trên |

Mỗi notebook (`02_01`–`02_06`) chạy lưới đầy đủ **K ∈ {4,8,16} × seed ∈ {5 seeds} × 6
classifier**, ghi kết quả **incremental** vào `results/fs_<method>_raw.csv` — nếu kernel
crash giữa chừng, chạy lại sẽ tự bỏ qua các combination đã hoàn thành (`load_done_set` trong
`common.py`).

Notebook [02_99](notebook/02_feature_selection/02_99_fs_merge_results.ipynb) gộp 6 file kết
quả, sinh: bảng mean±std, FS tốt nhất theo từng classifier, heatmap F1 (method × K, theo
classifier), và kiểm định Wilcoxon so với baseline.

---

## 7. Phase 3b — Feature Engineering / Extraction (5 phương pháp, đã viết code, chưa chạy)

| Notebook | Phương pháp | Loại (theo đề cương §9) | Ghi chú kỹ thuật |
|---|---|---|---|
| [03_01](notebook/03_geature_extraction/03_01_fe_pca.ipynb) | PCA | Unsupervised Linear | fit chỉ trên train, in ra explained-variance-ratio mỗi K |
| [03_02](notebook/03_geature_extraction/03_02_fe_lda.ipynb) | LDA | **Supervised** Linear | `fit_transform(X_train, y_train)` — dùng nhãn đúng cách. **Giới hạn cứng**: tối đa `n_classes - 1 = 9` thành phần. K=16 sẽ bị clip về 9 và zero-pad cho khớp shape (cột `actual_K` trong kết quả ghi rõ giá trị thật) |
| [03_03](notebook/03_geature_extraction/03_03_fe_kernelpca.ipynb) | Kernel PCA (RBF) | Nonlinear Manifold | Đây là phương pháp tốn bộ nhớ nhất — kernel matrix có shape `(n_samples, n_fit_samples)` sẽ nổ RAM với 3.4M dòng. **Fix**: fit trên subsample ngẫu nhiên 2,000 mẫu, transform theo batch 50,000 dòng |
| [03_04](notebook/03_geature_extraction/03_04_fe_autoencoder.ipynb) | AutoEncoder (shallow) | Deep Nonlinear | `MLPRegressor` D→K(ReLU)→D, học reconstruct input (unsupervised). Fit trên stratified subsample 200,000 mẫu (train MLPRegressor trên full 3.4M quá chậm với solver của sklearn); sau khi fit, encode (`ReLU(X·W+b)`) áp dụng cho **toàn bộ** train/val/test |
| [03_05](notebook/03_geature_extraction/03_05_fe_statfeatures.ipynb) | Statistical Features | Domain-Knowledge | Sinh 10 thống kê (mean/std/min/max/range/median/skew/kurtosis/p25/p75) trên 20 raw feature gốc cho mỗi dòng, chọn top-K theo variance trên train. **Chỉ có 10 thống kê khả dụng** → K=16 bị clip về 10 và zero-pad, giống cơ chế của LDA |

Notebook [03_99](notebook/03_geature_extraction/03_99_fe_merge_results.ipynb) gộp 5 file kết
quả, sinh thêm **biểu đồ so sánh FS-tốt-nhất vs FE-tốt-nhất theo từng classifier và K**
(`fs_vs_fe_comparison.png`) — phụ thuộc cả `02_99` đã chạy trước.

> ⚠️ **Điểm cần lưu ý khi viết Limitations**: với K=16, LDA và StatFeatures thực chất chỉ dùng
> 9 và 10 chiều thật (phần còn lại là zero-pad) — khi so sánh "K=16" giữa các phương pháp, kết
> quả của 2 phương pháp này tại K=16 **không hoàn toàn tương đương về số chiều thông tin thật**
> với PCA/KernelPCA/AutoEncoder/FS (vốn dùng đủ 16 chiều). Cột `actual_K` trong file kết quả
> được thiết kế sẵn để xử lý đúng vấn đề này khi phân tích.

---

## 8. Module dùng chung — [notebook/common.py](notebook/common.py)

Thiết kế để đảm bảo **tính công bằng và khả năng tái lập** giữa tất cả 11 phương pháp FS/FE:

- `load_data()` — nạp `splits.npz`, `label_encoder.pkl`, `meta.json`, `best_params.json` (raise lỗi rõ ràng nếu thiếu, kèm hướng dẫn notebook nào cần chạy trước).
- `resample_train()` — RandomUnderSampler → BorderlineSMOTE, **chỉ áp dụng cho train**, theo seed.
- `compute_metrics()` — accuracy/precision/recall/F1 (macro) + PR-AUC (macro one-vs-rest) + FPR/FNR (macro trên confusion matrix).
- `make_model()` — factory tạo 6 classifier với hyperparameter cố định từ `best_params.json`.
- `run_experiment_grid()` — vòng lặp K × seed × classifier **resumable**: ghi từng dòng kết quả ngay sau khi hoàn thành, bỏ qua các combination đã có sẵn trong CSV khi chạy lại (chống mất tiến độ khi Colab disconnect hoặc kernel crash).
- Cấu hình thực nghiệm tập trung một nơi: `SEEDS = [42,52,62,72,82]`, `K_VALUES = [4,8,16]`, `CAP_MAJORITY=20_000`, `MIN_MINORITY=10_000`, `RANKING_SAMPLE_SIZE=100_000`.

Toàn bộ 11 notebook FS/FE đều hỗ trợ chạy trên **Google Colab** (tự mount Drive, tự `git clone`
repo nếu chưa có) hoặc **local** (qua biến môi trường `UAV_BASE_DIR`/`UAV_CODE_DIR`), không cần
sửa code khi đổi môi trường.

---

## 9. Việc đã làm — tổng hợp ngắn

✅ Thiết kế toàn bộ pipeline 7 phase theo đúng đề cương (CLAUDE.md).
✅ Thu thập 3 bài báo liên quan ([paper/](paper/)) làm related work.
✅ Tiền xử lý dữ liệu hoàn chỉnh, có kiểm tra và **vá lỗi data leakage nghiêm trọng** (timestamp → session identity).
✅ Thiết kế lại chính sách giữ feature (16 → 72 feature) dựa trên hiểu đúng ý nghĩa missing value của dữ liệu mạng.
✅ Giải quyết vấn đề OOM khi đọc CSV lớn (chunked reading).
✅ Code hoàn chỉnh, resumable, chạy được cả local/Colab cho: baseline + tuning, 6 phương pháp FS, 5 phương pháp FE, và 2 notebook gộp/so sánh kết quả.
✅ Thiết kế đúng các ràng buộc kỹ thuật của từng phương pháp (LDA supervised + giới hạn n_classes-1, KernelPCA fit-subsample + batch transform, AutoEncoder fit-subsample).

## 10. Việc chưa làm — kế hoạch tiếp theo

⬜ **Chạy [01_baseline.ipynb](notebook/01_baseline.ipynb)** → sinh `models/best_params.json` (bắt buộc, mọi notebook sau phụ thuộc vào file này).
⬜ Chạy `02_00` → `02_06` (Feature Selection) → `02_99` (gộp + so sánh với baseline).
⬜ Chạy `03_01` → `03_05` (Feature Engineering) → `03_99` (gộp + so sánh FS vs FE).
⬜ Viết phần "Kết quả & Phân tích" của báo cáo/luận văn dựa trên số liệu thật sau khi chạy xong.
⬜ Theo yêu cầu reviewer (đề cương §15): Related Work table, per-class metrics, FPR/FNR, PR-AUC,
  Mean±Std, statistical significance — **phần code đã có sẵn cho tất cả các mục này**, chỉ còn
  chờ chạy thực nghiệm để có số liệu điền vào.

## 11. Rủi ro / điểm cần phản biện khi review

1. **Temporal hold-out, không phải true leave-session-out** (mục 4.3) — vì hầu hết lớp chỉ có 1
   session. Cần làm rõ giới hạn này, tránh bị reviewer bác vì "data leakage còn sót".
2. **LDA/StatFeatures tại K=16 dùng ít chiều thật hơn nêu danh** (9 và 10) — cần trình bày rõ
   `actual_K` khi so sánh, tránh so sánh không công bằng.
3. **KernelPCA và AutoEncoder fit trên subsample** (2,000 và 200,000 mẫu) thay vì full 3.4M dòng
   — cần nêu rõ đây là quyết định thực tế (memory/compute), không phải sơ suất, và đánh giá xem
   subsample này có đại diện đủ tốt không.
4. **Encode device-identity feature (MAC/IP) bị bỏ qua hoàn toàn** ở pha hiện tại — vẫn là
   "future work", có thể bị hỏi tại sao chưa làm trong khi đề cương coi đây là tín hiệu quan trọng.
5. **Chưa có số liệu thực nghiệm nào** — báo cáo hiện tại là báo cáo về *thiết kế và tiền xử lý*,
   chưa phải báo cáo *kết quả*. Cần nói rõ ràng điều này với người review để tránh kỳ vọng sai.

---

*Báo cáo này được tạo dựa trên kiểm tra trực tiếp mã nguồn, notebook, và file dữ liệu trong repo
tại thời điểm 2026-06-20.*
