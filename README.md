# Ling-Spam Hands-on

Bài thực hành **phân loại email spam** trên bộ dữ liệu Ling-Spam: cho nội dung một email, dự đoán đó là
**spam** (email rác) hay **ham** (email bình thường).

- Notebook gốc (EDA): <https://www.kaggle.com/code/sharmoul/ling-spam-dataset-eda>
- Dataset: <https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset> (license CC0-1.0)

---

## 1. Bài toán

| Khái niệm | Ý nghĩa |
|---|---|
| **Spam** | Email rác: quảng cáo, lừa đảo, gửi hàng loạt không mong muốn (vd. *"great part-time or summer job!"*) |
| **Ham** | Email hợp lệ, bình thường. Trong Ling-Spam là các email học thuật từ mailing list ngôn ngữ học *Linguist* |

Đây là bài toán **phân loại văn bản nhị phân** (binary text classification):

- **Input:** cột `Body` (nội dung email, gồm cả dòng `Subject:`)
- **Output:** cột `Label` — `0` = ham, `1` = spam

## 2. Dữ liệu

File chính: `data/raw/lingSpam.csv`

| Cột | Kiểu | Mô tả |
|---|---|---|
| *(không tên)* | int | Index thừa từ lúc export, bỏ đi khi đọc (`index_col=0`) |
| `Body` | str | Nội dung email |
| `Label` | int | `0` = ham, `1` = spam |

Thống kê (chạy `scripts/count_samples.py`):

| Label | Nhãn | Số mẫu | Tỉ lệ |
|---|---|---|---|
| 0 | ham | 2,172 | 83.38% |
| 1 | spam | 433 | 16.62% |
| | **Tổng** | **2,605** | |

- Không có giá trị thiếu.
- Có **14 email trùng nội dung** → nên `drop_duplicates("Body")` trước khi chia train/test để tránh rò rỉ dữ liệu.
- Dữ liệu **mất cân bằng** (~5 ham : 1 spam) → không chỉ dùng accuracy, cần xem precision / recall / F1.

Dataset trên Kaggle còn 2 file khác (cùng định dạng `Body`, `Label`), có thể dùng để mở rộng / kiểm tra tổng quát hoá:

| File | Dung lượng |
|---|---|
| `lingSpam.csv` | 25.7 MB |
| `enronSpamSubset.csv` | 15.8 MB |
| `completeSpamAssassin.csv` | 10.8 MB |

## 3. Cấu trúc thư mục

```
ling-spam-hands-on/
├── data/raw/                          # dữ liệu CSV tải từ Kaggle
├── notebooks/ling-spam-dataset-eda/   # notebook EDA gốc từ Kaggle + kernel-metadata.json
├── scripts/
│   ├── download_data.sh               # tải lại notebook + dataset từ Kaggle
│   ├── count_samples.py               # đếm số mẫu tổng và theo nhãn
│   ├── tfidf_features.py              # chuyển văn bản -> ma trận TF-IDF
│   └── train_models.py                # train + đánh giá mô hình phân loại
├── requirements.txt
└── README.md
```

## 4. Cài đặt

Yêu cầu: Python ≥ 3.10, macOS / Linux.

```bash
git clone https://github.com/anhdd17/ling-spam-hands-on.git
cd ling-spam-hands-on

python3 -m venv .venv
source .venv/bin/activate          # dòng lệnh sẽ hiện (.venv)
pip install -r requirements.txt
```

### 4.1. (Tuỳ chọn) Tải lại dữ liệu từ Kaggle

Dữ liệu đã có sẵn trong `data/raw/`. Chỉ cần bước này nếu muốn tải lại.

1. Tạo API token: kaggle.com → **Settings** → **API** → **Create New Token**.
2. Cấu hình **một** trong các cách sau (token **không** được commit vào repo):
   - `~/.kaggle/access_token` (chmod 600)
   - biến môi trường `KAGGLE_API_TOKEN`
   - `~/.kaggle/kaggle.json` (chmod 600)
   - biến môi trường `KAGGLE_USERNAME` + `KAGGLE_KEY`
3. Chạy:

```bash
scripts/download_data.sh           # bỏ qua nếu file đã tồn tại
scripts/download_data.sh --force   # ép tải lại
```

## 5. Sử dụng

### 5.1. Đếm số lượng mẫu

```bash
python scripts/count_samples.py                                # mặc định: data/raw/lingSpam.csv
python scripts/count_samples.py data/raw/enronSpamSubset.csv   # file khác
```

Kết quả mẫu:

```
Tổng số mẫu: 2605
Số mẫu thiếu Body: 0
Số mẫu trùng Body: 14

label  count  percent
  ham   2172    83.38
 spam    433    16.62
```

Chưa kích hoạt venv thì gọi trực tiếp: `.venv/bin/python scripts/count_samples.py`.

### 5.2. Chuyển văn bản sang TF-IDF và train mô hình

```bash
python scripts/tfidf_features.py --save   # bỏ trùng, chia train/test 80/20 (stratified), fit TF-IDF trên train
python scripts/train_models.py            # train MultinomialNB, LogisticRegression, LinearSVC và đánh giá
```

`tfidf_features.py` lưu ma trận sparse + vectorizer vào `data/processed/` (không commit, chạy lại là có).
Tuỳ chọn: `--max-features` (mặc định 5000), `--test-size` (0.2), `--top-k` (15).

Kết quả trên tập test (519 email: 434 ham, 85 spam):

| Mô hình | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|---|
| MultinomialNB | 0.9942 | 0.9881 | 0.9765 | 0.9822 |
| LogisticRegression | 0.9942 | 0.9881 | 0.9765 | 0.9822 |
| LinearSVC | 0.9923 | 1.0000 | 0.9529 | 0.9759 |

### 5.3. Chạy notebook EDA

```bash
jupyter lab
```

Mở `notebooks/ling-spam-dataset-eda/ling-spam-dataset-eda.ipynb`.

> **Lưu ý đường dẫn:** notebook gốc đọc `/kaggle/input/email-spam-dataset/lingSpam.csv` (đường dẫn trên Kaggle).
> Khi chạy local, sửa cell `read_csv` thành:
>
> ```python
> ling_spam_df = pd.read_csv("../../data/raw/lingSpam.csv")
> ```
>
> (Jupyter chạy notebook với thư mục làm việc là thư mục chứa notebook.)

Notebook gồm các bước:

1. Đọc dữ liệu, bỏ cột index thừa, kiểm tra missing values
2. Biểu đồ tròn tỉ lệ ham / spam
3. In 10 email mẫu mỗi lớp
4. Word cloud toàn bộ dữ liệu và riêng từng lớp
5. Phân phối số từ mỗi email theo nhãn (histogram + `describe()`)

### 5.4. Chạy trong VS Code

1. `Cmd+Shift+P` → **Python: Select Interpreter** → chọn `./.venv/bin/python`.
2. Mở file `.py` → bấm ▶ **Run Python File**, hoặc mở `.ipynb` và chọn kernel `.venv`.

## 6. Hướng phát triển

Notebook hiện tại mới dừng ở EDA. Hướng xây mô hình gợi ý:

1. **Tiền xử lý:** lowercase, bỏ stopword, tách từ, bỏ email trùng.
2. **Chia dữ liệu:** `train_test_split(..., stratify=y)` hoặc `StratifiedKFold` để giữ tỉ lệ spam/ham.
3. **Biểu diễn:** Bag-of-Words / TF-IDF (`CountVectorizer`, `TfidfVectorizer`).
4. **Mô hình:** Multinomial Naive Bayes (baseline kinh điển), Logistic Regression, Linear SVM.
5. **Đánh giá:** precision, recall, F1, confusion matrix. Với lọc spam, **precision lớp spam** rất quan trọng
   — đánh nhầm email thật thành spam gây hại hơn để lọt một email spam.

## 7. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân / cách sửa |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Đang dùng Python hệ thống. Kích hoạt `.venv` hoặc chọn đúng interpreter. |
| `FileNotFoundError: /kaggle/input/...` | Chưa sửa đường dẫn trong notebook — xem mục 5.3. |
| `ERROR: no Kaggle credentials found` | Chưa cấu hình token — xem mục 4.1. |
| `401 Unauthorized` khi tải từ Kaggle | Token sai / đã bị thu hồi. Tạo token mới. |
