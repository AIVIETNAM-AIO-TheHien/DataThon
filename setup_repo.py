import os

# Cấu trúc thư mục
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/data_prep",
    "src/models",
    "report/template",
    "submissions"
]

# Tạo thư mục và file .gitkeep
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    # Tạo file .gitkeep để push được thư mục rỗng lên GitHub
    with open(os.path.join(folder, ".gitkeep"), "w") as f:
        pass

# Tạo .gitignore (cực kỳ quan trọng để không push data lên)
gitignore_content = """# Data - Tuyệt đối không push file csv lên đây
data/
!data/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.env
venv/
env/
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db
"""
with open(".gitignore", "w", encoding="utf-8") as f:
    f.write(gitignore_content)

# Tạo requirements.txt để đảm bảo tính tái lập (Reproducibility)
req_content = """pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
jupyter
"""
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(req_content)

# Tạo README.md
readme_content = """# Gridbreaker Datathon 2026

Repo quản lý code và phân tích cho cuộc thi Datathon 2026.

## Cấu trúc thư mục:
- `data/`: Chứa dữ liệu (đã được ignore, ae tự tải file csv từ Kaggle bỏ vào đây).
- `notebooks/`: Phân tích EDA, Feature Engineering và SHAP.
- `src/`: Source code tiền xử lý dữ liệu và train mô hình dự báo.
- `report/`: Báo cáo kỹ thuật định dạng LaTeX (NeurIPS).
- `submissions/`: File kết quả nộp lên Kaggle.
"""
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

# Tạo sẵn mấy file notebook và script trống
files_to_create = [
    "notebooks/01_EDA_Visualizations.ipynb",
    "notebooks/02_Feature_Engineering.ipynb",
    "notebooks/03_Model_SHAP_Analysis.ipynb",
    "src/data_prep/clean_data.py",
    "src/models/train.py",
]

for file in files_to_create:
    with open(file, "w", encoding="utf-8") as f:
        f.write("")

print("Tạo xong cấu trúc repo! Sẵn sàng push lên GitHub.")