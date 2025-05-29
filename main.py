import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from utils.multi_target_pipeline import run_pipeline_for_all_targets
from utils.data_source_utils import GetRateData


if __name__ == "__main__":
    # Sadece belirli parametreler için eğitim ve tahmin
    selected_columns = [
        'buyPrice', 'sellPrice'  # Örnek: sadece bu kolonlar için
    ]
    # source_type 'api' veya 'local' olabilir
    csv_files = GetRateData(source_type='local')
    for csv_path in csv_files:
        output_dir = os.path.splitext(os.path.basename(csv_path))[0]
        run_pipeline_for_all_targets(
            csv_path, output_dir, target_columns=selected_columns,
            time_step=100, epochs=50, batch_size=64, future_days=30
        )