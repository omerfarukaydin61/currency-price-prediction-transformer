import os
import requests
import json
import pandas as pd

def GetRateData(source_type='api', local_dir=None):
    """
    source_type: 'api' veya 'local'.
    'api' -> API'den indirir (mevcut davranış)
    'local' -> Sadece mevcut content klasöründeki csv dosyalarını döndürür
    local_dir: 'local' seçilirse csv dosyalarının bulunduğu klasör (varsayılan: ./content)
    """
    base_url = "https://exrate.kivierp.com/api/ExRate/GetCurrencyGraphData"
    csv_files = []
    if local_dir is None:
        current_dir = os.path.join(os.getcwd(), 'content')
    else:
        current_dir = local_dir

    if not os.path.exists(current_dir):
        os.makedirs(current_dir)

    if source_type == 'local':
        # Sadece mevcut csv dosyalarını döndür
        for fname in os.listdir(current_dir):
            if fname.endswith('.csv'):
                csv_files.append(os.path.join(current_dir, fname))
        return csv_files

    # API'den veri çekme (mevcut davranış)
    for source in range(1,5):
        for ticker in range(1, 26):
            json_filename = os.path.join(current_dir, f"{source}-{ticker}.json")
            csv_filename = os.path.join(current_dir, f"{source}-{ticker}.csv")
    
            # JSON dosyası yoksa indir
            if not os.path.exists(json_filename):
                url = f"{base_url}?ticker={ticker}&source={source}"
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(json_filename, "w", encoding="utf-8") as file:
                        file.write(response.text)
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data for source {source} ticker {ticker}: {e}")
                    continue
    
            # JSON'dan CSV'ye dönüştür
            if not os.path.exists(csv_filename):
                try:
                    with open(json_filename, "r", encoding="utf-8") as file:
                        data = json.load(file)
                    df = pd.DataFrame(data)
                    df.to_csv(csv_filename, index=False)
                except Exception as e:
                    print(f"Error converting {json_filename} to CSV: {e}")
    
            if os.path.exists(csv_filename):
                csv_files.append(csv_filename)
    
    return csv_files
