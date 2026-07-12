import requests
import pandas as pd
import json

def jalankan_prediksi():
    # Load data test buat simulasi input
    df_uji = pd.read_csv('../Membangun_model/hotel_bookings_preprocessing/test.csv')
    
    # Ambil beberapa baris awal aja
    sampel_tes = df_uji.drop(columns=['is_canceled']).head(5)
    
    # MLflow butuh format spesifik kayak dataframe_split
    payload_data = {
        "dataframe_split": sampel_tes.to_dict(orient="split")
    }

    # Endpoint Flask buat nge-hit prometheus exporter
    target_url = "http://127.0.0.1:8001/predict"

    header_req = {
        "Content-Type": "application/json"
    }

    print("Kirim request prediksi ke:", target_url)
    print("Isi datanya:")
    print(json.dumps(payload_data, indent=2))
    
    try:
        respon = requests.post(target_url, json=payload_data, headers=header_req)
        if respon.status_code == 200:
            print("\nHasil prediksi:")
            print(respon.json())
        else:
            print(f"\nAda error nih: {respon.status_code}")
            print(respon.text)
    except requests.exceptions.ConnectionError:
        print("\nGagal connect! Pastiin MLflow serve jalan di port 5002 dan exporter Flask jalan di port 8001.")

if __name__ == "__main__":
    jalankan_prediksi()
