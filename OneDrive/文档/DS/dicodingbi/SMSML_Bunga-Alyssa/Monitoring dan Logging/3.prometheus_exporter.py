import time
import requests
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

app = Flask(__name__)

# 1. Total hit prediksi
total_prediksi_hit = Counter('hotel_prediksi_total_hit', 'Total hit ke endpoint prediksi')

# 2. Total error saat prediksi
total_error_prediksi = Counter('hotel_prediksi_error_total', 'Total error saat manggil model')

# 3. Request yang lagi jalan (active)
request_aktif = Gauge('hotel_request_aktif', 'Jumlah request prediksi yang lagi jalan')

# 4. Latency
waktu_respon = Histogram('hotel_waktu_respon_detik', 'Lama waktu respon dalam detik', buckets=[0.1, 0.5, 1.0, 2.0, 5.0])

# 5. Processing time
waktu_proses = Summary('hotel_waktu_proses_detik', 'Waktu proses prediksi')

# 6. Total tamu yang diprediksi batal (cancel)
total_prediksi_batal = Counter('hotel_prediksi_batal_total', 'Total tamu yang diprediksi bakal cancel')

URL_MODEL = "http://127.0.0.1:5002/invocations"

@app.route('/predict', methods=['POST'])
@waktu_proses.time()
def lakukan_prediksi():
    total_prediksi_hit.inc()
    request_aktif.inc()
    mulai = time.time()
    
    try:
        # Terusin request ke MLflow serve
        payload = request.json
        header = {"Content-Type": "application/json"}
        hasil_respon = requests.post(URL_MODEL, json=payload, headers=header)
        
        durasi = time.time() - mulai
        waktu_respon.observe(durasi)
        
        if hasil_respon.status_code == 200:
            hasil_pred = hasil_respon.json().get("predictions", [])
            # Handle format beda dari mlflow
            if not isinstance(hasil_pred, list):
                hasil_pred = hasil_respon.json()
                
            # Hitung kalo ada yang diprediksi cancel (asumsi 1 = cancel)
            for hasil in hasil_pred:
                if hasil == 1:
                    total_prediksi_batal.inc()
                    
            request_aktif.dec()
            return jsonify(hasil_respon.json()), 200
        else:
            total_error_prediksi.inc()
            request_aktif.dec()
            return jsonify({"error": "Gagal prediksi model", "detail": hasil_respon.text}), hasil_respon.status_code
            
    except Exception as error_msg:
        total_error_prediksi.inc()
        request_aktif.dec()
        return jsonify({"error": str(error_msg)}), 500

if __name__ == '__main__':
    # Start up the prometheus metrics server on port 8000
    start_http_server(8000)
    print("Prometheus Exporter started on port 8000...")
    
    # Run the Flask app on port 8001
    print("Flask app started on port 8001. Send POST requests to http://127.0.0.1:8001/predict")
    app.run(host='0.0.0.0', port=8001)
