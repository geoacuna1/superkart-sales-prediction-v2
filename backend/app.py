
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart")

# Load the trained model from the same directory as this app.
model = joblib.load("superkart_model.joblib")

@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart System"

@superkart_api.post("/v1/predict")
def predict_sales():
    data = request.get_json()

    required_fields = [
        "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
        "Product_MRP", "Store_Size", "Store_Location_City_Type", "Store_Type",
        "Product_Id_char", "Store_Age_Years", "Product_Type_Category"
    ]

    missing = [field for field in required_fields if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    input_data = pd.DataFrame([{field: data[field] for field in required_fields}])
    prediction = float(model.predict(input_data)[0])

    return jsonify({"Sales": prediction})

@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    if "file" not in request.files:
        return jsonify({"error": "No CSV file was uploaded under the 'file' field."}), 400

    file = request.files["file"]
    input_data = pd.read_csv(file)
    predictions = model.predict(input_data)

    output = input_data.copy()
    output["Predicted_Product_Store_Sales_Total"] = predictions.round(2)

    return output.to_dict(orient="records")

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
