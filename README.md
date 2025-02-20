# 📌 **Flask API for Bottle & Can Size and Brand Prediction**  

## 🏷️ **Overview**  
This API utilizes **Convolutional Neural Networks (CNN)** built with **TensorFlow** to classify **bottle and can size** and **brand** from an uploaded image.  

## 🧠 **Model Architecture**  
The project consists of two separate APIs for predicting bottle and can properties:  

- **Bottle API** → Predicts the bottle's size and brand.  
- **Can API** → Predicts the can's size and brand.  

> 🎯 If the confidence score is below `0.5`, the API will return `"unknown"`.  

---

## 🚀 **Setup & Installation**  
### 1⃣ Install Python and Dependencies  
Run the following command to install required libraries:  
```bash
pip install -r requirements.txt
```
> 🔹 **Recommended Python Version**: 3.8+  

### 2⃣ Download the Pretrained Model Files  
Since the model files are too large to be included in the repository, you will need to download them from the provided Google Drive link.  

- **[Download the model files here](https://drive.google.com/drive/folders/1lduS0K_6Qn_KHlAnk7JgnOENbIOg1NZY?usp=sharing)**  
- After downloading, extract the contents and place them in the `model/` directory.

### 3⃣ Run the API Server  
```bash
python model/api.py
```
The server will be available at: `http://127.0.0.1:5000`  

---

## 🔍 **API Usage**  
### **POST** `/predict/bottle`
- **Description**: Accepts an image of a bottle and predicts its size and brand.  
- **Content-Type**: `multipart/form-data`  
- **Request Parameters**:  
  - `file`: The image file of the bottle.  

#### ✅ **Example API Request (cURL)**
```bash
curl -X POST "http://127.0.0.1:5000/predict/bottle" \
     -F "file=@sample_bottle.jpg"
```

#### 📩 **Response Example**
```json
{
    "size": "bottle_500ml",
    "brand": "coke"
}
```
- If the confidence score is low:  
```json
{
    "size": "unknown",
    "brand": "unknown"
}
```

---

### **POST** `/predict/can`
- **Description**: Accepts an image of a can and predicts its size and brand.  
- **Content-Type**: `multipart/form-data`  
- **Request Parameters**:  
  - `file`: The image file of the can.  

#### ✅ **Example API Request (cURL)**
```bash
curl -X POST "http://127.0.0.1:5000/predict/can" \
     -F "file=@sample_can.jpg"
```

#### 📩 **Response Example**
```json
{
    "size": "can_330ml",
    "brand": "pepsi"
}
```
- If the confidence score is low:  
```json
{
    "size": "unknown",
    "brand": "unknown"
}
```

## 📚 **Testing API with Postman**  
### **Step 1: Open Postman**  
If you don’t have **Postman**, download it from [here](https://www.postman.com/downloads/).  

### **Step 2: Create a New Request**  
1. Open Postman and select **"New Request"**.  
2. Set the request method to **POST**.  
3. Enter the API endpoint:  
   ```
   http://127.0.0.1:5000/predict/bottle
   ```  

### **Step 3: Upload an Image**  
1. Go to the **Body** tab.  
2. Select **form-data**.  
3. Add a new **key** with:  
   - **Key:** `file`  
   - **Type:** `File`  
   - **Value:** (Select an image file of a bottle)  

### **Step 4: Send the Request**  
Click the **"Send"** button.  

### **Step 5: Check the Response**  
If successful, you will get a JSON response like:  
```json
{
    "size": "bottle_500ml",
    "brand": "coke"
}
```
If the model is not confident enough:  
```json
{
    "size": "unknown",
    "brand": "unknown"
}
```

---

---

## 📂 **Project Structure**  
The project has the following directory structure:

```
bottle-can-prediction-api/
├── model/                  
│   ├── api.py              # Main Flask API server
│   ├── bottle_size_model.h5
│   ├── bottle_brand_model.h5     
│   ├── can_size_model.h5       
│   ├── can_brand_model.h5       
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── data/                   # (Optional) Dataset used for training models

```



