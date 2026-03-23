import tkinter as tk
import requests

def predict():
    text = input_text.get("1.0", tk.END).strip()

    if not text:
        result_label.config(text="Enter some text")
        return

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": text}
        )

        result = response.json()
        result_label.config(text="Sentiment: " + result["sentiment"])

    except Exception as e:
        result_label.config(text="Error connecting to API")
        print(e)


# ================================
# GUI
# ================================
root = tk.Tk()
root.title("Sentiment Analyzer")
root.geometry("600x400")

label = tk.Label(root, text="Enter text", font=("Arial", 14))
label.pack(pady=10)

input_text = tk.Text(root, height=6, width=50)
input_text.pack()

predict_btn = tk.Button(root, text="Predict Sentiment", command=predict)
predict_btn.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack()

root.mainloop()