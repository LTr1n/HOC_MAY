import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ zoo.csv
data = pd.read_csv("zoo.csv")

# Xác định các nhóm động vật
class_labels = {
    1: "Mammals", 2: "Birds", 3: "Reptiles", 4: "Fish", 5: "Amphibians", 6: "Insects", 7: "Invertebrates"
}
genders = sorted(data['class_type'].unique())  # Các loại động vật

# Chia dữ liệu theo nhóm động vật
animals = {gender: data[data['class_type'] == gender] for gender in genders}

# Các đặc trưng sử dụng để phân loại
selected_features = ['legs', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator']

class Feature:
    def __init__(self, data, name=None, bin_width=None):
        self.name = name
        self.bin_width = bin_width
        self.freq_dict = dict(Counter(data))
        self.freq_sum = sum(self.freq_dict.values())
        self.unique_values = len(set(data))
    
    def frequency(self, value, smoothing=1):
        return (self.freq_dict.get(value, 0) + smoothing) / (self.freq_sum + smoothing * self.unique_values)

# Lựa chọn thuộc tính để phân loại
fts = {gender: {feature: Feature(animals[gender][feature]) for feature in selected_features} for gender in genders}

# Vẽ biểu đồ phân bố số chân của từng nhóm động vật
plt.figure(figsize=(10, 6))
for gender in genders:
    frequencies = sorted(fts[gender]['legs'].freq_dict.items(), key=lambda x: x[0])
    if frequencies:
        X, Y = zip(*frequencies)
        plt.bar(X, Y, label=f'{class_labels[gender]}', alpha=0.75)

plt.legend(loc='upper right')
plt.title("Distribution of Legs Among Animal Classes")
plt.xlabel("Number of Legs")
plt.ylabel("Frequency")
plt.show()

class NBclass:
    def __init__(self, name, features):
        self.features = features
        self.name = name
    
    def log_probability(self, data_point):
        log_prob = 0
        for feature_name in selected_features:
            log_prob += np.log(self.features[feature_name].frequency(data_point[feature_name]))
        return log_prob

cls = {gender: NBclass(class_labels[gender], fts[gender]) for gender in genders}

class Classifier:
    def __init__(self, nbclasses):
        self.nbclasses = nbclasses
    
    def predict(self, data_point, show_all=True):
        log_probs = [(nbclass.log_probability(data_point), nbclass.name) for nbclass in self.nbclasses]
        log_probs.sort(reverse=True, key=lambda x: x[0])  # Sắp xếp theo xác suất giảm dần
        
        # Chuyển đổi log-prob thành xác suất thực tế
        exp_probs = np.exp([p[0] for p in log_probs])
        prob_sum = np.sum(exp_probs)
        probabilities = [(exp_probs[i] / prob_sum, log_probs[i][1]) for i in range(len(log_probs))]
        
        return probabilities if show_all else max(probabilities)

# Tạo feature vector và target label
X = data[selected_features]
y = data['class_type']

# Chia dữ liệu thành 80% huấn luyện và 20% kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện Naive Bayes Classifier
train_fts = {gender: {feature: Feature(animals[gender][feature]) for feature in selected_features} for gender in genders}
train_cls = {gender: NBclass(class_labels[gender], train_fts[gender]) for gender in genders}
classifier = Classifier([train_cls[g] for g in genders])

# Đánh giá trên tập kiểm tra
correct_predictions = 0
for i in range(len(X_test)):
    test_data = X_test.iloc[i].to_dict()  # Chuyển dữ liệu test thành dictionary
    actual_class = y_test.iloc[i]
    predicted_probabilities = classifier.predict(test_data, show_all=False)
    predicted_class = predicted_probabilities[1]  # Lấy class có xác suất cao nhất

    # Kiểm tra kết quả
    if predicted_class == class_labels[actual_class]:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test)
print(f'Accuracy on test set: {accuracy * 100:.2f}%')


# Kiểm tra với một mẫu dữ liệu
sample_data = {'legs': 4, 'hair': 1, 'feathers': 0, 'eggs': 0, 'milk': 1, 'airborne': 0, 'aquatic': 0, 'predator': 1}
print(f'Predicted probabilities for sample: {classifier.predict(sample_data, show_all=True)}')