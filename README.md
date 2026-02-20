# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
A neural network regression model is used to predict continuous numerical values from input data. It learns the relationship between input and output variables by adjusting weights through training. The network consists of an input layer, hidden layers, and an output layer that produces the predicted value. Hidden layers use activation functions like ReLU to capture complex patterns. The model is trained using a loss function such as Mean Squared Error (MSE) to measure prediction error. Backpropagation updates weights to minimize this error. Data normalization improves training stability and performance. After training, the model is tested on unseen data to evaluate its prediction accuracy.
## Neural Network Model






<img width="888" height="915" alt="Screenshot 2026-02-20 154442" src="https://github.com/user-attachments/assets/85cb7ef2-0be7-4e23-a384-20900bea044e" />



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Somalaraju Rohini
### Register Number: 212224240156
```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Salary_Data.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

print("\nSalary Dataset:")
print(data.head(10))   # 👈 displays table like Excel view

X = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32).view(-1, 1)

# Normalize input
X = X / X.max()
class SalaryPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SalaryPredictor()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
losses = []

for epoch in range(500):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.show()

print("\nNew Sample Data Prediction")

sample = torch.tensor([[0.9]], dtype=torch.float32)  # normalized input
prediction = model(sample)

print(prediction)



```
## Dataset Information




<img width="382" height="284" alt="Screenshot 2026-02-06 155525" src="https://github.com/user-attachments/assets/0c071d67-970d-4b2c-8d73-89c8f6a6e0af" />


## OUTPUT

### Training Loss Vs Iteration Plot




<img width="749" height="609" alt="Screenshot 2026-02-06 155535" src="https://github.com/user-attachments/assets/06c388a9-47ab-4d6a-8a40-1de4ca4cef56" />



### New Sample Data Prediction




<img width="729" height="127" alt="Screenshot 2026-02-06 155817" src="https://github.com/user-attachments/assets/053a2e49-dfdc-41b2-b4eb-1ae6b1523f7c" />





<img width="653" height="81" alt="Screenshot 2026-02-06 155542" src="https://github.com/user-attachments/assets/beeb37c3-3199-4cad-90af-13f52de77e19" />


## RESULT

Include your result here
