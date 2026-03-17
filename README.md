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
```
# Name:Somalaraju Rohini
# Register Number:212224240156
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple Feedforward Network
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

        # Store training history
        self.history = {'loss': []}

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)


# Name: Somalaraju Rohini
# Register Number: 212224240156
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        # Forward pass
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store loss
        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information




<img width="655" height="258" alt="Screenshot 2026-03-17 185650" src="https://github.com/user-attachments/assets/f8597cf2-b6fe-4466-92c1-af1b216b3c39" />



## OUTPUT

### Training Loss Vs Iteration Plot



<img width="1118" height="628" alt="Screenshot 2026-03-17 185723" src="https://github.com/user-attachments/assets/63723ada-1cf7-453c-bc20-f58b5c6936c3" />




### New Sample Data Prediction




<img width="1059" height="263" alt="Screenshot 2026-03-17 185805" src="https://github.com/user-attachments/assets/50b63a5f-c917-46da-999a-bb1de748d5dd" />



## RESULT

The dataset contains 30 records with 2 columns: YearsExperience and Salary, with no missing values.
It shows a positive relationship where salary increases as years of experience increases.
