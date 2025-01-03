﻿# Iris-Flower
# Iris Flower Classification App

This project is a **Streamlit-based web application** for classifying Iris flower species using a machine learning model. The application allows users to input flower measurements and provides a prediction of the flower's species along with additional details and visuals. Below are the main features and instructions for using the app.

---

## Features

- **User-Friendly Interface**: Interactive sliders and input fields to specify Sepal and Petal dimensions.
- **Real-Time Predictions**: Predicts the Iris species based on user inputs using a pre-trained Random Forest Classifier.
- **Probability Display**: Shows the prediction confidence for each Iris species.
- **Species Information**: Provides descriptions and images for the predicted species.
- **Educational Tool**: Explains terms like Sepal and Petal for better understanding.

---

## Requirements

To run the application, ensure you have the following installed:

- Python 3.8 or later
- Required Python libraries:
  ```bash
  pip install pandas numpy streamlit scikit-learn pillow
  ```

---

## File Structure

```
project-directory/
├── app.py                 # Main Streamlit app file
├── IRIS.csv               # Dataset containing Iris flower measurements
├── images/                # Folder containing images of the flower species
│   ├── setosa.png
│   ├── versicolor.png
│   └── virginica.png
```

---

## How to Run

1. Clone this repository or download the files.
2. Navigate to the project directory.
3. Run the following command to launch the app:
   ```bash
   streamlit run app.py
   ```
4. Open the provided URL in your browser to interact with the app.

---

## Dataset

The application uses the classic **Iris dataset**. It contains measurements of Sepal Length, Sepal Width, Petal Length, and Petal Width for three flower species:

- **Setosa**
- **Versicolor**
- **Virginica**

---

## Application Workflow

1. Users enter the dimensions of Sepal and Petal using input fields.
2. The app uses a **Random Forest Classifier** trained on the Iris dataset to predict the flower species.
3. Prediction results include:
   - **Predicted Species Name**
   - **Prediction Probability**
   - **Description** of the species
   - **Image** of the species

---

## Custom Styling

The app is designed with a clean and modern look, featuring:

- A responsive layout
- Custom colors and fonts for better readability
- Styled input elements and buttons

---

## Future Improvements

- Adding more datasets and flower species for classification.
- Providing advanced visualization of input data.
- Deploying the app to a cloud platform like Heroku or AWS for public access.

---

## Credits

- **Dataset**: The Iris dataset is a publicly available dataset often used in machine learning.
- **Images**: Flower images are stored in the `images/` directory.

Feel free to contribute or suggest improvements to the project!

