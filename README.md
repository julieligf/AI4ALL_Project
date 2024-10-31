# AI4ALL Project
# Road Condition Classification using Decision Trees

This project utilizes computer vision techniques to classify road conditions (wet, dry, foggy, icy) based on images captured in those conditions. The images are preprocessed, converted to numerical data, and used to train a decision tree model for classification.

### Data Set:
●	National Highway Traffic Safety Administration (NHTSA)
●	Waymo Open Dataset
●	Federal Highway Administration (FHWA)


## Project Structure

```
│
├── /images
│   ├── /wet
│   ├── /dry
│   ├── /foggy
│   └── /icy
│
├── /scripts
│   ├── main.py                # Entry point of your application
│   ├── image_conversion.py     # Script for converting images
│   └── preprocessing.py        # Script for preprocessing images
│
└── /models                    # If you are using machine learning models
    ├── model.py
    └── training.py
```

## Requirements

### Python Packages

Make sure you have the following Python packages installed:

- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

You can install them using `pip`. Open your terminal and run:

```bash
pip install opencv-python numpy scikit-learn
```

### Visual Studio Code Extensions

If you are using Visual Studio Code, consider installing the following extensions to enhance your development experience:

1. **Python**  
   Publisher: Microsoft  
   Description: Provides rich support for the Python language, including IntelliSense and debugging.

2. **Pylance**  
   Publisher: Microsoft  
   Description: Fast, feature-rich language support for Python.

3. **OpenCV Snippets**  
   Publisher: Gauri Shankar Gupta  
   Description: A snippets generator for OpenCV.  
   Install via: [OpenCV Snippets Marketplace](https://marketplace.visualstudio.com/items?itemName=gsGupta.opencv-snippets)

## How to Run the Project

1. Clone the repository or download the project files to your local machine.
2. Ensure you have all the required packages installed as mentioned above.
3. Organize your images in the `/Images` directory with subfolders for each weather condition (wet, dry, foggy, icy).
4. Open a terminal in the project directory and run the main script:

   ```bash
   python scripts/main.py
   ```

## Contributing

Contributions are welcome! Please create a new branch for your feature or bug fix before submitting a pull request. Feel free to open an issue to discuss changes or improvements.

## License

This project is licensed under the MIT License.
