## CAR CLASSIFICATION PROJECT
*K-NEAREST NEIGHBOURS*

a classification algorithm / deals with irregular data

###  Attributes
1. Buying cost _"buying"
2. Maintenance _"maint"
3. Number of doors _"door"
4. Number of Persons _"persons"
5. Boot size _"lug_boot"
6. Safety degree _"safety"

###  Label / The prediction
7. class = ["unacc", "acc", "good", "vgood"]

### Requirements
1. pandas
   
    ```python import pandas as pd ```

2. Numpy
   
    ```python import numpy as np ```


4. sklearn
   
    ```python 
    import sklearn 
    from sklearn import linear_model, preprocessing
    from sklearn.utils import shuffle 
    from sklearn.neighbors import KNeighborsClassifier
    ```


## Steps of the project:

STEP 1:

- We have to read in our dataset. Using panda.

    ```python 
    data = pd.read_csv("car.data")
    ```

STEP 2:

- To encode the non-integral data values

    ```python 
    encode = preprocessing.LabelEncoder()

    buying = encode.fit_transform(list(data["buying"]))
    maint = encode.fit_transform(list(data["maint"]))
    door = encode.fit_transform(list(data["door"]))
    persons = encode.fit_transform(list(data["persons"]))
    lug_boot = encode.fit_transform(list(data["lug_boot"]))
    safety = encode.fit_transform(list(data["safety"]))
    cls = encode.fit_transform(list(data["class"]))
    ```

STEP 3: 

- Here we define what we want to predict. The label.

    ```python
    predict = "cls"
    ```

    ```python
    x = list(zip(buying, maint, door, persons, lug_boot, safety)) 
    ```

- This line defines attributes that will help with prediction.
  
    ```python 
    y = list(cls)
    ```
- This line gives only the '**class**' value.
  
STEP 4:

- We divide x & y into four('x train', 'y train', 'x test', 'y test'). Using sklearn!
  
    ```python
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) 
    ```

- This line splits our data X and Y, This is for TRAINING & TESTING.
  
- The line 'test_size=0.1', means that from our dataset 10% will be used for testing.


STEP 5:
- create the training model.
  
    ```python 
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(x_train, y_train)
    ```
- **n_neighbours=7** , means a maximum of 7 neighbours is allowed

- You can use more or less eg: **n_neighbours=5**, **n_neighbours=10** etc..

    ```python 
    accuracy = model.score(x_test, y_test)
    print(accuracy)
    ```

- **model-score**, finds how accurate the model is!

## HOW K-NEAREST NEIGHBOURS works!!!
  
![Web capture_20-6-2023_91031_app whiteboard microsoft com](https://github.com/edyprogramz/Car-Classification-Project/assets/116636391/69c5b0f2-4812-4349-b5c9-99db8fd6255f)
<br>
![Web capture_20-6-2023_91842_app whiteboard microsoft com](https://github.com/edyprogramz/Car-Classification-Project/assets/116636391/bc4ae833-3528-40b2-bf6f-3a5fae2f9044)
<br>
![Web capture_20-6-2023_93627_app whiteboard microsoft com](https://github.com/edyprogramz/Car-Classification-Project/assets/116636391/325f6ae2-dfb1-4008-abc9-01d4061dd745)

STEP 6:

    ```python 
    prediction = model.predict(x_test)
    names = ["unacc", "acc", "good", "vgood"]

    for x in range(len(prediction)):
        print(names[prediction[x]], x_test[x], names[y_test[x]])
        # or
        print("Predicted: ", names[prediction[x]], "Actual: ", names[y_test[x]])
        
    ```

- On the first line our model makes a prediction.
- A for loop to iterate through each prediction!
- names variable & n helps give proper class names to our output.
  
STEP 7:

- Saving our model

- If we were to save our model it would consume alot of space, it's one limitations of this algorithm.

- This because of the magnitude calculations of each single value with all the other possibilities.

The END

