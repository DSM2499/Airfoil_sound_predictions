# ML Pipeline for Airfoil Noise Prediction

### Motivation:

This project is driven by the critical need to accurately predict sound levels generated by airfoils, a key factor in optimizing aerospace components. In an era where environmental impact and public acceptance are paramount, understanding and minimizing noise pollution from aircraft and other aerospace technologies has become increasingly important.
    
By harnessing the capabilities of machine learning and big data analytics through Apache Spark, this project aims to develop a robust predictive model. This model will serve as a valuable tool for engineers and researchers, enabling data-driven decision-making in airfoil design and noise reduction strategies.
    
The project encompasses several key phases. It begins with data preparation, involving the cleaning and preprocessing of a large dataset to ensure quality input for the model. This is followed by the construction of a scalable and efficient machine learning pipeline designed to handle big data effectively. Finally, the model undergoes rigorous evaluation using various regression metrics to assess its performance.
    
The ultimate objective is to create a reliable, high-accuracy tool for predicting airfoil-generated sound levels. This endeavor not only addresses a significant challenge in aerospace engineering but also contributes to broader efforts in noise reduction and environmental sustainability in aviation.
    
By bridging the gap between big data analytics and aerospace engineering, this project aims to drive innovation in airfoil design, potentially leading to quieter, more efficient aircraft and improved quality of life in areas affected by aviation noise.

![Airfoil Sound Prediction 1](https://camo.githubusercontent.com/b43aaf71ea89eab01f6921bccdb8a5f93fc8f1589753676299752de36a856a0a/68747470733a2f2f63662d636f75727365732d646174612e73332e75732e636c6f75642d6f626a6563742d73746f726167652e617070646f6d61696e2e636c6f75642f49424d536b696c6c734e6574776f726b2d424430323331454e2d436f7572736572612f696d616765732f416972666f696c5f616e676c655f6f665f61747461636b2e6a7067)

![Airfoil Sound Prediction 2](https://camo.githubusercontent.com/157ef2b6c697b82c0b747a554167df1297119d43e84fc288f87afb6973d2a181/68747470733a2f2f63662d636f75727365732d646174612e73332e75732e636c6f75642d6f626a6563742d73746f726167652e617070646f6d61696e2e636c6f75642f49424d536b696c6c734e6574776f726b2d424430323331454e2d436f7572736572612f696d616765732f416972666f696c5f776974685f666c6f772e706e67)

### Approach:

In this project, I developed a robust predictive model for airfoil-generated sound levels utilizing Apache Spark and advanced machine learning techniques. The process began with the creation of a Spark session to efficiently manage data processing tasks. After acquiring the dataset, I loaded it into a Spark DataFrame and performed initial data cleaning, which included removing duplicate rows and addressing missing values.
    
The data preprocessing phase involved renaming relevant columns for clarity and saving the cleaned data to a Parquet file, ensuring efficient storage and retrieval. Upon reloading the data, I designed a comprehensive machine learning pipeline incorporating feature vectorization, scaling, and linear regression.
    
To prepare for modeling, I strategically split the dataset into training and testing sets. The pipeline model was then fitted to the training data and subsequently used to generate predictions on the test data. Model performance was rigorously evaluated using multiple regression metrics, including Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2).
    
Throughout the development process, I prioritized best practices in code formatting and documentation, ensuring clarity and maintainability of the codebase. The final step involved saving the trained pipeline model for future applications.
    
This structured approach resulted in the creation of an efficient and scalable predictive model capable of providing valuable insights into airfoil noise levels. The model serves as a powerful tool for engineers and researchers, facilitating data-driven decision-making in airfoil design and noise reduction strategies.

### Dataset Description:

**Dataset:** [NASA Airfoil Noise Raw](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-BD0231EN-Coursera/datasets/NASA_airfoil_noise_raw.csv)

![Dataset Description](https://github.com/DSM2499/Airfoil_sound_predictions/blob/main/airfoil%20images/Airfoil%20Dataset%20(1).png)

### Model Results:

  ![Model Results](https://github.com/DSM2499/Airfoil_sound_predictions/blob/main/airfoil%20images/Airfoil%20results.png)

### Inferences:

- **Model Performance:**
    - The R-squared value of 0.42 indicates that the model explains about 42% of the variance in the sound levels. While this is a moderate level of explanatory power, it suggests that there are other factors influencing the sound levels that are not captured by the model.
    - The Mean Squared Error (MSE) and Mean Absolute Error (MAE) values indicate the average magnitude of the errors in the model's predictions. The MAE of 3.88 suggests that on average, the model's predictions are off by about 3.88 decibels.

  - **Impact of Features:**
    - **Frequency**: The negative coefficient (-0.00136) suggests that higher frequencies are associated with slightly lower sound levels. This effect is small but consistent.
    - **Angle of Attack**: The negative coefficient (-0.349) indicates that an increase in the angle of attack reduces the sound levels. This suggests that modifying the angle of attack could be a practical way to manage noise levels.
    - **Chord Length**: The significant negative coefficient (-34.41) shows that larger chord lengths are strongly associated with lower sound levels. This suggests that increasing the chord length can significantly reduce noise.
    - **Free Stream Velocity**: The positive coefficient (0.104) indicates that higher free stream velocities are associated with higher sound levels, though the effect is relatively small.
    - **Suction Side Displacement**: The large negative coefficient (-171.16) indicates that increasing the suction side displacement thickness can lead to a significant reduction in noise levels.

  - **Practical Insights:**
    - **Design Adjustments:**
      - **Chord Length**: Given the strong negative impact of chord length on sound levels, designers of airfoils can consider increasing the chord length to reduce noise. This could be particularly useful in applications where noise reduction is critical, such as in commercial aviation or urban drone delivery systems.
      - **Suction Side Displacement**: The significant impact of suction side displacement thickness suggests that adjustments in the airfoil design to increase this thickness can effectively reduce noise levels. This could involve modifications in the surface geometry or the use of materials that enhance this displacement.
  - **Operational Strategies:**
      - **Angle of Attack**: Pilots and operators can adjust the angle of attack during flight operations to manage noise levels. For instance, during takeoff and landing, where noise is a significant concern, adjusting the angle of attack to optimal levels can help minimize noise pollution.
      - **Velocity Management**: Controlling the free stream velocity can be another strategy to manage noise, especially in environments where reducing noise is essential. This might involve optimizing flight paths and speeds to balance performance and noise levels.
  - **Further Research:**
      - The moderate R-squared value suggests that further research is needed to identify additional factors influencing sound levels. This could involve exploring other aerodynamic parameters, material properties, or environmental conditions.
      - Advanced modeling techniques, such as non-linear models or machine learning approaches, could be employed to capture more complex relationships and improve the predictive power of the model.



### Future Work:

Future work on this project presents several promising avenues for enhancement and expansion:
- **Model Sophistication:** Incorporating more advanced machine learning techniques such as random forests, gradient boosting machines, or deep learning architectures could potentially yield improved prediction accuracy. These methods may capture complex relationships in the data that linear regression might miss.
- **Data Enrichment:** Expanding the dataset with more diverse and comprehensive airfoil data could enhance the model's ability to generalize across various designs and operating conditions. This could involve collecting data from a wider range of airfoil types and environmental scenarios.
- **Real-time Analytics:** Implementing a real-time data pipeline using Apache Kafka for streaming data and Spark Streaming for real-time analytics could enable continuous monitoring and prediction of noise levels. This would allow for more dynamic and responsive noise management strategies.
- **Advanced Feature Engineering:** Conducting a more in-depth feature engineering process could uncover additional relevant attributes from the raw data. This might involve deriving new features based on domain knowledge or using automated feature extraction techniques, potentially leading to improved model performance and new insights.
- **User Interface Development:** Integrating the predictive model into a web-based application with an intuitive interface would significantly enhance its accessibility and utility. This would allow engineers and researchers to easily interact with the model, facilitating its incorporation into their design and optimization processes.

By pursuing these enhancements, the project could evolve into a more comprehensive and powerful tool for airfoil noise prediction and management. These advancements would not only improve the model's accuracy and applicability but also increase its value in real-world aerospace engineering scenarios.
