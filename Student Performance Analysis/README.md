# Student Performance Analysis #

This data approach student achievement in secondary education of two Portugese schools. The data attributes includes student grades, demographic, social and school related features. It was collected by using school reports and questionnaires.  Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por). In [Cortez and Silva, 2008], the two datasets were modeled under binary/five-level classification and regression tasks. Important note: the target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 1st and 2nd period grades.

The dataset itself can be found on the [UC Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)

## Folder Structure ##

1. Notebook: It consists of data_preparation.ipynb and model_training.ipynb file which we will used for EDA and model training purpose respectively
2. src: It is a flask app to predict the student performance based on their stats.
3. Artifacts: It consists of the model files and the dataset files.

### Attribute Information ###

Attribute information can be found from the student.txt file which is located under the artifacts folder
