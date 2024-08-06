DATASET: https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset/data

# The structure of the dataset
This dataset has multiple classes, thus we will be dealing with **multiclass classification**. 
> **StudentID**, type=int, predictor variable
>
> **Age**, type=int, predictor variable
>
> **Gender**, type=int, predictor variable
>
> **Ethnicity**, type=int, predictor variable
>
> **ParentalEducation**, type=int, predictor variable
>
> **StudyTimeWeekly**. type=float, predictor variable
>
> **Absences**, type=int, predictor variable
>
> **Tutoring**, type=int, predictor variable
>
> **ParentalSupport**, type=int, predictor variable
>
> **Extracurricular**, type=int, predictor variable
>
> **Sports**, type=int, predictor variable
>
> **Music**, type=int, predictor variable
>
> **Volunterring**, type=int, predictor variable
>
> **GPA**, type=float, predictor variable
>
> **GradeClass**, type=int, **response variable**

As you can see this dataset has already done encoding on the categorical features, thus we require few preprocessing steps before training the model.
