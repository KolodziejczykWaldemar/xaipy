# xaipy
Library for e**X**plainable **AI** in **PY**thon!

## Description
Library with model agnostic methods for explaining regression and classification models. Implementations based on 
[Christoph Molnar's book *"Interpretable Machine Learning"*](https://christophm.github.io/interpretable-ml-book).

Methods covered in this library:
* PDP - Partial Dependence Plot (1D and 2D support)
* ICE Plot - Individual Conditional Expectation Plot
* ALE Plot - Accumulated Local Effects Plot (only 1D support)
* Permutation Importance

All methods are built assuming *scikit-learn*-like interface of models:
```python
# regression case
y_pred = model.predict(X)

# classification case
y_pred_proba = model.predict_proba(X)
```

## Authors
Waldemar Kołodziejczyk - kolodziejczykwaldemar222@gmail.com