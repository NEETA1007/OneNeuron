# OneNueron
OneNueron|Perceptron


#command used -


'''bash
git add . && git commit -m "docstring updated" && git push origin main
''''

'''bash
cp Research\ noteook/demo.ipynb .
'''bash

## Add URL-
[Git handbook](https://guides.github.com/introduction/git-handbook/)

<a href="https://www.w3schools.com">Visit W3Schools.com!</a>>

## Add Images-
![Sample Image](plots/and.png)

<img src="plots/and.png" alt="Girl in a jacket" width="500" height="600">

## Python Code
```python
def main(data, eta, epochs, filename, plotFileName):

    df = pd.DataFrame(data)
    print(df)
    X,y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _=model.total_loss()  # dummy variable

    save_model(model, filename=filename)
    save_plot(df, plotFileName, model)

```