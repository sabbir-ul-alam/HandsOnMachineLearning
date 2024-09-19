import models
def train_model(model,example, label):
    model.fit(example,label)
    return model



if __name__ == '__main__':
    from data_preprocessing import prep_data

    housing, housin_label = prep_data()
    model = train_model(models.make_LinearRegression(),housing,housin_label)
    predicted_data = model.predict(housing)
    print(predicted_data[:5].round(-2))