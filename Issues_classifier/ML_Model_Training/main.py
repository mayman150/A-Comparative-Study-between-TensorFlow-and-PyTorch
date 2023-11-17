def training_loop(model, epochs=1500):
    criterion = torch.nn.BCELoss()
    X_train, X_test, y_train, y_test = load_data()
    model = LogisticRegression(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters())
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    for epoch in range(epochs):
        
        loss = criterion(model(X_train), y_train)
        val_loss = criterion(model(X_test), y_test)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history["loss"].append(loss.data[0])
        history["val_loss"].append(val_loss.data[0])
        history["acc"].append(accuracy(model(X_train), y_train))
        history["val_acc"].append(accuracy(model(X_test), y_test))
    

