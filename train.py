from dataset import get_data
from model import *

# def Cal_

######################## Get train/test dataset ########################
X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")
Linear_train = False
LinearPath = './models/Linear.npy'
LinearParaPath = './models/LinearPara.npy'

Ridge_train = False
RawRidgePath = './models/Ridge'
RawRidgeParaPath = './models/RidgePara'

Lasso_train = True
use_coordinate_descent = True
RawLassoPath = './models/Lasso'
RawLassoParaPath = './models/LassoPara'


######################## Linear regression #############################
print("-------------- Linear Regression --------------")
lr =  0.00002
LinearEpochs = int(1e6)
figName = 'Linear'
Linear = LinearRegression(X_train, Y_train)

if Linear_train:
    for epoch in range(LinearEpochs):
            loss = Linear.loss()
            deri = Linear.derivative()
            Linear.update(lr)
            if(epoch%100000 == 0):
                print(f'Epoch {epoch}: loss = {loss}, derivative = {deri}')
    np.save(LinearPath, Linear.beta)
    np.save(LinearParaPath, np.array([LinearEpochs, lr]))
    print(f"LinearRegression Model has been saved at {LinearPath}.")
else:
    LinearEpochs, lr = np.load(LinearParaPath)
    Linear.load(LinearPath)

polt_beta(Linear.beta, figName)
print(f"Bar chart of Beta has been saved at ./fig/{figName}.png")
SSE = sse_loss(Linear.predict(X_test), Y_test)
print(f"LinearRegression SSE = {SSE}, Parameters: Epochs = {LinearEpochs}, lr = {lr}.")
print()

######################## Ridge regression ##############################
print("-------------- Ridge Regression --------------")
Beta = [Linear.beta, ]
Labelist = ['lambda = 0', ]
lr =  0.000033
Lambdas = [1, 10, 100, 1000]
# RidgeEpochs = int(1e5)
RawfigName = 'Ridge'

for Lambda in Lambdas:
    if Lambda < 100: RidgeEpochs = int(1e6)
    else: RidgeEpochs = int(1e5)
    Ridge = RidgeRegression(X_train, Y_train, Lambda)
    figName = RawfigName + f'_lambda{Lambda}'
    RidgePath = RawRidgePath + f'_lambda{Lambda}.npy'
    RidgeParaPath = RawRidgeParaPath + f'_lambda{Lambda}.npy'
    if Ridge_train:
        # Ridge.analysis_fit(X_train, Y_train)
        for epoch in range(RidgeEpochs):
                loss = Ridge.loss()
                deri = Ridge.derivative()
                Ridge.update(lr)
                # if(epoch%10000 == 0):
                #     print(f'Epoch {epoch}: loss = {loss}, derivative = {deri}')
        np.save(RidgePath, Ridge.beta)
        np.save(RidgeParaPath, np.array([RidgeEpochs, lr, Lambda]))
        print(f"RidgeRegression Model has been saved at {RidgePath}.")
    else:
        RidgeEpochs, lr, Lambda = np.load(RidgeParaPath)
        Ridge.load(RidgePath)

    Beta.append(Ridge.beta)
    Labelist.append(f'lambda = {Lambda}')

    polt_beta(Ridge.beta, figName)
    print(f"Bar chart of Beta has been saved at ./fig/{figName}.png")
    SSE = sse_loss(Ridge.predict(X_test), Y_test)
    print(f"RidgeRegression SSE = {SSE}, Parameters: Epochs = {RidgeEpochs}, lr = {lr}, Lambda = {Lambda}.")

figName = 'Ridge'
plot_multiBeta(Beta, Labelist, figName)
print(f"Line chart of different Betas have been saved at ./fig/{figName}.png")
print()

######################## RBF Kernel regression #########################
print("-------------- RBF Kernel Regression --------------")
C = []
Labelist = []
sigma = 50 # [1, 50, 500, 5000]
Lambdas = [1, 10, 100, 1000]
RawfigName = 'Kernel'

for Lambda in Lambdas:
    Kernel = RBFKernelRegression(X_train, Y_train, Lambda, sigma)
    figName = RawfigName + f'_lambda{Lambda}_Sigma{sigma}'
    # KernelPath = RawKernelPath + f'_lambda{Lambda}.npy'
    # KernelParaPath = RawKernelParaPath + f'_lambda{Lambda}.npy'
    Kernel.analysis_fit()
    C.append(Kernel.c)
    Labelist.append(f'lambda={Lambda}_Sigma={sigma}')

    plot_c(Kernel.c, figName)
    print(f"Bar chart of c has been saved at ./fig/{figName}.png")
    SSE = sse_loss(Kernel.predict(X_test), Y_test)
    print(f"KernelRegression SSE = {SSE}, Parameters: Lambda = {Lambda}, Sigma = {sigma}.")

figName = 'Kernel'
plot_multiC(C, Labelist, figName)
print(f"Line chart of different Cs have been saved at ./fig/{figName}.png")
print()

######################## Lasso regression ##############################
print("-------------- Lasso Regression --------------")
Beta = [Linear.beta, ]
Labelist = ['lambda = 0', ]
lr =  0.0001
Lambdas = [1, 10, 100, 1000]
a = 1.7
b = 0.01
interval = int(10000)
LassoEpochs = int(1e6)
RawfigName = 'Lasso'

if not use_coordinate_descent:
    for Lambda in Lambdas:
        if Lambda < 100: LassoEpochs = int(1e6)
        else: LassoEpochs = int(1e5)
        Lasso = LassoRegression(X_train, Y_train, Lambda)
        figName = RawfigName + f'_lambda{Lambda}'
        LassoPath = RawLassoPath + f'_lambda{Lambda}.npy'
        LassoParaPath = RawLassoParaPath + f'_lambda{Lambda}.npy'
        if Lasso_train:
            for epoch in range(LassoEpochs):
                    loss = Lasso.loss()
                    deri = Lasso.derivative()
                    Lasso.update(lr)
                    # if(epoch%10000 == 0):
                    #     print(f'Epoch {epoch}: loss = {loss}, derivative = {deri}')
            np.save(LassoPath, Lasso.beta)
            np.save(LassoParaPath, np.array([LassoEpochs, lr, Lambda]))
            print(f"LassoRegression Model has been saved at {LassoPath}.")
        else:
            LassoEpochs, lr, Lambda = np.load(LassoParaPath)
            Lasso.load(LassoPath)

        polt_beta(Lasso.beta, figName)
        print(f"Bar chart of Beta has been saved at ./fig/{figName}.png")
        SSE = sse_loss(Lasso.predict(X_test), Y_test)
        print(f"LassoRegression SSE = {SSE}, Parameters: Epochs = {LassoEpochs}, lr = {lr}, Lambda = {Lambda}, Method: GD.")

else:
    Lasso = LassoRegression(X_train, Y_train, Lambda)
    figName = RawfigName + f'_a{a}_b{b}_interval{interval}.png'
    LassoPath = RawLassoPath + f'_a{a}_b{b}_interval{interval}.npy'
    LassoParaPath = RawLassoParaPath + f'_a{a}_b{b}_interval{interval}.npy'
    if Lasso_train:
        Lasso.CoordinateDescent(a, b, interval)
        np.save(LassoPath, Lasso.beta)
        np.save(LassoParaPath, np.array([a, b, interval]))
        print(f"LassoRegression Model has been saved at {LassoPath}.")
    else:
        a, b, interval = np.load(LassoParaPath)
        Lasso.load(LassoPath)

    polt_beta(Lasso.beta, figName)
    print(f"Bar chart of Beta has been saved at ./fig/{figName}.png")
    SSE = sse_loss(Lasso.predict(X_test), Y_test)
    print(f"LassoRegression SSE = {SSE}, Parameters: a = {a}, b = {b}, intervel = {interval}, Method: CD.")

for Lambda in Lambdas:
    path = RawLassoPath + f'_lambda{Lambda}.npy'
    beta = np.load(path)
    Beta.append(beta)
    Labelist.append(f'lambda = {Lambda}')

path = RawLassoPath + f'_a{a}_b{b}_interval{interval}.npy'
beta = np.load(path)
Beta.append(beta)
Labelist.append('Coordinate_Decent')

figName = 'Lasso'
plot_multiBeta(Beta, Labelist, figName)
print(f"Line chart of different Betas have been saved at ./fig/{figName}.png")
print()

print('--------------------------------------------')
print("All Done.")


########################################################################
######################## Implement you code here #######################
########################################################################