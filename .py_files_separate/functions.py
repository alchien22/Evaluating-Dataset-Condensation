# Calculate the categorization error
#y: target value
#yhat: predicted value
#cerr: % incorrect
class Functions():
    def eval_cat_err(y, yhat):
        m = len(y)
        incorrect = 0
        for i in range(m):
            if yhat[i] != y[i]:
                incorrect+=1
        cerr = incorrect / m
        
        return(cerr)