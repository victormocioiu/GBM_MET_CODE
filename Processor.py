__author__ = 'Sunny'

class Proc_unit(object):
    '''Wrapper class to help start the processing part
    DON'T FORGET TO WRITE THE DOCS STUPID

    '''
    def __init__(self,DataHolder,classifiers):
        self.data = DataHolder
        self.classifiers = classifiers
        self.cat_predictions = []
        self.prob_predictions = []

    def start(self):
        self.data.extract_features()
        for classifier in self.classifiers:
            classifier.fit(self.data.train_features,self.data.y_train)
            self.cat_predictions.append((classifier.predict(self.data.test_features)))
            self.prob_predictions.append(classifier.predict_proba(self.data.test_features))