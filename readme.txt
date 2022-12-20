splits => stratified or not

vectors => matrix of one-hot encoded vectors or a single vector

evaluation metrics => try different on svm and mutli-layer perceptron

window-size => 1, 2, 3, 4 (only test each window size on one model, and assume )

stopwords => keep or take out

vector length => make sure that every vector is the same length (i.e. some windows might contain fewer words than others. We should add [unk, unk])

