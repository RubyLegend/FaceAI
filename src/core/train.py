# Code is blatantly stolen from github.
# I'll learn it a little bit later
# and probably rewrite it
# because I want to have full understanding of what is happening here
# and most importantly, how and why.

# Right now, this code won't even work, because I 
# haven't copied all functions

"""# Learning phase.
We write the generators that will give our model batches of data to train on, then we run the training.
"""

def generator(batch_size):
  
  while 1:
    X=[]
    y=[]
    switch=True
    for _ in range(batch_size):
   #   switch += 1
      if switch:
     #   print("correct")
        X.append(create_couple_rgbd("faceid_train/").reshape((2,200,200,4)))
        y.append(np.array([0.]))
      else:
     #   print("wrong")
        X.append(create_wrong_rgbd("faceid_train/").reshape((2,200,200,4)))
        y.append(np.array([1.]))
      switch=not switch
    X = np.asarray(X)
    y = np.asarray(y)
    XX1=X[0,:]
    XX2=X[1,:]
    yield [X[:,0],X[:,1]],y

def val_generator(batch_size):
  
  while 1:
    X=[]
    y=[]
    switch=True
    for _ in range(batch_size):
      if switch:
        X.append(create_couple_rgbd("faceid_val/").reshape((2,200,200,4)))
        y.append(np.array([0.]))
      else:
        X.append(create_wrong_rgbd("faceid_val/").reshape((2,200,200,4)))
        y.append(np.array([1.]))
      switch=not switch
    X = np.asarray(X)
    y = np.asarray(y)
    XX1=X[0,:]
    XX2=X[1,:]
    yield [X[:,0],X[:,1]],y

gen = generator(16)
val_gen = val_generator(4)

outputs = model_final.fit_generator(gen, steps_per_epoch=30, epochs=50, validation_data = val_gen, validation_steps=20)

