from keras.models import Sequential
from keras.layers import Activation,Dense
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt  

train_datagen=ImageDataGenerator(rescale=1./255,rotation_range =90)
validation_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    "data/train",
    target_size=(64,64),
    batch_size=100
)

validation_generator=validation_datagen.flow_from_directory(
    "data/validation",
    target_size=(64,64),
    batch_size=100
)

test_generator=test_datagen.flow_from_directory(
    "data/test",
    target_size=(64,64),
    batch_size=100
)

#print(train_generator.class_indices)


def main():
    model=Sequential()
    model.add(Conv2D(64,(3,3),input_shape=(64,64,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    #model.summary()
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history=model.fit_generator(
        train_generator,
        epochs=1000,
        steps_per_epoch=10,
        validation_data=validation_generator,
        validation_steps=10
    )

    loss = history . history["loss"]
    val_loss = history. history["val_loss"]
    learning_count= len ( loss)+1

    plt.plot (range (1, learning_count ), loss, marker = "+", label = " loss" )
    plt.plot (range(1, learning_count ) , val_loss, marker = ".", label = "val_loss")
    plt.legend( loc = "best", fontsize = 10)
    plt.xlabel (" learning_count")
    plt.ylabel(" loss" )
    plt.show( )

    accuracy = history . history[ "accuracy" ]
    val_accuracy = history. history[ "val_accuracy" ]
    learning_count = len(loss)+1

    plt.plot(range (1, learning_count), accuracy, marker = "+", label = "accuracy" )
    plt.plot(range (1, learning_count ), val_accuracy, marker=".", label = "val_accuracy" )
    plt.legend(loc = "best", fontsize = 10)
    plt.xlabel(" learning_count ")
    plt.ylabel(" accuracy ")
    plt.show( )
    
    model.evaluate(test_generator)
    model.save("model.h5")
    
if __name__=="__main__":
    main()