# MOODIFY
 _Single destination to all your music needs_
 
---

Have you ever felt like the world is against you? Ever feel you can not keep up with the race of life. Ever wanted to take a break, put your headphones on, and listen to some tunes, but can not decide what to listen to? Moodify is here to help you. 
Moodify allows you to start up your webcam and will suggest a playlist based off your mood. 
## Technical Information
I have trained my model using CNN on [FER 2013 dataset](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi3s6q0wrnxAhXIdisKHU0mDOoQFnoECAMQAw&url=https%3A%2F%2Fwww.kaggle.com%2Fmsambare%2Ffer2013&usg=AOvVaw3t4rPsegSFubk8eSGcPlll) that has been provided in the data folder. The data consists of 48x48 pixel grayscale images of faces that are categorized into 7 categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral) based on their facial expression. The dataset that I used was part of a [challange](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi3s6q0wrnxAhXIdisKHU0mDOoQFnoECAIQAw&url=https%3A%2F%2Fwww.kaggle.com%2Fc%2Fchallenges-in-representation-learning-facial-expression-recognition-challenge%2Fdata&usg=AOvVaw2k7h0plld9TiTmZ3GfLYYC) that containes a csv file with all the pixel values contained as a string. 
The model that I used has 47 layers which are a combination of Convolutional, Pooling layers and Dense layers. Total trainable parameters are 8,333,959 and I used Adam as the optimizer. I trained the model for 150 epoch with a batch size of 64, and made checkpoints for every reduction in loss.

The program uses your webcam to get a frame, uses [Haar Cascades](http://alereimondo.no-ip.org/OpenCV/34/) to get the face from the frame, crops it out and sends the face to the model for prediction. Once the mood is detected, it randomly decides a genre according to the mood and uses spotify api to fetch a playlist of that genre. 
## Dependencies and Requirements
Moodify is coded in python and will require python 3.7.0 or above to work properly.
It also need some dependent libraries to run properly. These are as follows:
- OpenCV
- Numpy
- Tensorflow

Hardware requirements are as follows:
- Webcam
- Internet connection

If you want to train the model yourself, both the dataset and the code are provided in the folder. But you will need a few more dependencies for it.

**These are not required for runnning the application, just if you want to train the model yourself.**
- Keras
- Pandas
- Matplotlib
- GPU (Although not necesaary, but recomended if you dont want to be stuck in training phase for 60 hours.)
 
Another thing you can do if you dont have a GPU is use [Google Collab](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwir9vnWxrnxAhXhQ3wKHbehCzcQFnoECAUQAw&url=https%3A%2F%2Fresearch.google.com%2Fcolaboratory%2F&usg=AOvVaw38J01zt_Dlb6pQ1fe6FGrI) and run your code there. It provides a GPU runtime and has all the libraries preloaded.

---
### Footnote:
*Moodify is still a work in progress app that I designed for my summer training of 2021. I have a lot of plans for it in the future. It might be full of bugs as I am an student/independent dev. If you want to report any bugs or contact me in general, you can mail me at chiragcjtiwari@gmail.com or contact me on [linkedin](https://www.linkedin.com/in/chirag-tiwari-rao-SoftwareDev/)*