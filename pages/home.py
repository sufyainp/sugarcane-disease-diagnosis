import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from PIL import ImageOps
import requests

from streamlit_option_menu import option_menu


primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"


def dense_net():
     st.write('''This project will help to detect and 
            classify whether the sugarcane leaf is 
            diseased or healthy. This model has achieved its goal by 
            effectively detecting and classifying sugarcane images into 
            healthy and diseased category based on the pattern of leaves. 
            ''')
     
    
     
     image1 = Image.open('../images/sugarcane1.jpg')
     st.image(image1, caption=' ',use_column_width=True)
        
    
        # st.write("Accuracy of the Model")
     image1 = Image.open('../images/densenet_i.png')
     st.image(image1, caption=' ',use_column_width=True)


     
    #  image1 = Image.open('../images/sugarcane1.jpg')
    #  st.image(image1, caption=' ',use_column_width=True)
    #  st.write("Accuracy of the Model")
    #  image1 = Image.open('../images/densenet_i.png')
    #  st.image(image1, caption=' ',use_column_width=True)

     instructions = """
            Upload image.
            The image you select or upload will be fed
            through model in real-time
            and the output will be displayed to the screen.
            """
     st.write(instructions)
        
     def load_model():
        loaded_model= tf.keras.models.load_model('../models/densenet_m.h5')
        return loaded_model
     model=load_model() 
     file = st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])
     st.write("Here is the file you uploaded ")
    
     file 

     def import_and_predict(image_data,model):
        size= (256,256)
        image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img =np.asarray(image)
        img_reshape=img[np.newaxis,...]
        prediction123 = model.predict(img_reshape)
        return prediction123

     if file is None:
        st.text("Please Upload file")

     if st.button('Predict',on_click=None):
            st.write('Results')
            image=Image.open(file)
            st.image(image,use_column_width=True)
            prediciton= import_and_predict(image,model)
            class_names=["Bacterial Blight","Healthy","Red Rot", "Red Rust"]
            if (class_names[np.argmax(prediciton)])=='Healthy':
                string="This plant is Healthy"
                st.success(string)
            else:
                string="This plant has :"+class_names[np.argmax(prediciton)]
                st.success(string)     
     
def cnn():
     st.write('''This project will help to detect and 
            classify whether the sugarcane leaf is 
            diseased or healthy. This model has achieved its goal by 
            effectively detecting and classifying sugarcane images into 
            healthy and diseased category based on the pattern of leaves. 
            ''')
     image1 = Image.open('../images/sugarcane1.jpg')
     st.image(image1, caption=' ',use_column_width=True)

     image1 = Image.open('../images/cnn_i.png')
     st.image(image1, caption=' ',use_column_width=True)

     instructions = """
            Upload image.
            The image you select or upload will be fed
            through model in real-time
            and the output will be displayed to the screen.
            """
     st.write(instructions)
        
     def load_model():
        loaded_model= tf.keras.models.load_model('../models/densenet_m.h5')
        return loaded_model
     model=load_model() 
     file = st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])
     st.write("Here is the file you uploaded ")
    
     file 

     def import_and_predict(image_data,model):
        size= (256,256)
        image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img =np.asarray(image)
        img_reshape=img[np.newaxis,...]
        prediction123 = model.predict(img_reshape)
        return prediction123

     if file is None:
        st.text("Please Upload file")

     if st.button('Predict',on_click=None):
            st.write('Results')
            image=Image.open(file)
            st.image(image,use_column_width=True)
            prediciton= import_and_predict(image,model)
            class_names=["Bacterial Blight","Healthy","Red Rot", "Red Rust"]
            if (class_names[np.argmax(prediciton)])=='Healthy':
                string="This plant is Healthy"
                st.success(string)
            else:
                string="This plant has :"+class_names[np.argmax(prediciton)]
                st.success(string) 
    
def alexnet():
     st.write('''This project will help to detect and 
            classify whether the sugarcane leaf is 
            diseased or healthy. This model has achieved its goal by 
            effectively detecting and classifying sugarcane images into 
            healthy and diseased category based on the pattern of leaves. 
            ''')
     col1, col2,  = st.columns(2)
    
     
     image1 = Image.open('../images/sugarcane1.jpg')
     st.image(image1, caption=' ',use_column_width=True)
        
    
     st.write("Accuracy of the Model")
     image1 = Image.open('../images/alexnet_i.png')
     st.image(image1, caption=' ',use_column_width=True)


     
    #  image1 = Image.open('../images/sugarcane1.jpg')
    #  st.image(image1, caption=' ',use_column_width=True)
    #  st.write("Accuracy of the Model")
    #  image1 = Image.open('../images/densenet_i.png')
    #  st.image(image1, caption=' ',use_column_width=True)

     instructions = """
            Upload image.
            The image you select or upload will be fed
            through model in real-time
            and the output will be displayed to the screen.
            """
     st.write(instructions)
        
     def load_model():
        loaded_model= tf.keras.models.load_model('../models/Alexnet_m.hdf5')
        return loaded_model
     model=load_model() 
     file = st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])
     st.write("Here is the file you uploaded ")
    
     file 

     def import_and_predict(image_data,model):
        size= (224,224)
        image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img =np.asarray(image)
        img_reshape=img[np.newaxis,...]
        prediction123 = model.predict(img_reshape)
        return prediction123

     if file is None:
        st.text("Please Upload file")

     if st.button('Predict',on_click=None):
            st.write('Results')
            image=Image.open(file)
            st.image(image,use_column_width=True)
            prediciton= import_and_predict(image,model)
            class_names=["Bacterial Blight","Healthy","Red Rot", "Red Rust"]
            if (class_names[np.argmax(prediciton)])=='Healthy':
                string="This plant is Healthy"
                st.success(string)
            else:
                string="This plant has :"+class_names[np.argmax(prediciton)]
                st.success(string)

def efficientnet():
     st.write('''This project will help to detect and 
            classify whether the sugarcane leaf is 
            diseased or healthy. This model has achieved its goal by 
            effectively detecting and classifying sugarcane images into 
            healthy and diseased category based on the pattern of leaves. 
            ''')
     
    
     
     image1 = Image.open('../images/sugarcane1.jpg')
     st.image(image1, caption=' ',use_column_width=True)
        
    
        # st.write("Accuracy of the Model")
     image1 = Image.open('../images/efficientnet_i.png')
     st.image(image1, caption=' ',use_column_width=True)


     
    #  image1 = Image.open('../images/sugarcane1.jpg')
    #  st.image(image1, caption=' ',use_column_width=True)
    #  st.write("Accuracy of the Model")
    #  image1 = Image.open('../images/densenet_i.png')
    #  st.image(image1, caption=' ',use_column_width=True)

     instructions = """
            Upload image.
            The image you select or upload will be fed
            through model in real-time
            and the output will be displayed to the screen.
            """
     st.write(instructions)
        
     def load_model():
      loaded_model= tf.keras.models.load_model('../models/efficientnet_m.h5')
      return loaded_model
     model=load_model() 
     file = st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])
     st.write("Here is the file you uploaded ")
    
     file 

     def import_and_predict(image_data,model):
        size= (224,224)
        image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img =np.asarray(image)
        img_reshape=img[np.newaxis,...]
        prediction123 = model.predict(img_reshape)
        return prediction123

     if file is None:
        st.text("Please Upload file")

     if st.button('Predict',on_click=None):
            st.write('Results')
            image=Image.open(file)
            st.image(image,use_column_width=True)
            prediciton= import_and_predict(image,model)
            class_names=["Bacterial Blight","Healthy","Red Rot", "Red Rust"]
            if (class_names[np.argmax(prediciton)])=='Healthy':
                string="This plant is Healthy"
                st.success(string)
            else:
                string="This plant has :"+class_names[np.argmax(prediciton)]
                st.success(string)

def inceptionnet():
     st.write('''This project will help to detect and 
            classify whether the sugarcane leaf is 
            diseased or healthy. This model has achieved its goal by 
            effectively detecting and classifying sugarcane images into 
            healthy and diseased category based on the pattern of leaves. 
            ''')
     col1, col2,  = st.columns(2)
    
     
     image1 = Image.open('../images/sugarcane1.jpg')
     st.image(image1, caption=' ',use_column_width=True)
        
    
        # st.write("Accuracy of the Model")
     image1 = Image.open('../images/inception_i.jpg')
     st.image(image1, caption=' ',use_column_width=True)


     
    #  image1 = Image.open('../images/sugarcane1.jpg')
    #  st.image(image1, caption=' ',use_column_width=True)
    #  st.write("Accuracy of the Model")
    #  image1 = Image.open('../images/densenet_i.png')
    #  st.image(image1, caption=' ',use_column_width=True)

     instructions = """
            Upload image.
            The image you select or upload will be fed
            through model in real-time
            and the output will be displayed to the screen.
            """
     st.write(instructions)
        
     def load_model():
        loaded_model= tf.keras.models.load_model('../models/inceptionv3_m.h5')
        return loaded_model
     model=load_model() 
     file = st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])
     st.write("Here is the file you uploaded ")
    
     file 

     def import_and_predict(image_data,model):
        size= (224,224)
        image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img =np.asarray(image)
        img_reshape=img[np.newaxis,...]
        prediction123 = model.predict(img_reshape)
        return prediction123

     if file is None:
        st.text("Please Upload file")

     if st.button('Predict',on_click=None):
            st.write('Results')
            image=Image.open(file)
            st.image(image,use_column_width=True)
            prediciton= import_and_predict(image,model)
            class_names=["Bacterial Blight","Healthy","Red Rot", "Red Rust"]
            if (class_names[np.argmax(prediciton)])=='Healthy':
                string="This plant is Healthy"
                st.success(string)
            else:
                string="This plant has :"+class_names[np.argmax(prediciton)]
                st.success(string)

def mobilenet():
     st.write('''This project will help to detect and 
            classify whether the sugarcane leaf is 
            diseased or healthy. This model has achieved its goal by 
            effectively detecting and classifying sugarcane images into 
            healthy and diseased category based on the pattern of leaves. 
            ''')
     
    
     
     image1 = Image.open('../images/sugarcane1.jpg')
     st.image(image1, caption=' ',use_column_width=True)
        
    
        # st.write("Accuracy of the Model")
     image1 = Image.open('../images/mobilenet_i.jpg')
     st.image(image1, caption=' ',use_column_width=True)


     
    #  image1 = Image.open('../images/sugarcane1.jpg')
    #  st.image(image1, caption=' ',use_column_width=True)
    #  st.write("Accuracy of the Model")
    #  image1 = Image.open('../images/densenet_i.png')
    #  st.image(image1, caption=' ',use_column_width=True)

     instructions = """
            Upload image.
            The image you select or upload will be fed
            through model in real-time
            and the output will be displayed to the screen.
            """
     st.write(instructions)
        
     def load_model():
        loaded_model= tf.keras.models.load_model('../models/mobilenet_m.h5')
        return loaded_model
     model=load_model() 
     file = st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])
     st.write("Here is the file you uploaded ")
    
     file 

     def import_and_predict(image_data,model):
        size= (224,224)
        image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img =np.asarray(image)
        img_reshape=img[np.newaxis,...]
        prediction123 = model.predict(img_reshape)
        return prediction123

     if file is None:
        st.text("Please Upload file")

     if st.button('Predict',on_click=None):
            st.write('Results')
            image=Image.open(file)
            st.image(image,use_column_width=True)
            prediciton= import_and_predict(image,model)
            class_names=["Bacterial Blight","Healthy","Red Rot", "Red Rust"]
            if (class_names[np.argmax(prediciton)])=='Healthy':
                string="This plant is Healthy"
                st.success(string)
            else:
                string="This plant has :"+class_names[np.argmax(prediciton)]
                st.success(string)

def resnet50():
     
     st.write('''This project will help to detect and 
            classify whether the sugarcane leaf is 
            diseased or healthy. This model has achieved its goal by 
            effectively detecting and classifying sugarcane images into 
            healthy and diseased category based on the pattern of leaves. 
            ''')
     
    
     
     image1 = Image.open('../images/sugarcane1.jpg')
     st.image(image1, caption=' ',use_column_width=True)
        
    
        # st.write("Accuracy of the Model")
     image1 = Image.open('../images/resnet50_i.png')
     st.image(image1, caption=' ',use_column_width=True)


     
    #  image1 = Image.open('../images/sugarcane1.jpg')
    #  st.image(image1, caption=' ',use_column_width=True)
    #  st.write("Accuracy of the Model")
    #  image1 = Image.open('../images/densenet_i.png')
    #  st.image(image1, caption=' ',use_column_width=True)

     instructions = """
            Upload image.
            The image you select or upload will be fed
            through model in real-time
            and the output will be displayed to the screen.
            """
     st.write(instructions)
        
     def load_model():
        loaded_model= tf.keras.models.load_model('../models/densenet_m.h5')
        return loaded_model
     model=load_model() 
     file = st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])
     st.write("Here is the file you uploaded ")
    
     file 

     def import_and_predict(image_data,model):
        size= (256,256)
        image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img =np.asarray(image)
        img_reshape=img[np.newaxis,...]
        prediction123 = model.predict(img_reshape)
        return prediction123

     if file is None:
        st.text("Please Upload file")

     if st.button('Predict',on_click=None):
            st.write('Results')
            image=Image.open(file)
            st.image(image,use_column_width=True)
            prediciton= import_and_predict(image,model)
            class_names=["Bacterial Blight","Healthy","Red Rot", "Red Rust"]
            if (class_names[np.argmax(prediciton)])=='Healthy':
                string="This plant is Healthy"
                st.success(string)
            else:
                string="This plant has :"+class_names[np.argmax(prediciton)]
                st.success(string)

def resnet101():
     st.write('''This project will help to detect and 
            classify whether the sugarcane leaf is 
            diseased or healthy. This model has achieved its goal by 
            effectively detecting and classifying sugarcane images into 
            healthy and diseased category based on the pattern of leaves. 
            ''')
     col1, col2,  = st.columns(2)
    
     
     image1 = Image.open('../images/sugarcane1.jpg')
     st.image(image1, caption=' ',use_column_width=True)
        
    
        # st.write("Accuracy of the Model")
     image1 = Image.open('../images/resnet101_i.png')
     st.image(image1, caption=' ',use_column_width=True)


     
    #  image1 = Image.open('../images/sugarcane1.jpg')
    #  st.image(image1, caption=' ',use_column_width=True)
    #  st.write("Accuracy of the Model")
    #  image1 = Image.open('../images/densenet_i.png')
    #  st.image(image1, caption=' ',use_column_width=True)

     instructions = """
            Upload image.
            The image you select or upload will be fed
            through model in real-time
            and the output will be displayed to the screen.
            """
     st.write(instructions)
        
     def load_model():
        loaded_model= tf.keras.models.load_model('../models/vgg19_m.h5')
        loaded_model.load_weights('../models/vgg19_m.h5')
        return loaded_model
     model=load_model() 
     file = st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])
     st.write("Here is the file you uploaded ")
    
     file 

     def import_and_predict(image_data,model):
        size= (224,224)
        image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img =np.asarray(image)
        img_reshape=img[np.newaxis,...]
        prediction123 = model.predict(img_reshape)
        return prediction123

     if file is None:
        st.text("Please Upload file")

     if st.button('Predict',on_click=None):
            st.write('Results')
            image=Image.open(file)
            st.image(image,use_column_width=True)
            prediciton= import_and_predict(image,model)
            class_names=["Bacterial Blight","Healthy","Red Rot", "Red Rust"]
            if (class_names[np.argmax(prediciton)])=='Healthy':
                string="This plant is Healthy"
                st.success(string)
            else:
                string="This plant has :"+class_names[np.argmax(prediciton)]
                st.success(string)

def vgg16():
     st.write('''This project will help to detect and 
            classify whether the sugarcane leaf is 
            diseased or healthy. This model has achieved its goal by 
            effectively detecting and classifying sugarcane images into 
            healthy and diseased category based on the pattern of leaves. 
            ''')
     col1, col2,  = st.columns(2)
    
     
     image1 = Image.open('../images/sugarcane1.jpg')
     st.image(image1, caption=' ',use_column_width=True)
        
    
        # st.write("Accuracy of the Model")
     image1 = Image.open('../images/vgg16_i.jpg')
     st.image(image1, caption=' ',use_column_width=True)


     
    #  image1 = Image.open('../images/sugarcane1.jpg')
    #  st.image(image1, caption=' ',use_column_width=True)
    #  st.write("Accuracy of the Model")
    #  image1 = Image.open('../images/densenet_i.png')
    #  st.image(image1, caption=' ',use_column_width=True)

     instructions = """
            Upload image.
            The image you select or upload will be fed
            through model in real-time
            and the output will be displayed to the screen.
            """
     st.write(instructions)
        
     def load_model():
        loaded_model= tf.keras.models.load_model('../models/VGG16_m.h5')
        return loaded_model
     model=load_model() 
     file = st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])
     st.write("Here is the file you uploaded ")
    
     file 

     def import_and_predict(image_data,model):
        size= (224,224)
        image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img =np.asarray(image)
        img_reshape=img[np.newaxis,...]
        prediction123 = model.predict(img_reshape)
        return prediction123

     if file is None:
        st.text("Please Upload file")

     if st.button('Predict',on_click=None):
            st.write('Results')
            image=Image.open(file)
            st.image(image,use_column_width=True)
            prediciton= import_and_predict(image,model)
            class_names=["Bacterial Blight","Healthy","Red Rot", "Red Rust"]
            if (class_names[np.argmax(prediciton)])=='Healthy':
                string="This plant is Healthy"
                st.success(string)
            else:
                string="This plant has :"+class_names[np.argmax(prediciton)]
                st.success(string)

def vgg19():
     st.write('''This project will help to detect and 
            classify whether the sugarcane leaf is 
            diseased or healthy. This model has achieved its goal by 
            effectively detecting and classifying sugarcane images into 
            healthy and diseased category based on the pattern of leaves. 
            ''')
     col1, col2,  = st.columns(2)
    
     
     image1 = Image.open('../images/sugarcane1.jpg')
     st.image(image1, caption=' ',use_column_width=True)
        
    
        # st.write("Accuracy of the Model")
     image1 = Image.open('../images/vgg19_i.png')
     st.image(image1, caption=' ',use_column_width=True)


     
    #  image1 = Image.open('../images/sugarcane1.jpg')
    #  st.image(image1, caption=' ',use_column_width=True)
    #  st.write("Accuracy of the Model")
    #  image1 = Image.open('../images/densenet_i.png')
    #  st.image(image1, caption=' ',use_column_width=True)

     instructions = """
            Upload image.
            The image you select or upload will be fed
            through model in real-time
            and the output will be displayed to the screen.
            """
     st.write(instructions)
        
     def load_model():
        loaded_model= tf.keras.models.load_model('../models/vgg19_m.h5')
        return loaded_model
     model=load_model() 
     file = st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])
     st.write("Here is the file you uploaded ")
    
     file 

     def import_and_predict(image_data,model):
        size= (224,224)
        image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img =np.asarray(image)
        img_reshape=img[np.newaxis,...]
        prediction123 = model.predict(img_reshape)
        return prediction123

     if file is None:
        st.text("Please Upload file")

     if st.button('Predict',on_click=None):
            st.write('Results')
            image=Image.open(file)
            st.image(image,use_column_width=True)
            prediciton= import_and_predict(image,model)
            class_names=["Bacterial Blight","Healthy","Red Rot", "Red Rust"]
            if (class_names[np.argmax(prediciton)])=='Healthy':
                string="This plant is Healthy"
                st.success(string)
            else:
                string="This plant has :"+class_names[np.argmax(prediciton)]
                st.success(string)

def comparison():
    st.caption("Here is a comparison of accuracies of all the models we have implemented ")
    image1 ='../images/cnn_i.png'
    image2 ='../images/densenet_i.png'
    image3 ='../images/efficientnet_i.png'
    image4 ='../images/resnet50_i.png'
    image5 ='../images/resnet101_i.png'
    image6 ='../images/inception_i.jpg'
    image7 ='../images/vgg16_i.jpg'
    image8 ='../images/vgg19_i.png'
    image9 ='../images/mobilenet_i.jpg'
    
    images123 = [image1,image2,image3,
                 image4,image5,image6,
                 image7,image8,image9]
    
    caption=['CNN','DENSENET','EFFICIENTNET','RESNET50','RESNET101','InceptionNET','VGG16','VGG19','MOBILENET']
    st.image(images123, width=225, caption=caption)

    # n_rows = 1 + len(images123) // int(3)
    # rows = [st.container() for _ in range(n_rows)]
    # cols_per_row = [r.columns(3) for r in rows]
    # cols = [column for row in cols_per_row for column in row]

    # for image_index, cat_image in enumerate(images123):
    #     cols[image_index].image(cat_image)  
    
    



with st.sidebar:
    selected = option_menu('Sugarcane Disease Predicition using Deep Learning',
                          
                          ['DenseNet',
                           'CNN',
                           'AlexNet',
                           'EfficientNet',
                           'InceptionNet',
                           'MobileNet',
                           'ResNet50',
                           'ResNet101',
                           'VGG16',
                           'VGG19',
                           'Comparison'],
                           
                          default_index=0)


if (selected == 'DenseNet'):
    
    # page title
    st.title('Sugarcane Disease Prediction Using AI (DESNET)')
    dense_net()

if (selected == 'CNN'):
    
    st.title('Sugarcane Disease Prediction Using AI (CNN)')
    cnn()

if (selected == 'AlexNet'):
    
    st.title('Sugarcane Disease Prediction Using AI (Alexnet)')
    alexnet()

if (selected == 'EfficientNet'):
    
    st.title('Sugarcane Disease Prediction Using AI (EfficientNet)')
    efficientnet()

if (selected == 'InceptionNet'):
    
    st.title('Sugarcane Disease Prediction Using AI (InceptionNet)')
    inceptionnet()

if (selected == 'MobileNet'):
    
    st.title('Sugarcane Disease Prediction Using AI (MobileNet)')
    mobilenet()

if (selected == 'ResNet50'):
    
    st.title('Sugarcane Disease Prediction Using AI (ResNet50)')
    resnet50()  

if (selected == 'ResNet101'):
    
    st.title('Sugarcane Disease Prediction Using AI (ResNet101)')
    resnet101()

if (selected == 'VGG16'):
    
    st.title('Sugarcane Disease Prediction Using AI (VGG16)')
    vgg16()
if (selected == 'VGG19'):
    
    st.title('Sugarcane Disease Prediction Using AI (VGG19)')
    vgg19()

if (selected == 'Comparison'):
    
    st.title('Comparing Accuracies of Different models')
    comparison()



