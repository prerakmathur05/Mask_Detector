from keras.models import load_model
import cv2
import numpy as np
import time

model = load_model('model-030.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
k=0
while(True):

		ret,img=source.read()
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		faces=face_clsfr.detectMultiScale(gray,1.3,5)  

		for x,y,w,h in faces:

			face_img=img[y:y+h,x:x+w]
			resized=cv2.resize(face_img,(100,100))
			print(resized.shape)

			normalized= resized/255.0

			reshaped=np.reshape(normalized,(1,100,100))
			result=model.predict(reshaped)
			print(f"here its is {result}")
			print(type(result))



			label=np.argmax(result)
			print(label)

			#labell=np.round(result)
			#label=int(labell[0])
			#print(label)


			cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
			cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
			cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
			
			if (int(label)==1 and int(time.time())%5==0):

				imgname=f"defaulter {k} image.png"
				cv2.imwrite(imgname,img)
				print(f"The defaulter number {k} has been found and his/her image is saved ")
				k+=1

		cv2.imshow('Welcome to Prerak\'s Live cam',img)
		key=cv2.waitKey(1)
		if(key==27):
			break
    
cv2.destroyAllWindows()
source.release()
