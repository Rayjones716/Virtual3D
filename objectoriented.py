#This wil be an object oriented version
#Of the virtual3d game
import cv2
import numpy

print ('Starting OO Virtual3D')


class Tunnel:
	pass


class Facefinder:
	def __init__(self):
		print('Face Finder Initialize')
		faces = self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	def find_face(self,frame):
		"""Returns face center (x,y), draws rect on frame"""

		#Convert to Greyscale:
		gray = cv2.cvtColor(frame,cv2.BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray)	

		#Draw Rectangle

		if faces is none:
			return none
		bx=by=bw=bh=0	

		for (x, y, w, h) in faces:
		    if w > bw:
		        bx, by, bw, bh = b, w, y, h
		    cv2.rectangle(img, (x, y), (bx + bw, by + bh), (0, 255, 255), 3)
		    return (bx+bw/2),(by+bh/2)

#--------------------

ff = Facefinder()
print('Virtual3D Complete')



ff = FaceFinder()
cap = cv2.VideoCapture(cv2.CAP_ANY)
if not cap.isOpened():
	print('couldnt open cam')
	exit()




while True:
  retval, frame = cap.read()
  if retval == False:
    print('camera error!')

  ff.find_face(frame)
  cv2.imshow('q to quit', frame)

  if cv2.waitKey(30) == ord('q'):
    break



pause = input('press enter to end')

cap.release()

cv2.destroyAllWindows()


print('virtual3d complete')#This wil be an object oriented version
#Of the virtual3d game
import cv2
import numpy

print ('Starting OO Virtual3D')


class Tunnel:
	pass


class Facefinder:
	def __init__(self):
		print('Face Finder Initialize')
		faces = self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	def find_face(self,frame):
		"""Returns face center (x,y), draws rect on frame"""

		#Convert to Greyscale:
		gray = cv2.cvtColor(frame,cv2.BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray)	

		#Draw Rectangle

		if faces is none:
			return none
		bx=by=bw=bh=0	

		for (x, y, w, h) in faces:
		    if w > bw:
		        bx, by, bw, bh = b, w, y, h
		    cv2.rectangle(img, (x, y), (bx + bw, by + bh), (0, 255, 255), 3)
		    return (bx+bw/2),(by+bh/2)

#--------------------

ff = Facefinder()
print('Virtual3D Complete')



ff = FaceFinder()
cap = cv2.VideoCapture(cv2.CAP_ANY)
if not cap.isOpened():
	print('couldnt open cam')
	exit()




while True:
  retval, frame = cap.read()
  if retval == False:
    print('camera error!')

  ff.find_face(frame)
  cv2.imshow('q to quit', frame)

  if cv2.waitKey(30) == ord('q'):
    break



pause = input('press enter to end')

cap.release()

cv2.destroyAllWindows()


print('virtual3d complete')
