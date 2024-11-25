#This wil be an object oriented version
#Of the virtual3d game
import cv2
import numpy

print ('Starting OO Virtual3D')


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
		    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 0), 5)
		    return (bx+bw/x),(by+bh/2)
