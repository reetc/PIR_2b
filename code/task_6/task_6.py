
import numpy as np
import cv2
import glob
import argparse
# from google.colab.patches import cv2_imshow
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

import sys

import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 1)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,0,255), 1)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,255), 1)
    return img






import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(objp,xloc,yloc,zloc):
  print('plotting in 3D')
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(objp[0],objp[1])
  ax.plot(objp[1],objp[2])
  ax.plot(objp[2],objp[3])
  ax.plot(objp[3],objp[0])
  ax.scatter3D(xloc,yloc,zloc)
  ax.view_init(azim=60)
  plt.show()
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.show()
#     plt.pause(0.5)

def pose_estimate(M1,dist1,M2,dist2,camera):
    import numpy as np
    # M1=[[423.27381306,0,341.34626532],[0,421.27401756,269.28542111],[0,0,1]]
    # dist1=[-0.43394157423038077,0.26707717557547866,-0.00031144347020293427,0.000563893810148836,-0.10970452266148858]
    #
    # M2=[[420.91160482,0,352.16135589],[0,418.72245958,264.50726699],[0,0,1]]
    # dist2=[-0.4145817681176909,0.19961273246897668,-0.0001483209114165653,-0.0013686760437966467,-0.05113584625015141]
    #
    objp = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])


    M1 = np.asarray(M1)
    dist1 = np.asarray(dist1)
    M2 = np.asarray(M2)
    dist2 = np.asarray(dist2)


    if camera == 'left':
      camera_matrix=np.array(M1,dtype = "double")
      dist = np.asarray(dist1)
      images=glob.glob("../../images/task_6/left*.png")
    else:
      camera_matrix=np.array(M2,dtype = "double")
      dist = np.asarray(dist2)
      images=glob.glob("../../images/task_6/right*.png")# print(type(M1))


    import numpy as np
    import cv2
    # from google.colab.patches import cv2_imshow

    #dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    #dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    xloc = []
    yloc = []
    zloc = []
    i=0



    while(i<len(images)):
        try:
        # Capture frame-by-frame
        # ret, frame = cap.read()
          frame = cv2.imread(images[i])
          # Our operations on the frame come here
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

          res = cv2.aruco.detectMarkers(gray,dictionary)
      #   print(res[0],res[1],len(res[2]))

          if len(res[0]) > 0:
              cv2.aruco.drawDetectedMarkers(frame,res[0],res[1],borderColor=(0, 255, 0))
          # Display the resulting frame
          imgWithAruco=frame
          x1 = (res[0][0][0][0][0], res[0][0][0][0][1])
          x2 = (res[0][0][0][1][0], res[0][0][0][1][1])
          x3 = (res[0][0][0][2][0], res[0][0][0][2][1])
          x4 = (res[0][0][0][3][0], res[0][0][0][3][1])
          imgp = np.asarray([x1,x2,x3,x4])
          # print(x1,x2)

          # print(objp, imgp)
          cv2.line(gray, x1, x2, (255, 0, 0),1)
          cv2.line(gray, x2, x3, (255, 0, 0), 1)
          cv2.line(gray, x3, x4, (255, 0, 0), 1)
          cv2.line(gray, x4, x1, (255, 0, 0), 1)
          _,rvecs, tvecs = cv2.solvePnP(objp, imgp, camera_matrix, dist)
          # print(tvecs)


          imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist)

          ZYX,_ = cv2.Rodrigues(rvecs)
          # orient = -orient
          totalrotmax=np.array([[ZYX[0,0],ZYX[0,1],ZYX[0,2],tvecs[0]],[ZYX[1,0],ZYX[1,1],ZYX[1,2],tvecs[1]],[ZYX[2,0],ZYX[2,1],ZYX[2,2],tvecs[2]],[0.,0.,0.,1.]],dtype='float')
          WtoC=np.mat(totalrotmax)
          inverserotmax=np.linalg.inv(totalrotmax)
          f=inverserotmax
          # print(inverserotmax)
          xloc.append(inverserotmax[0][3])
          yloc.append(inverserotmax[1][3])
          zloc.append(inverserotmax[2][3])


          # print(tvecs)
          filename='../../output/task_6/img_'+camera+str(i)+'.jpg'
          # cv2.imshow('frame',frame)
          cv2.imwrite(filename, frame)
          print('image {} saved in output folder'.format(i) )
          i+=1
        except:
          print('skipping image {}'.format(i))
          i+=1
          continue




    # print(xloc)

    plot(objp,xloc,yloc,zloc)

if __name__ == "__main__":
    import pathlib
    pathlib.Path('../../output/task_6').mkdir(parents=True, exist_ok=True)
    M1=[[423.27381306,0,341.34626532],[0,421.27401756,269.28542111],[0,0,1]]
    dist1=[-0.43394157423038077,0.26707717557547866,-0.00031144347020293427,0.000563893810148836,-0.10970452266148858]

    M2=[[420.91160482,0,352.16135589],[0,418.72245958,264.50726699],[0,0,1]]
    dist2=[-0.4145817681176909,0.19961273246897668,-0.0001483209114165653,-0.0013686760437966467,-0.05113584625015141]
    camera = sys.argv[1]
    objp = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    pose_estimate(M1,dist1,M2,dist2,camera)
