#*
#    SLAM.py: the implementation of SLAM
#    created and maintained by Ty Nguyen
#    tynguyen@seas.upenn.edu
#    Feb 2020
#*
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:
    sys.path.remove(ros_path)
from MapUtils.bresenham2D import *
from probs_utils import *
import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import os, sys, time
import p3_util as ut
from read_data import LIDAR, JOINTS
import probs_utils as prob
import math
import cv2
import transformations as tf
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import logging
if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
 

class SLAM(object):
    def __init__(self):
        self._characterize_sensor_specs()
    
    def _read_data(self, src_dir, dataset=0, split_name='train'):
        self.dataset_= str(dataset)
        if split_name.lower() not in src_dir:
            src_dir  = src_dir + '/' + split_name
        print('\n------Reading Lidar and Joints (IMU)------')
        self.lidar_  = LIDAR(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_lidar'+ self.dataset_)
        print ('\n------Reading Joints Data------')
        self.joints_ = JOINTS(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_joint'+ self.dataset_)

        res = self.lidar_.data_[0]['res'][0][0]
        self.lidar_angles = np.arange(-135*np.pi/180,135.003*np.pi/180,res).reshape(1,-1)

        self.num_data_ = len(self.lidar_.data_)
        # Position of odometry
        self.odo_indices_ = np.empty((2,self.num_data_),dtype=np.int64)
        self.temp = 0

    def _characterize_sensor_specs(self, p_thresh=None):
        # High of the lidar from the ground (meters)
        self.h_lidar_ = 0.93 + 0.33 + 0.15
        # Accuracy of the lidar
        self.p_true_ = 9
        self.p_false_ = 1.0/9
        
        #TODO: set a threshold value of probability to consider a map's cell occupied  
        self.p_thresh_ = 0.6 if p_thresh is None else p_thresh # > p_thresh => occupied and vice versa
        # Compute the corresponding threshold value of logodd
        self.logodd_thresh_ = prob.log_thresh_from_pdf_thresh(self.p_thresh_)

    def getJointTime(self,time):
        t = self.temp
        # print("getinggg",self.joints_.data_['ts'][0][1]-self.joints_.data_['ts'][0][0])
        # print(len(self.lidar_.data_),len(self.joints_.data_['ts'][0]))
        while ((self.lidar_.data_[time]['t'][0]-self.lidar_.data_[0]['t'][0]>(self.joints_.data_['ts'][0][t]-self.joints_.data_['ts'][0][0])) and ( t+1 < len(self.joints_.data_['ts'][0]))):
            # print(self.lidar_.data_[time]['t'][0],self.joints_.data_['ts'][0][t])
            t = t+1
        self.temp = t
        return t-1

    def _init_particles(self, num_p=0, mov_cov=None, particles=None, weights=None, percent_eff_p_thresh=None):
        # Particles representation
        self.num_p_ = num_p
        #self.percent_eff_p_thresh_ = percent_eff_p_thresh
        self.particles_ = np.zeros((3,self.num_p_),dtype=np.float64) if particles is None else particles
        
        # Weights for particles
        self.weights_ = 1.0/self.num_p_*np.ones(self.num_p_) if weights is None else weights

        # Position of the best particle after update on the map
        self.best_p_indices_ = np.empty((2,self.num_data_),dtype=np.int64)
        # Best particles
        self.best_p_ = np.empty((3,self.num_data_))
        # Corresponding time stamps of best particles
        self.time_ =  np.empty(self.num_data_)
       
        # Covariance matrix of the movement model
        tiny_mov_cov   = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0 , 1e-8]])
        self.mov_cov_  = mov_cov if mov_cov is not None else tiny_mov_cov
        # To generate random noise: x, y, z = np.random.multivariate_normal(np.zeros(3), mov_cov, 1).T
        # this return [x], [y], [z]

        # Threshold for resampling the particles
        self.percent_eff_p_thresh_ = percent_eff_p_thresh

    def _init_map(self, map_resolution=0.05):
        '''*Input: resolution of the map - distance between two grid cells (meters)'''
        # Map representation
        MAP= {}
        MAP['res']   = map_resolution #meters
        MAP['xmin']  = -20  #meters
        MAP['ymin']  = -20
        MAP['xmax']  =  20
        MAP['ymax']  =  20
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        self.MAP_ = MAP
        self.bresenDict = {}

        self.log_odds_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        self.occu_ = np.ones((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.uint64)

    # def getCorrMap(self,map,occX,occY):
    #     r = np.zeros(map.shape)
    #     r[occX,occY] = 1
    #     return np.sum(np.logical_and(r,map))

    def getMapCoord(self,x,y):
        x_map = np.uint16((x-self.MAP_['xmin'])/self.MAP_['res'])
        y_map = np.uint16((y-self.MAP_['ymin'])/self.MAP_['res'])
        return x_map,y_map

    def buildMAP(self,t,x,y,theta):
        # MAP = self.MAP_
        tj0 = self.getJointTime(t)
        scan = self.lidar_.data_[t]['scan'][0]
        neck_angle,head_angle = self.joints_.data_['head_angles'][:,tj0][0],self.joints_.data_['head_angles'][:,tj0][1]
        body_2_lidar_rot = tf.homo_transform(np.dot(tf.rot_z_axis(neck_angle), tf.rot_y_axis(head_angle)),[0,0,0])
        ground_2_body = tf.homo_transform(tf.rot_z_axis(theta),[x,y,self.h_lidar_])
        #Got the transforms
        homoscan = np.empty((4,(self.lidar_angles).shape[1]),dtype=np.float)
        homoscan[0,:] = np.cos(self.lidar_angles)*scan
        homoscan[1,:] = np.sin(self.lidar_angles)*scan
        homoscan[2,:] = np.zeros((1,self.lidar_angles.shape[1]))
        homoscan[3,:] = np.ones(self.lidar_angles.shape[1])
        # xscan = np.cos(self.lidar_angles)*scan
        # yscan = np.sin(self.lidar_angles)*scan
        # zscan = np.zeros((1,xscan.shape[1]))
        # Onescan = np.ones(zscan.shape)
        ground_2_lidar = np.dot(ground_2_body,body_2_lidar_rot)
        # homoscan = np.vstack((xscan,yscan,zscan,Onescan))
        trans_scan = (np.dot(ground_2_lidar,homoscan)).astype(np.float16)
        ground_zz = trans_scan[2]<0.1
        x_new = ((trans_scan[0]-self.MAP_['xmin'])//self.MAP_['res']).astype(np.uint16)
        y_new = ((trans_scan[1]-self.MAP_['ymin'])//self.MAP_['res']).astype(np.uint16)
        x_start = ((x-self.MAP_['xmin'])//self.MAP_['res']).astype(np.uint16)
        y_start = ((y-self.MAP_['xmin'])//self.MAP_['res']).astype(np.uint16)
        if not(self.psx==x_start and self.psy==y_start):
            self.bresenDict = {}
        
        self.psx = x_start
        self.psy = y_start
        for x_n,y_n,ground in zip(x_new,y_new,ground_zz):
            if abs(x_n)<self.MAP_['sizex'] and abs(y_n)<self.MAP_['sizey']:
                if (self.bresenDict.get((x_n,y_n)) is None):
                    ray_cells = bresenham2D(x_start,y_start,x_n,y_n)
                    self.bresenDict[(x_n,y_n)] = ray_cells
                else:
                    ray_cells = self.bresenDict[(x_n,y_n)]
                    self.dict_use_count = self.dict_use_count +1 
                    # print("using dict")
                x = np.asarray(ray_cells[0],dtype=int)
                y = np.asarray(ray_cells[1],dtype=int)
                x_end = x[-1]
                y_end = y[-1]
                x = x[:-1]
                y = y[:-1]
                self.log_odds_[x,y] = self.log_odds_[x,y] + np.log(self.p_false_)
                self.MAP_['map'][x,y] = self.log_odds_[x,y]>np.log(1.5)
                self.log_odds_[x_end,y_end] = self.log_odds_[x_end,y_end]+(1-ground)*np.log(self.p_true_)
            
                # for temp in range(ray_cells.shape[1]-1):


                #     x = ray_cells[:,temp][1]
                #     y = ray_cells[:,temp][0]
                #     # print(x,y)
		    

                #     self.log_odds_[int(y),int(x)] = self.log_odds_[int(y),int(x)] + np.log(self.p_false_)
                #     if self.log_odds_[int(y),int(x)]>np.log(1.5):
                #         self.MAP_['map'][int(y),int(x)] = 1
                #     else:
                #         self.MAP_['map'][int(y),int(x)] = 0
                # if ground==0:
                #     self.log_odds_[int(ray_cells[:,-1][0]),int(ray_cells[:,-1][1])] = self.log_odds_[int(ray_cells[:,-1][0]), \
                #                                                             int(ray_cells[:,-1][1])] + np.log(self.p_true_)
                #     if self.log_odds_[int(ray_cells[:,-1][0]),int(ray_cells[:,-1][1])]>np.log(1.5):
                #         self.MAP_['map'][int(ray_cells[:,-1][0]),int(ray_cells[:,-1][1])] = 1
                #     else:
                #         self.MAP_['map'][int(ray_cells[:,-1][0]),int(ray_cells[:,-1][1])] = 0

	

    def _build_first_map(self,t0=0,use_lidar_yaw=True):
        """Build the first map using first lidar"""
        self.t0 = t0
        # Extract a ray from lidar data
        # MAP = self.MAP_

        print('\n--------Doing build the first map--------')


        #TODO: student's input from here

        # tj0 = self.getJointTime(t0)
        # scan = self.lidar_.data_[t0]['scan'][0]
        # neck_angle,head_angle = self.joints_.data_['head_angles'][:,tj0][0],self.joints_.data_['head_angles'][:,tj0][1]
        # print("printtt",neck_angle,head_angle)
        # pos_init = self.lidar_.data_[t0]['pose'][0]
        # body_2_lidar_rot = tf.homo_transform(np.dot(tf.rot_z_axis(neck_angle), tf.rot_y_axis(head_angle)),[0,0,0])
        # ground_2_body = tf.homo_transform(tf.rot_z_axis(pos_init[2]),[pos_init[0],pos_init[1],self.h_lidar_])
        #Got the transforms
        # xscan = np.cos(self.lidar_angles)*scan
        # yscan = np.sin(self.lidar_angles)*scan
        # zscan = np.zeros((1,xscan.shape[1]))
        # Onescan = np.ones(zscan.shape)
        # ground_2_lidar = np.dot(ground_2_body,body_2_lidar_rot)
        # homoscan = np.vstack((xscan,yscan,zscan,Onescan))
        # trans_scan = np.dot(ground_2_lidar,homoscan)
        # ground_zz = trans_scan[2]<0.1
        # x_new = np.array((trans_scan[0]-MAP['xmin'])/MAP['res'],dtype=int)
        # y_new = np.array((trans_scan[1]-MAP['ymin'])/MAP['res'],dtype=int)
        # x_start = (pos_init[0]-MAP['xmin'])/MAP['res']
        # y_start = (pos_init[1]-MAP['xmin'])/MAP['res']
        best_p_i = np.argmax(self.weights_)
        best_p = self.particles_[:,best_p_i]
        x = best_p[0]
        y = best_p[1]
        self.psx = x
        self.psy = y
        theta = best_p[2]
        self.buildMAP(t0,x,y,theta)
        self.best_p_[:,t0] = best_p
        self.best_p_indices_[:,t0] = self.getMapCoord(x,y)

        # print(max(x_new-x_start),max(y_new-y_start))
        # plt.plot(self.MAP_)
        # for x_n,y_n,ground in zip(x_new,y_new,ground_zz):
        #     # print("printinggg",x_start,y_start)
        #     ray_cells = bresenham2D(x_start,y_start,x_n,y_n)
        #     for temp in range(ray_cells.shape[1]-1):
        #         x = ray_cells[:,temp][1]
        #         y = ray_cells[:,temp][0]
        #         # print(x,y)
        #         self.log_odds_[int(y),int(x)] = self.log_odds_[int(y),int(x)] + np.log(self.p_false_)
        #         if self.log_odds_[int(y),int(x)]>np.log(1.5):
        #             self.MAP_['map'][int(y),int(x)] = 1
        #         else:
        #             self.MAP_['map'][int(y),int(x)] = 0
        #     if ground==0:

        #         self.log_odds_[int(ray_cells[:,-1][0]),int(ray_cells[:,-1][1])] = self.log_odds_[int(ray_cells[:,-1][0]), \
        #                                                                 int(ray_cells[:,-1][1])] + np.log(self.p_true_)
        #         if self.log_odds_[int(ray_cells[:,-1][0]),int(ray_cells[:,-1][1])]>np.log(1.5):
        #             self.MAP_['map'][int(ray_cells[:,-1][0]),int(ray_cells[:,-1][1])] = 1
        #         else:
        #             self.MAP_['map'][int(ray_cells[:,-1][0]),int(ray_cells[:,-1][1])] = 0




    def _predict(self,t,use_lidar_yaw=True):
        logging.debug('\n-------- Doing prediction at t = {0}------'.format(t))


        #TODO: student's input from here 
        # for i in range(self.num_p_):
        #     w_i = np.random.multivariate_normal(np.zeros(3),self.mov_cov_)
        #     self.particles_[:,i] = tf.twoDSmartPlus(self.particles_[:,i],w_i)
        for i in range(self.num_p_):
            #odo_diff = tf.twoDSmartMinus(self.lidar_.data_[t]['pose'][0],self.lidar_.data_[t-1]['pose'][0])
            #delta = tf.twoDSmartPlus(np.random.multivariate_normal(np.zeros(3),self.mov_cov_),odo_diff)
            delta = np.random.multivariate_normal(np.zeros(3),self.mov_cov_)
            self.particles_[:,i] = tf.twoDSmartPlus(self.particles_[:,i],delta)
        #End student's input 




    def _update(self,t,t0=0,fig='on'):
        """Update function where we update the """

        # MAP = self.MAP_
        if t == t0:
            self._build_first_map(t0,use_lidar_yaw=True)
            return

        #TODO: student's input from here
        tj0 = self.getJointTime(t)
        neck_angle,head_angle = self.joints_.data_['head_angles'][:,tj0][0],self.joints_.data_['head_angles'][:,tj0][1]
        body_2_lidar_rot = tf.homo_transform(np.dot(tf.rot_z_axis(neck_angle), tf.rot_y_axis(head_angle)),[0,0,0])
        scan = self.lidar_.data_[t]['scan'][0]
        xscan = np.cos(self.lidar_angles)*scan
        yscan = np.sin(self.lidar_angles)*scan
        zscan = np.zeros((1,xscan.shape[1]))
        Onescan = np.ones(zscan.shape)
        corr = []
        for i in range(self.num_p_):
            pose_i = self.particles_[:,i]
            ground_2_body = tf.homo_transform(tf.rot_z_axis(pose_i[2]),[pose_i[0],pose_i[1],self.h_lidar_])
            #Got the transforms
            ground_2_lidar = np.dot(ground_2_body,body_2_lidar_rot)
            homoscan = np.vstack((xscan,yscan,zscan,Onescan))
            trans_scan = np.dot(ground_2_lidar,homoscan)
            # vp = np.vstack((xscan,yscan))
            # print("zz",min(trans_scan[1]))
            trans_scan = trans_scan[:,abs(trans_scan[0])<20]
            trans_scan = trans_scan[:,abs(trans_scan[1])<20]
            x_ind,y_ind = self.getMapCoord(trans_scan[0],trans_scan[1])

            occupied_indices = np.vstack((x_ind,y_ind))
            corr.append(mapCorrelation(self.MAP_['map'],occupied_indices))
        # print(self.num_p_)
            # corr.append(self.getCorrMap(self.MAP_['map'],x_ind,y_ind))
	    # space = 0.02
            # xs = np.arange(pose_i[0]-space,pose_i[0]+space,space/4)
            # ys = np.arange(pose_i[1]-space,pose_i[1]+space,space/4)
            #self.weights_[i] = self.weights_[i]*np.sum(mapCorrelation(self.MAP_['map'], np.array([self.MAP_['xmin'],self.MAP_['xmax']]), \
                                         # np.array([self.MAP_['ymin'],self.MAP_['ymax']]), vp, xs, ys))
            #self.weights_[i] = self.weights_[i]*mapCorrelation(self.MAP_['map'],occupied_indices)
        
        #self.weights_ = self.weights_/np.sum(self.weights_)
        corr = np.asarray(corr)
        self.weights_ = prob.update_weights(self.weights_,corr)
        best_p_i = np.argmax(self.weights_)
        best_p = self.particles_[:,best_p_i]
        x = best_p[0]
        y = best_p[1]
        theta = best_p[2]
        self.buildMAP(t,x,y,theta)
        self.best_p_[:,t] = best_p
        self.best_p_indices_[:,t] = self.getMapCoord(x,y)

        #End student's input 

        # self.MAP_ = MAP
        return self.MAP_
