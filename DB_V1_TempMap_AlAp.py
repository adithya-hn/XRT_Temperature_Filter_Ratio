#=========
#Improving pairing logic done
#output values kept to 6 decimals-accurate up to 30 sec DOB can be used as Unique ID-use DOB code to ge back the date
# 30/5/22
# >Almesh Tipoly
# >Improved adopted from 6 almesh code (CH noise, XBP 2k img, xbp Number)
# >BG issue fixed
# >5% radius reduction is incorporated in terms of pixels
# 1/6/22
# >circle fit was much needed, for both radius and center estimation.
# >Used al mesh center pixel for Ti poly center correction
# >File format?
# 3/6/22
# > XBP counting-Overlapping issue is solved
# >Databse file
# >Image name base on two imgs name
# > Compatible for Al-Ap pairs, interpolation is icreased to 400 pts
#
#22/2/23 -corrections CH and interpolation
# > CH threshold changed back tp 0.3 md
# > There is no round off of ratios
# > Scipy interpolation is implemented
# > Anything less or more than response ratio are considered as 0

#18/2/23 -error values are added
# > Error calculation from xrtpy are used
# > Error sum values are added to export file
# > Error threshold set to 50%
# > Resp function reverted to IDL
# > Area corrected for thresholding 

#09/10/23
# >

#=========


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.io import fits
from astropy import visualization as av
from astropy.time import Time
from astropy import units as u
from astropy.constants import c, h
import scipy.misc
import math as mt
#from jdcal import gcal2jd, jd2gcal
import datetime
import os
import sys
import timeit
import scipy as sp
import cv2
import pathlib
import imageio
import numpy.ma as ms
from scipy.ndimage import label, generate_binary_structure, find_objects, measurements, map_coordinates,shift
import scipy.stats as si
import zipfile
from scipy import interpolate

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

#import xrt_teem_mod
import sunpy.map
import xrtpy
#import xrt_teem_mod
from xrtpy.response.temperature_response import TemperatureResponseFundamental

#Program timer
startTime = timeit.default_timer()
totelIm = 0
filetime=0
prvTime=startTime
#----

pair_imgs=np.loadtxt('V1_Al_Ap_Pairs_D08_10_23.dat',dtype='str')                  #al mesh and and Ti poly image pairs
r_data = (np.loadtxt("V1_Resp_data_AP_Al.txt")).transpose() #Response function from IDL
Segfold='/media/adithyahn/New Volume/DB_Irradiance/DB_V1/RGNmaps_DB_V1'

temp = r_data[0]
Tiresp = r_data[1]
Alresp = r_data[2]

#Folder creation
pathlib.Path("Al_Ap_TempMaps").mkdir(parents=True, exist_ok=True)
pathlib.Path("ERmaps").mkdir(parents=True, exist_ok=True)
#pathlib.Path("imgs").mkdir(parents=True, exist_ok=True)
pathlib.Path("TempFits").mkdir(parents=True, exist_ok=True)
pathlib.Path("ApPNmaps").mkdir(parents=True, exist_ok=True) #TiPNmaps
pathlib.Path("AlPNmaps").mkdir(parents=True, exist_ok=True) 
#----

#Empty array creation
CHarray = []
BParray = []
ARarray = []
BGarray = []
FDarray = []
LBarray = []

CHarea = []
BParea = []
ARarea = []
BGarea = []

CHA = []
BPA = []
ARA = []
BGA = []
FDA = []

CHa = []
BPa = []
ARa = []
BGa = []
FDa = []
LBa = []

CHi = []
BPi = []
ARi = []
BGi = []
Fdi = []

n_AR = []
n_BP = []
n_CH = []
l_DOB = []
l_dname=[]
conv_rat=[]

IR_data = []
bpIRsum=[]
chIRsum=[]
bgIRsum=[]
ArIRsum=[]

bpERsum=[]
chERsum=[]
arERsum=[]
bgERsum=[]
fdERsum=[]

shapeArry=[]
size_n_match=[]

Length=(pair_imgs.shape)[0]

def deriv(x, y):
    x0 = x[:-2]
    x1 = x[1:-1]
    x2 = x[2:]
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]
    x01 = x0 - x1
    x02 = x0 - x2
    x12 = x1 - x2
    dydx1 = (y0 * x12 / (x01 * x02) + y1 * (1.0 / x12 - 1.0 / x01) - y2 * x01 / (x02 * x12))
    dydx0 = (y0[0] * (x01[0] + x02[0]) / (x01[0] * x02[0])- y1[0] * x02[0] / (x01[0] * x12[0])+ y2[0] * x01[0] / (x02[0] * x12[0]))
    dydxN = (-y0[-1] * x12[-1] / (x01[-1] * x02[-1])+ y1[-1] * x02[-1] / (x01[-1] * x12[-1])- y2[-1] * (x02[-1] + x12[-1]) / (x02[-1] * x12[-1]))
    return np.append(np.insert(dydx1, 0, dydx0), dydxN)

for l in range(Length):
    try:
        #print(pair_imgs[l][1])
        img1 = fits.open(pair_imgs[l][1])
        Fname1 = (os.path.splitext((pair_imgs[l][1].split(os.sep))[-1]))[0]
        #/media/adithyahn/New Volume/DB_Irradiance/DB_V1/RGNmaps_DB_V1/comp_XRT20080105_105413.3_seg.fits.zip
        scidata1 = img1[0].data
        DOB1 = img1[0].header['DATE_OBS']
        x1cen = img1[0].header['XCEN']
        y1cen = img1[0].header['YCEN']
        x1scale = img1[0].header['XSCALE']
        y1scale = img1[0].header['YSCALE']
        #scidata1=shift(scidata1,((x1cen/x1scale),(y1cen/y1scale)),cval=1)
        scidata = scidata1
        size1 = scidata1.shape
        seg_zip_file=Segfold+'/'+Fname1+'_seg.fits.zip'
        seg_file=Fname1+'_seg.fits' 
        #print(seg_file)
        with zipfile.ZipFile(seg_zip_file, 'r') as zip_file:
          zip_file.extract(seg_file)
        sg = fits.open(seg_file)  
        sgData=sg[0].data
        Rad=int(sg[0].header['RSUN_OBS']/x1scale)
        segx1cen = sg[0].header['XCEN']/x1scale
        segy1cen = sg[0].header['YCEN']/x1scale
        #print(Rad,(1024-segx1cen),(1024-segy1cen))

        if size1[0] == 1024:
            #kcanny = canny(scidata1, sigma=2, low_threshold=8, high_threshold=100)
            #hough_radii = np.arange(466, 478, 2)
            #hough_res = hough_circle(kcanny, hough_radii)
            #accums, cx, cy, rad = hough_circle_peaks(hough_res, hough_radii,total_num_peaks=1)
            #Rad = int(rad[0])
            Reduce=int(Rad*0.05) #5% of radius
            Radlb=int(Rad*1.05)
            R = int(Rad - Reduce)
            center = (int(512-segx1cen),int(512-segy1cen))
            #scidata1 = shift(scidata1, (-(512-center[0]), -(512-center[1])), cval=1)
            #center=(512,512)
            xbpTpix = 10
            chTpix = 10
            arTpix = 1000
            
            #print(Rad,center)

        else:
            #kcanny = canny(scidata1, sigma=2, low_threshold=5, high_threshold=30)
            #hough_radii = np.arange(920, 955, 2)
            #hough_res = hough_circle(kcanny, hough_radii)
            #accums, cx, cy, rad = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
            #Rad = int(rad[0])
            center = (int(1024-segx1cen),int(1024-segy1cen))
            #print(1024-center[0],1024-center[1])
            #scidata1 = shift(scidata1, (-(1024-center[0]), -(1024-center[1])), cval=1)
            #center=(1024,1024)
            Reduce = int(Rad*0.05)
            Radlb=int(Rad*1.05)
            R = int(Rad - Reduce)
            xbpTpix = 10 * 4
            chTpix = 10 * 4
            arTpix = 1000 * 4
            print(Rad,center)


    except:
        pass

    try:
        img2 = fits.open(pair_imgs[l][2])
        Fname2 =(os.path.splitext((pair_imgs[l][2].split(os.sep))[-1]))[0]
        scidata2 = img2[0].data
        DOB2 = img2[0].header['DATE_OBS']
        x2cen = img2[0].header['XCEN']
        y2cen = img2[0].header['YCEN']
        x2scale = img2[0].header['XSCALE']
        y2scale = img2[0].header['YSCALE']
        #scidata2 = shift(scidata2, ((x2cen / x2scale), (y2cen / y2scale)),cval=1)
        size2 = scidata2.shape



    except:
        print('skipped')
        pass

    #print(size1,size2)
    if size1 == size2:
        start = 1

    else:
        start = 0
        print('size not matched',DOB1, size1,size2)
        size_n_match.append(DOB1)


    if start == 1:
        #print(R,Rad,Rad*0.95)
        shapeArry.append(size1[0])
        m = scidata1.mean()
        dob_str = DOB1
        dob_obj = datetime.datetime.strptime(dob_str, '%Y-%m-%dT%H:%M:%S.%f')
        dtm = Time(dob_obj,format='datetime') #date and time
        obsDate=dtm.decimalyear
        l_DOB.append(np.round(obsDate,6))
        circ = np.zeros((size1))
        lb   = np.zeros((size1))
        CHmask = (np.where(sgData==8,255,0)).astype(np.uint8)
        BPmask = (np.where(sgData==16,255,0)).astype(np.uint8)
        #BPmask1 = np.zeros(size1, np.uint8)
        #ARmask = np.zeros(size1, np.uint8)
        ARmask =(np.where(sgData==32,255,0)).astype(np.uint8)
        BGmask = (np.where(sgData==2,255,0)).astype(np.uint8)

        #kernel = np.ones((15, 15), np.uint8)
        #CHmask = cv2.morphologyEx(CHmask, cv2.MORPH_CLOSE, kernel)
        CH_cont, hierarchy = cv2.findContours(CHmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        BP_cont, hierarchy = cv2.findContours(BPmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        AR_cont, hierarchy = cv2.findContours(ARmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #BG_cont, hierarchy = cv2.findContours(BGmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        #print(center)
        H = center[0]
        k = center[1]
        #print(H,k)
        circ = cv2.circle(circ, (H, k), R, (255, 0, 0), -1)  # disk
        cv2.circle(lb, (H, k), Radlb,(255, 0, 0), -1)
        cv2.circle(lb, (H, k), R, (0, 0, 0), -1)
        circle = circ.astype(np.bool_)
        Circle = np.invert(circle)  # hole
        mask = ms.array(scidata1, mask=Circle)
        disk = ms.array(scidata1, mask=circle) 


        
        ###################--Temp map--#############################################################################-
       
        #map2=sunpy.map.sources.XRTMap(img1[0].data,img1[0].header)# Almeh
        #map1=sunpy.map.sources.XRTMap(img2[0].data,img2[0].header)# Tipoly

        IR = abs(scidata2) / abs(scidata1)  # Ti-Poly/Al-Mesh

        sld2 = (l + 1) * 61  # selected data array
        sld1 = l * 61

        
        tresp2 = xrtpy.response.TemperatureResponseFundamental(img1[0].header['EC_FW2_'], DOB1) #Almesh
        tresp1 = xrtpy.response.TemperatureResponseFundamental(img2[0].header['EC_FW1_'], DOB1) #Alpoly
        #Tire = tresp1.temperature_response().value
        #Alre = tresp2.temperature_response().value
        #RaT=np.array(Tire/Alre)
        ratio =np.array(Tiresp[sld1:sld2] / Alresp[sld1:sld2]) 
    
        ratio=np.insert(ratio,0,0)
        temP=np.insert(temp[0:61],0,0) #np.insert(tresp1.CHIANTI_temperature.value,0,0)
        respf = interpolate.interp1d(ratio, temP)

        TempMap = np.zeros(IR.shape)
        IR= np.where(IR>ratio[-1],0,IR)
        IR= np.where(IR<ratio[1],0,IR)

        #for m in range(len(y1)):
        TempMap=respf(IR)
        TempMap_fi=TempMap #full image

        Slb= (np.where(lb,TempMap,0)).sum()
        disk = ms.array(TempMap, mask=circle)  # hided disk
        DD = disk * 0
        TempMap = DD.data
        

        #########################################################################################///////////////////
        # error calculataion
        T_e=np.log10(TempMap_fi)
        wvl = tresp1.channel_wavelength
        eVe = tresp1.ev_per_electron
        gain = tresp1.ccd_gain_right
        e2dn = (h.to(u.eV * u.s) * c.to(u.angstrom / u.s) / (wvl * eVe * gain)).value
        dwvl = wvl[1:] - wvl[:-1]
        dwvl = np.append(dwvl, dwvl[-1]).value
        Tmodel = tresp1.CHIANTI_temperature.value
        #print(Tmodel,temP)
        #print(len(Tire),len(Tmodel))
        logTmodel = np.log10(Tmodel)
        effarea1 = tresp1.effective_area().value
        effarea2 = tresp2.effective_area().value
        spect1 = tresp1.spectra().value
        spect2 = tresp2.spectra().value
        flux1 = Tiresp[sld1:sld2]#tresp1.temperature_response().value
        flux2 = Alresp[sld1:sld2]#tresp2.temperature_response().value
        model_ratio = flux1/flux2
        dlnR_dlnT_mod = np.abs(deriv(np.log(Tmodel), np.log(model_ratio)))
        dlnR_dlnT = np.interp(T_e, logTmodel, dlnR_dlnT_mod, left=0.0, right=0.0)
        dlnR_dlnT = np.ma.masked_where(((dlnR_dlnT == 0.0) | (T_e <= 0.0)), dlnR_dlnT)
        
        #print(len(dwvl),len(e2dn))

        K1_mod = np.array([(s1 * effarea1 * e2dn**2 * dwvl).sum()/ (s1 * effarea1 * e2dn * dwvl).sum() for s1 in spect1])
        K2_mod = np.array([(s2 * effarea2 * e2dn**2 * dwvl).sum()/ (s2 * effarea2 * e2dn * dwvl).sum() for s2 in spect2])
        K1 = np.interp(T_e, logTmodel, K1_mod, left=0.0, right=0.0)
        K2 = np.interp(T_e, logTmodel, K2_mod, left=0.0, right=0.0)
        K1 = np.ma.masked_where(((K1 == 0.0) | (T_e <= 0.0)), K1)
        K2 = np.ma.masked_where(((K2 == 0.0) | (T_e <= 0.0)), K2)
        dlnf1_dlnT_mod = deriv(np.log(Tmodel), np.log(flux1))
        dlnf1_dlnT = np.interp(T_e, logTmodel, dlnf1_dlnT_mod, left=0.0, right=0.0)
        #print(T_e[512,512],np.count_nonzero(dlnf1_dlnT))
        dlnf1_dlnT = np.ma.masked_where(((dlnf1_dlnT == 0.0) | (T_e <= 0.0)), dlnf1_dlnT)
        
        
        dlnf2_dlnT_mod = deriv(np.log(Tmodel), np.log(flux2))
        dlnf2_dlnT = np.interp(T_e, logTmodel, dlnf2_dlnT_mod, left=0.0, right=0.0)
        dlnf2_dlnT = np.ma.masked_where(((dlnf2_dlnT == 0.0) | (T_e <= 0.0)), dlnf2_dlnT)
        
        data1 = scidata2#Ti_poly #map1.data # np.ma.masked_where(map1.mask, map1.data)
        data2 = scidata1#almesh  #map2.data #np.ma.masked_where(map2.mask, map2.data)
        T_error = np.ma.log10(np.sqrt(K1 / data1 + K2 / data2) / dlnR_dlnT) + T_e
        T_error = T_error.filled(0.0)
        T_error=np.array([list(item) for item in T_error])
        ERMap=10**(T_error)
        Kd1 = np.sqrt(K1 / data1)
        Kd1 = Kd1.filled(0.0)
        Kd2 = np.sqrt(K2 / data2)
        Kd2 = Kd2.filled(0.0)
        #print((10**(T_error[520,530]))/1000000,(10**(T_e[520,530]))/1000000)
        

        TempMap_fi=np.where((T_error - T_e) <= np.log10(0.5),TempMap_fi,0)
        #TempMap=np.where((Kd1 <= 0.5),TempMap,0)
        #TempMap=np.where((Kd2 <= 0.5),TempMap,0)

        #
        ct=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        xcorr=int(int(img1[0].header['CRPIX1'])-H)*img1[0].header['XSCALE']
        ycorr=int((int(img1[0].header['CRPIX2'])-k))*img1[0].header['YSCALE']

        hdu = fits.PrimaryHDU(TempMap_fi)
        dlen=len(img1[0].header)-1
        key_list=list((img1[0].header).keys()) #Keyword list
        hist_pos=key_list.index('HISTORY')
        hdu.header['DATE']=  ct #datetime.datetime.now() #img1[0].header['DATE']
        hdu.header['DATE_OBS']=img1[0].header['DATE_OBS']
        hdu.header['TIME-OBS']=img1[0].header['TIME-OBS'] 
        hdu.header['FL_ORIG1']= Fname1+'.fits'
        hdu.header['FL_ORIG2']= Fname2+'.fits'
        hdu.header['SATELLIT']=img1[0].header['SATELLIT']
        hdu.header['TELESCOP']=img1[0].header['TELESCOP']
        hdu.header['INSTRUME']=img1[0].header['INSTRUME']
        hdu.header['ORIGIN']= 'H.N. Adithya'
        hdu.header['CONTRIB']= 'Kariyappa et. al'
        hdu.header['PROG_VER']= 'DB_V1.00_Temp_Al_Ap_Oct_23'
        hdu.header['CRPIX1']= img1[0].header['CRPIX1']
        hdu.header['CRPIX2']= img1[0].header['CRPIX2']
        hdu.header['CRVAL1']= img1[0].header['CRVAL1']
        hdu.header['CRVAL2']= img1[0].header['CRVAL2']
        hdu.header['CDELT1']= img1[0].header['CDELT1']
        hdu.header['CDELT2']= img1[0].header['CDELT2']
        hdu.header['CUNIT1']= img1[0].header['CUNIT1']
        hdu.header['CUNIT2']= img1[0].header['CUNIT2']
        hdu.header['CTYPE1']= (img1[0].header['CTYPE1'], 'Provide by XRT')
        hdu.header['CTYPE2']= (img1[0].header['CTYPE2'], 'Provide by XRT')
        if 'DSUN_OBS' in img1[0].header:
           hdu.header['DSUN_OBS']= (img1[0].header['DSUN_OBS'],'Provide by XRT')
           hdu.header['RSUN_REF']= (img1[0].header['RSUN_REF'],'Provide by XRT')
           hdu.header['SOLAR_B0']= img1[0].header['SOLAR_B0']
        hdu.header['RSUN_OBS']= (Rad*img1[0].header['YSCALE'],'By circle fit')
        hdu.header['SAT_ROT']= img1[0].header['SAT_ROT']
        hdu.header['INST_ROT']= img1[0].header['INST_ROT']
        hdu.header['CROTA1']= img1[0].header['CROTA1']
        hdu.header['CROTA2']= img1[0].header['CROTA2']
        #hdu.header['CROTA1']= img1[0].header['CROTA1']
        hdu.header['XCEN']= (xcorr,'By circle fit' )
        hdu.header['YCEN']= (ycorr,'By circle fit' )
        if 'CALTYPE' in img1[0].header:        
            hdu.header['CALTYPE']= img1[0].header['CALTYPE']
        hdu.header['XSCALE']= img1[0].header['XSCALE']
        hdu.header['YSCALE']= img1[0].header['YSCALE']
        hdu.header['FOVX']= img1[0].header['FOVX']
        hdu.header['FOVY']= img1[0].header['FOVY']
        hdu.header['PLATESCL']= img1[0].header['PLATESCL']
        hdu.header['BYTECNT']= img1[0].header['BYTECNT']
        hdu.header['PIXCNT']= img1[0].header['PIXCNT']
        hdu.header['BITSPP']= img1[0].header['BITSPP']
        hdu.header['HISTORY']=img1[0].header[hist_pos:(dlen+1)]
        
        #(hdu.header).append('HISTORY')
        #hdu.header['HISTORY']= (img1[0].header['HISTORY'])
        del hdu.header['EXTEND']

        hdu.writeto("{}_temp.fits".format(Fname1), overwrite=True)
        zip_file = zipfile.ZipFile("{}_temp.fits.zip".format(Fname1), 'w')
        zip_file.write("{}_temp.fits".format(Fname1), compress_type=zipfile.ZIP_DEFLATED)
        zip_file.close()
        os.replace(os.getcwd()+"/{}_temp.fits.zip".format(Fname1), os.getcwd()+'/TempFits/{}_temp.fits.zip'.format(Fname1))
        os.remove("{}_temp.fits".format(Fname1))

        Ehdu = fits.PrimaryHDU(T_error)
        Ehdu.writeto("{}_tempEr.fits".format(Fname1),overwrite=True)
        zip_file = zipfile.ZipFile("{}_tempEr.fits.zip".format(Fname1), 'w')
        zip_file.write("{}_tempEr.fits".format(Fname1), compress_type=zipfile.ZIP_DEFLATED)
        zip_file.close()
        os.replace(os.getcwd()+"/{}_tempEr.fits.zip".format(Fname1), os.getcwd()+'/ERmaps/{}_tempEr.fits.zip'.format(Fname1))
        os.remove("{}_tempEr.fits".format(Fname1))

        PhduAl = fits.PrimaryHDU(Kd2)
        PhduAl.writeto("{}_tempAlPN.fits".format(Fname1),overwrite=True)
        zip_file = zipfile.ZipFile("{}_tempAlPN.fits.zip".format(Fname1), 'w')
        zip_file.write("{}_tempAlPN.fits".format(Fname1), compress_type=zipfile.ZIP_DEFLATED)
        zip_file.close()
        os.replace(os.getcwd()+"/{}_tempAlPN.fits.zip".format(Fname1), os.getcwd()+'/AlPNmaps/{}_tempAlPN.fits.zip'.format(Fname1))
        os.remove("{}_tempAlPN.fits".format(Fname1))
        
        PhduAp = fits.PrimaryHDU(Kd1)
        PhduAp.writeto("{}_tempApPN.fits".format(Fname1),overwrite=True)
        zip_file = zipfile.ZipFile("{}_tempApPN.fits.zip".format(Fname1), 'w')
        zip_file.write("{}_tempApPN.fits".format(Fname1), compress_type=zipfile.ZIP_DEFLATED)
        zip_file.close()
        os.replace(os.getcwd()+"/{}_tempApPN.fits.zip".format(Fname1), os.getcwd()+'/ApPNmaps/{}_tempApPN.fits.zip'.format(Fname1))
        os.remove("{}_tempApPN.fits".format(Fname1))
        
        os.remove("{}_seg.fits".format(Fname1))


        #/media/adithyahn/Adi_Backup_drive/FitsTemp_map

        IRdisk = ms.array(IR, mask=circle)  # hided disk
        IRDD = IRdisk * 0
        IRMap = IRDD.data
        IRsum = IRMap.sum()
        #hdu = fits.PrimaryHDU(IRMap)
        #hdu.writeto("IR_fits/IR_Map{}.fits".format(DOB1),overwrite=True)

        index = np.nonzero(TempMap)
        mtxData1 = TempMap[index[0], index[1]]
        s1 = np.std(mtxData1)
        mn1 = mtxData1.mean()
        md1 = np.median(mtxData1)
        mn1_mk = mn1 / 1000000
        ######
        
        zimage = av.ZScaleInterval(n_samples=600, contrast=0.25, max_reject=0.5, min_npixels=5, krej=2.5,
                                   max_iterations=5)
        z1 = zimage.get_limits(TempMap_fi)
        ZI1 = np.clip(TempMap_fi, z1[0], z1[1])  # zimage(scidata)
        ZS1 = (ZI1 / ZI1.max()) * 255
        tsunz = ((ZS1).astype(np.uint8))
        comb_Temp = cv2.cvtColor(tsunz, cv2.COLOR_GRAY2BGR)
        '''
        ######--IR contribution---#######
        # scaling IR map
        zimage = av.ZScaleInterval(nsamples=600, contrast=0.25, max_reject=0.5, min_npixels=5, krej=2.5,
                                   max_iterations=5)
        z2 = zimage.get_limits(IRMap)
        ZI2 = np.clip(IRMap, z2[0], z2[1])  # zimage(scidata)
        ZS2 = (ZI2 / ZI2.max()) * 255
        IRz = ((ZS2).astype(np.uint8))

        CH_IR = cv2.cvtColor(IRz, cv2.COLOR_GRAY2BGR)
        BP_IR = cv2.cvtColor(IRz, cv2.COLOR_GRAY2BGR)
        AR_IR = cv2.cvtColor(IRz, cv2.COLOR_GRAY2BGR)
        BG_IR = cv2.cvtColor(IRz, cv2.COLOR_GRAY2BGR)
        #BG_Ir = cv2.cvtColor(IRz, cv2.COLOR_GRAY2BGR) #why?
        comb_IR = cv2.cvtColor(IRz, cv2.COLOR_GRAY2BGR)
        
        #Image
        xrt_Tmap=sunpy.map.sources.XRTMap(TempMap_fi,img1[0].header)
        Tfig = plt.figure()
        Tax = plt.subplot(projection=xrt_Tmap)
        xrt_Tmap.plot(cmap='gray',clip_interval=(10, 100)*u.percent)
        Tax.set_title('')
        plt.axis('off')
        
        if size1[0]==2048:
          Tfig.set_size_inches(20.48, 20.48)
        else: 
          Tfig.set_size_inches(10.24, 10.24)
        Tfig.subplots_adjust(left=0, bottom=0, right=1, top=1) 
        Tfig.canvas.draw()
        plot_Tmatrix = np.array(Tfig.canvas.renderer.buffer_rgba())
        plot_Tmatrix=(plot_Tmatrix[:,:,:3])[::-1]
        #plt.show(block=False)
        Tfig.clf()
        plt.close('all')

        comb_Temp=cv2.cvtColor(plot_Tmatrix, cv2.COLOR_RGB2BGR)
        '''

        CHirMask = np.zeros(size1, np.uint8)
        BPirMask = np.zeros(size1, np.uint8)
        ARirMask = np.zeros(size1, np.uint8)
        ARirMask1=np.zeros(size1, np.uint8)
        BGirMask = np.zeros(size1, np.uint8)

        cv2.drawContours(BGirMask, CH_cont, -1, (255, 0, 0), cv2.FILLED)
        cv2.drawContours(BGirMask, BP_cont, -1, (255, 0, 0), cv2.FILLED)
        cv2.drawContours(BGirMask, AR_cont, -1, (255, 0, 0), cv2.FILLED)
        cv2.drawContours(ARirMask1, AR_cont, -1, (255, 0, 0), cv2.FILLED)
        cv2.drawContours(CHirMask, CH_cont, -1, (255, 0, 0), cv2.FILLED)
        cv2.drawContours(BPirMask, BP_cont, -1, (255, 0, 0), cv2.FILLED)
        #cv2.drawContours(CH_IR, CHcont, -1, (255, 0, 255), 2)
        #cv2.drawContours(BP_IR, BP_cont, -1, (255, 0, 0), 2)
        #cv2.drawContours(comb_IR, CHcont, -1, (255, 0, 255), 2)
        #cv2.drawContours(comb_IR, BP_cont, -1, (255, 0, 0), 2)
        #cv2.drawContours(comb_IR, AR_cont, -1, (255, 255, 0), 2)

        Bo_CHirMask = CHirMask.astype(np.bool_)
        Bo_BPirMask = BPirMask.astype(np.bool_)
        Bo_BGirMask = BGirMask.astype(np.bool_)
        Bo_ARirMask1 = ARirMask1.astype(np.bool_)
        In_CHirMask = np.invert(Bo_CHirMask)
        In_BGirMask = np.invert(Bo_BGirMask)
        In_BPirMask = np.invert(Bo_BPirMask)
        In_ARirMask1 = np.invert(Bo_ARirMask1)

        # masking
        CH_IRmasked_sun = ms.array(IRMap, mask=In_CHirMask)
        BP_IRmasked_sun = ms.array(IRMap, mask=In_BPirMask)
        BG_IRmasked_sun = ms.array(IRMap, mask=Bo_BGirMask)
        AR_IRmasked1_sun = ms.array(IRMap, mask=In_ARirMask1)
        CH_IRmasked_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),CH_IRmasked_sun)
        AR_IRmasked1_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),AR_IRmasked1_sun)
        BG_IRmasked_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),BG_IRmasked_sun)
        BP_IRmasked_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),BP_IRmasked_sun)

        bgIRimage = ms.array(IRMap, mask=In_BGirMask)
        BGIr = ((bgIRimage * 0).data).astype(np.uint8)
        BPIr = ((BP_IRmasked_sun * 0).data).astype(np.uint8)
        BP_IRmasked1_sun = ms.array(IRMap, mask=In_BPirMask)
        bpIRsum1 = BP_IRmasked1_sun.sum()# both are same
        #print(bpIRsum1,BP_IRmasked_sun.sum())
        # Numebrs
        no_of_AR = (len(AR_cont))
        no_of_BP = (len(BP_cont))
        no_of_CH = len(CH_cont)

        #nan correction
        IRar=AR_IRmasked1_sun.sum()
        IRch=CH_IRmasked_sun.sum()
        IRbp=BP_IRmasked_sun.sum()

        if no_of_BP == 0:
            IRbp = 0
        if no_of_CH == 0:
            IRch = 0

        if no_of_AR == 0:
            IRar = 0

        bpIRsum.append(IRbp)
        chIRsum.append(IRch)
        ArIRsum.append(IRar)
        bgIRsum.append(BG_IRmasked_sun.sum())

        ######-------|||-------###########

        # Error maps #
        '''
        z3 = zimage.get_limits(ERMap)
        ZI3 = np.clip(ERMap, z3[0], z3[1])  # zimage(scidata)
        ZS3 = (ZI3 / ZI3.max()) * 255
        ERz = ((ZS3).astype(np.uint8))
        '''

        #comb_Er = cv2.cvtColor(ERz, cv2.COLOR_GRAY2BGR)
        CHerMask = CHmask #np.zeros(size1, np.uint8)
        BPerMask = BPmask#np.zeros(size1, np.uint8)
        ARerMask = ARmask#np.zeros(size1, np.uint8)
        ARerMask1= ARmask#np.zeros(size1, np.uint8)
        BGerMask = BGmask#np.zeros(size1, np.uint8)

        #cv2.drawContours(BGerMask, CH_cont, -1, (255, 0, 0), cv2.FILLED)
        ##cv2.drawContours(BGerMask, AR_cont, -1, (255, 0, 0), cv2.FILLED)
        #cv2.drawContours(ARerMask1, AR_cont, -1, (255, 0, 0), cv2.FILLED)
        ##cv2.drawContours(BPerMask, BP_cont, -1, (255, 0, 0), cv2.FILLED)
        #cv2.drawContours(CH_er, CHcont, -1, (255, 0, 255), 2)
        #cv2.drawContours(BP_er, BP_cont, -1, (255, 0, 0), 2)
        #cv2.drawContours(comb_er, CHcont, -1, (255, 0, 255), 2)
        #cv2.drawContours(comb_er, BP_cont, -1, (255, 0, 0), 2)
        #cv2.drawContours(comb_er, AR_cont, -1, (255, 255, 0), 2)

        Bo_CHerMask = CHerMask.astype(np.bool_)
        Bo_BPerMask = BPerMask.astype(np.bool_)
        Bo_BGerMask = BGerMask.astype(np.bool_)
        Bo_ARerMask1 = ARerMask1.astype(np.bool_)
        In_CHerMask = np.invert(Bo_CHerMask)
        In_BGerMask = np.invert(Bo_BGerMask)
        In_BPerMask = np.invert(Bo_BPerMask)
        In_ARerMask1 = np.invert(Bo_ARerMask1)

        # masking
        CH_ERmasked_sun = ms.array(ERMap, mask=In_CHerMask)
        BP_ERmasked_sun = ms.array(ERMap, mask=In_BPerMask)
        BG_ERmasked_sun = ms.array(ERMap, mask=In_BGerMask)
        AR_ERmasked1_sun = ms.array(ERMap, mask=In_ARerMask1)

        CH_ERmasked_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),CH_ERmasked_sun)
        AR_ERmasked1_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),AR_ERmasked1_sun)
        BG_ERmasked_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),BG_ERmasked_sun)
        BP_ERmasked_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),BP_ERmasked_sun)

        bgERimage = ms.array(IRMap, mask=In_BGirMask)
        BGEr = ((bgIRimage * 0).data).astype(np.uint8)
        BPEr = ((BP_IRmasked_sun * 0).data).astype(np.uint8)
        #BP_ERmasked1_sun = ms.array(ERmap, mask=In_BPerMask)
        #bpERsum1 = BP_ERmasked1_sun.sum()# both are same
        #print(bpIRsum1,BP_IRmasked_sun.sum())
        

        #nan correction
        ERar=AR_ERmasked1_sun.sum()
        ERch=CH_ERmasked_sun.sum()
        ERbp=BP_ERmasked_sun.sum()

        if no_of_BP == 0:
            ERbp = 0
        if no_of_CH == 0:
            ERch = 0

        if no_of_AR == 0:
            ERar = 0

        bpERsum.append(ERbp)
        chERsum.append(ERch)
        arERsum.append(ERar)
        bgERsum.append(BG_ERmasked_sun.sum())
        fdERsum.append(ERMap.sum()) #averaging is taken care in plotter


        ##---|||----#
        #Temp map
        '''
        CH_Temp = cv2.cvtColor(tsunz, cv2.COLOR_GRAY2BGR)
        BP_Temp = cv2.cvtColor(tsunz, cv2.COLOR_GRAY2BGR)
        AR_Temp = cv2.cvtColor(tsunz, cv2.COLOR_GRAY2BGR)
        BG_Temp = cv2.cvtColor(tsunz, cv2.COLOR_GRAY2BGR)
        BG_T = cv2.cvtColor(tsunz, cv2.COLOR_GRAY2BGR)
        comb_Temp = cv2.cvtColor(tsunz, cv2.COLOR_GRAY2BGR)
        '''

        CHTmask = CHmask#np.zeros(size1, np.uint8)
        BPTmask = BPmask#np.zeros(size1, np.uint8)
        ARTmask = ARmask#np.zeros(size1, np.uint8)
        ARTmask = ARmask#np.zeros(size1, np.uint8)
        BGTmask = BGmask#np.zeros(size1, np.uint8)

        #cv2.drawContours(BGTmask, CH_cont, -1, (255, 0, 0), cv2.FILLED)
        #cv2.drawContours(BGTmask, BP_cont, -1, (255, 0, 0), cv2.FILLED)
        #cv2.drawContours(BGTmask, AR_cont, -1, (255, 0, 0), cv2.FILLED)
        #cv2.drawContours(ARTmask, AR_cont, -1, (255, 0, 0), cv2.FILLED)
        #cv2.drawContours(CH_Temp, CHcont, -1, (255, 0, 255), 2)
        #cv2.drawContours(BP_Temp, BP_cont, -1, (255, 0, 0), 2)
        cv2.drawContours(comb_Temp, CH_cont, -1, (255, 0, 255), 2)
        cv2.drawContours(comb_Temp, BP_cont, -1, (255, 0, 0), 2)
        cv2.drawContours(comb_Temp, AR_cont, -1, (255, 255, 0), 2)
        #cv2.drawContours(CHTmask, CH_cont, -1, (255, 0, 0), cv2.FILLED)
        #cv2.drawContours(BPTmask, BP_cont, -1, (255, 0, 0), cv2.FILLED)

        Bo_CHmask = CHTmask.astype(np.bool_)
        Bo_BPmask = BPTmask.astype(np.bool_)
        Bo_BGmask = BGTmask.astype(np.bool_)
        Bo_ARmask = ARTmask.astype(np.bool_)
        In_CHmask = np.invert(Bo_CHmask)
        In_BGmask = np.invert(Bo_BGmask)
        In_BPmask = np.invert(Bo_BPmask)
        In_ARmask = np.invert(Bo_ARmask)

        # masking
        CH_masked_sun = ms.array(TempMap, mask=In_CHmask)
        BP_masked_sun = ms.array(TempMap, mask=In_BPmask)
        BG_masked_sun = ms.array(TempMap, mask=In_BGmask)
        AR_masked1_sun = ms.array(TempMap, mask=In_ARmask)
        #print('before',ms.MaskedArray.count(CH_masked_sun),ms.MaskedArray.count(AR_masked1_sun))
        #error threshold
        CH_masked_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),CH_masked_sun)
        AR_masked1_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),AR_masked1_sun)
        BG_masked_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),BG_masked_sun)
        BP_masked_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),BP_masked_sun)

        
        #CH_masked_sun = ms.masked_where((Kd1>0.2 ),CH_masked_sun)
        #CH_masked_sun = ms.masked_where((Kd2>0.2 ),CH_masked_sun)

        #AR_masked1_sun = ms.masked_where(((T_error - T_e) > np.log10(0.5) ),AR_masked1_sun)
        #AR_masked1_sun = ms.masked_where((Kd1>0.2 ),AR_masked1_sun)
        #AR_masked1_sun = ms.masked_where((Kd2>0.2 ),AR_masked1_sun)
        #print('aftC',ms.MaskedArray.count(CH_masked_sun),ms.MaskedArray.count(AR_masked1_sun))

        bgimage = ms.array(TempMap, mask=In_BGmask)
        BGI = ((bgimage * 0).data).astype(np.uint8)
        BPI = ((BP_masked_sun * 0).data).astype(np.uint8)
        BP_masked1_sun = ms.array(TempMap, mask=In_BPmask) # not using
        bpsum1 = BP_masked1_sun.sum() #not using


        # print (no_of_BP, no_of_AR)
        n_AR.append(no_of_AR)
        n_BP.append(no_of_BP)
        n_CH.append(no_of_CH)

        # Totel area
        cha_ = np.count_nonzero(CHTmask)
        bpa_ = np.count_nonzero(BPTmask)
        ara_ = np.count_nonzero(ARTmask)

        cha = ms.MaskedArray.count(CH_masked_sun)#Counts the un masked elements##np.count_nonzero(CHTmask)
        bpa = ms.MaskedArray.count(BP_masked_sun)#np.count_nonzero(BPTmask)
        ara = ms.MaskedArray.count(AR_masked1_sun)#np.count_nonzero(ARTmask1)
        fda = np.count_nonzero(TempMap)  # fuldisk size
        crc_area= np.pi*R*R
        c_rat=round(((fda/crc_area)*100),2) #convertion ratio
        A4 = np.count_nonzero(In_BGmask)  # Bg + oudside disk
        A6 = size1[0] * size1[1]
        A7 = A6 - crc_area 
        bga_ = A4 - A7#bg count
        bga= ms.MaskedArray.count(BG_masked_sun)

        '''
        CH_masked_sun=(CH_masked_sun.filled(50)).astype(np.uint8)
        BP_masked_sun=(BP_masked_sun.filled(50)).astype(np.uint8)
        BG_masked_sun=(BG_masked_sun.filled(50)).astype(np.uint8)
        AR_masked1_sun=(AR_masked1_sun.filled(50)).astype(np.uint8)
        
        imageio.imwrite('ch_img{}.jpg'.format(Fname1), CH_masked_sun[::-1])
        imageio.imwrite('ar_img{}.jpg'.format(Fname1), AR_masked1_sun[::-1])
        imageio.imwrite('bg_img{}.jpg'.format(Fname1), BG_masked_sun[::-1])
        imageio.imwrite('bo_img{}.jpg'.format(Fname1), BP_masked_sun[::-1])
        '''
        #no need to correct area since it does itself since we are dealing with average values
        '''
        if size1[0] == 2048: #area corrected for big size
            cha = cha / 2
            bpa = bpa / 2
            ara = ara / 2
            A8=A8/2
        '''
        limbA= (np.pi *Radlb*Radlb)-(np.pi *R*R)


        CHA.append(cha_)
        BPA.append(bpa_)
        ARA.append(ara_)
        BGA.append(bga_)
        FDA.append(crc_area)
        # Error masked area
        CHa.append(cha)
        BPa.append(bpa)
        ARa.append(ara)
        BGa.append(bga)
        FDa.append(fda)
        conv_rat.append(c_rat)
        LBa.append(limbA)

        # Totel Temperature
        bpsum = BP_masked_sun.sum()
        csum = CH_masked_sun.sum()
        bgsum = BG_masked_sun.sum()
        Arsum = AR_masked1_sun.sum()
        tsum = TempMap.sum()

        if no_of_BP == 0:
            bgsum = 0
        if no_of_CH == 0:
            csum = 0
            ch_temp =0
        else:
            ch_temp = csum / cha
        if no_of_AR == 0:
            Arsum = 0
            ar_temp =0
        else:
            ar_temp = Arsum / ara

        '''
        if size1[0]==2048:
            bpsum = BP_masked_sun.sum()/2
            csum = CH_masked_sun.sum()/2
            bgsum = BG_masked_sun.sum()/2
            Arsum = AR_masked1_sun.sum()/2
            tsum = TempMap.sum()/2
        '''
        # Totel=csum+bgsum+bpsum+Arsum
        XBP = bpsum  ##
        # Average temperature
        bp_temp = bpsum / bpa
        bg_temp = bgsum / bga
        FD_temp = tsum / fda
        #Totel = ar_temp + bg_temp + bp_temp + ch_temp

        IR_data.append(IRsum)#area not corrected, will be done in plotter
        CHarray.append(ch_temp)
        BParray.append(bp_temp)
        ARarray.append(ar_temp)
        BGarray.append(bg_temp)
        FDarray.append(FD_temp)
        LBarray.append(Slb)
        CHi.append((((csum) / tsum) * 100))
        BPi.append((((bpsum) / tsum) * 100))
        ARi.append((((Arsum) / tsum) * 100))
        BGi.append((((bgsum) / tsum) * 100))

        dname = Fname1+Fname2  
        l_dname.append(dname)
        #print(dname)
        color1 = (255, 165, 0)  #Orange
        color2 = (64, 254, 208) #Blue

        
        cv2.circle(comb_Temp,(H,k), R, color2, 1)
        cv2.circle(comb_Temp,(H,k), Rad, color1, 1)
        cv2.circle(comb_Temp,(H,k), (Radlb), color2, 1)
        
        
        imageio.imwrite('Al_Ap_TempMaps/{}_temp.jpg'.format(Fname1), comb_Temp[::-1])
        #imageio.imwrite('ERmaps/E_map_{}.jpg'.format(dname), ERz[::-1])
        #imageio.imwrite('imgs2/IR_{}.jpg'.format(dob_str), Org_img[::-1])
        #imageio.imwrite('IRmaps/COMB_img{}.jpg'.format(dob_str), IRz[::-1])
        f = open('V1_Al_Ap_Temp_data_1.dat', 'a')
        np.savetxt('V1_Al_Ap_Temp_data_1.dat', np.c_[
            CHarray, BParray, ARarray, BGarray,LBarray, CHi, BPi, ARi, BGi, n_BP, n_AR, n_CH, l_DOB, CHa, BPa, ARa, BGa,LBa, FDarray, IR_data,chIRsum,bpIRsum,ArIRsum,bgIRsum,conv_rat,shapeArry,fdERsum,chERsum,bpERsum,arERsum,bgERsum,CHA,BPA,ARA,BGA,FDA],
                   fmt='%11.6f',
                   header=' CH Int,   XBP Int,     AR Int,  Background Limb | %intensity-CH   XBP          AR         BG		nXBP		nAR		nCH		l_DOB	    CHa       BPa       ARa        BGa    LBa     FD-int    Int-rat     chIR    xbpIR    arIR    bgIR   conversion%     shape    FD error    CH error  BPerror   ARerror  BGerror  CHA   BPA   ARA  BGA   FDA' )
        f.close()
        fa = open('V1_Database_Al_Ap_Temp_data_11_10_23_1.dat', 'a')
        np.savetxt('V1_Database_Al_Ap_Temp_data_11_10_23_1.dat', np.c_[
            l_dname, l_DOB, FDarray, ARarray, BParray, CHarray, BGarray, LBarray, ARi, BPi, CHi, BGi, n_AR, n_BP, n_CH, FDarray, ARa, BPa, CHa , BGa,LBa, conv_rat, shapeArry],
                   fmt='%s',
                   header='                Img pair name                            |      DOB        |        FDT     |      ARt    |     XBPt   |       CHt    |       BGt    |  Limbt  |    AR%    |    XBP%      |     CH%   |      BG%     |    n_AR    |     nXBP    |     nCH    |  FDa |  ARa  | XBPa  |  CHa  |  BGa   |  LBa   |   conversion%  |   shape')
        fa.close()
        tempstopTime = timeit.default_timer()
        filetime = np.round((tempstopTime - prvTime),2)
        prvTime=tempstopTime
        print('[', l + 1, '/', Length, '] ', DOB1, 'Avg. FDT', mn1_mk.round(2),'CR',c_rat ,'TIME', filetime)

stopTime = timeit.default_timer()
runtime = (stopTime - startTime)
TotTime=runtime/3600 #in Hours

print('')
print('......COMPLEETED.....')
print('Time taken', TotTime,'Avg. Time per image', runtime/Length)
print('size not matched', len(size_n_match))
print('--------------------------------------------------')
