#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/thesispro/resources/mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!
import rospy
from std_msgs.msg import String
from thesispro.msg import screen
from thesispro.msg import WmState
from PyQt5 import QtCore, QtGui, QtWidgets
from enum import Enum

class Programs(Enum):
    algod = 1
    sinteticos = 2
    delicados = 3
    lana = 4
    vapor = 5
    okopower = 6
    antialergia = 7
    twentymin = 8
    outdoor = 9
    jeans = 10
class Options(Enum):
    aclarado = 1
    centrif = 2
    drenar = 3


class Ui_MainWindow(object):
    #digits information assigning to the LCDS
    def callbackUserInterface(self,data):
        self.lcdKg.display(float(data.Kilogram))
        self.lcdTemp.display(float(data.temperature))
        self.lcdCentr.display(float(data.User_select_Rinse_Nums))
        self.lcdHour.display(float(data.User_Select_Delay_Time))
        self.lcdSpeed.display(float(data.Spin_speed))
    #results of wash program, option, function identification
    def callbackSymbolProgramOption(self,data):
        if data.NormalTM2L7 == True:
            self.chkbxNormalTM2L7.setChecked(True)
        else:
            self.chkbxNormalTM2L7.setChecked(False)
        #######
        if data.DailyTM2L7 == True:
            self.chkbxDailyTM2L7.setChecked(True)
        else:
            self.chkbxDailyTM2L7.setChecked(False)
        #######
        if data.SuperQuickTM2L7 == True:
            self.chkbxSuperQuickTM2L7.setChecked(True)
        else:
            self.chkbxSuperQuickTM2L7.setChecked(False)
        #######
        if data.Economy == True:
            self.chkbxEconomy.setChecked(True)
        else:
            self.chkbxEconomy.setChecked(False)
        #######
        if data.Prewash == True:
            self.chkbxPrewash.setChecked(True)
        else:
            self.chkbxPrewash.setChecked(False)
        #######
        if data.Stain == True:
            self.chkbxStain.setChecked(True)
        else:
            self.chkbxStain.setChecked(False)
        #######
        if data.ExtraRinse == True:
            self.chkbxExtraRinse.setChecked(True)
        else:
            self.chkbxExtraRinse.setChecked(False)
        #######
        if data.Soft == True:
            self.chkbxSoft.setChecked(True)
        else:
            self.chkbxSoft.setChecked(False)
        #######
        if data.Night_Cycle == True:
            self.chkbxNightCycle.setChecked(True)
        else:
            self.chkbxNightCycle.setChecked(False)
        #######
        if data.Rinse_Hold == True:
            self.chkbxRinseHold.setChecked(True)
        else:
            self.chkbxRinseHold.setChecked(False)
        #######
        if data.Steaming == True:
            self.chkbxSteaming.setChecked(True)
        else:
            self.chkbxSteaming.setChecked(False)
        #######
        if data.Steam_Anticrease == True:
            self.chkbxSteamAnticrease.setChecked(True)
        else:
            self.chkbxSteamAnticrease.setChecked(False)
        #######
        if data.Low_Steam == True:
            self.chkbxLowSteam.setChecked(True)
        else:
            self.chkbxLowSteam.setChecked(False)
        #######
        if data.Washing == True:
            self.chkbxWashing.setChecked(True)
        else:
            self.chkbxWashing.setChecked(False)
        #######
        if data.Rinsing == True:
            self.chkbxRinsing.setChecked(True)
        else:
            self.chkbxRinsing.setChecked(False)
        #######
        if data.Spinning == True:
            self.chkbxSpinning.setChecked(True)
        else:
            self.chkbxSpinning.setChecked(False)

        #For Programme
        if data.program == Programs.algod.value:
            self.radioAlgod.setChecked(True)
        elif data.program == Programs.sinteticos.value:
            self.radioSinteticos.setChecked(True)
        elif data.program == Programs.delicados.value:
            self.radioDelicados.setChecked(True)
        elif data.program == Programs.lana.value:
            self.radioLana.setChecked(True)
        elif data.program == Programs.vapor.value:
            self.radioVapor.setChecked(True)
        elif data.program == Programs.okopower.value:
            self.radioOko.setChecked(True)
        elif data.program == Programs.antialergia.value:
            self.radioAntiAlergia.setChecked(True)
        elif data.program == Programs.twentymin.value:
            self.radio20min.setChecked(True)
        elif data.program == Programs.outdoor.value:
            self.radioOutdoor.setChecked(True)
        elif data.program == Programs.jeans.value:
            self.radioJeans.setChecked(True)
        else:
            self.radioAlgod.setChecked(False)
            self.radioOko.setChecked(False)
            self.radioOutdoor.setChecked(False)
            self.radioDelicados.setChecked(False)
            self.radioJeans.setChecked(False)
            self.radioLana.setChecked(False)
            self.radio20min.setChecked(False)
            self.radioSinteticos.setChecked(False)
            self.radioAntiAlergia.setChecked(False)
            print("Error or the washing machine is not working!")

        #For Option
        if data.option == Options.aclarado.value:
            self.radioAclarado.setChecked(True)
        elif data.option == Options.centrif.value:
            self.radioCentrif.setChecked(True)
        elif data.option == Options.drenar.value:
            self.radioDrenar.setChecked(True)
        else:
            self.radioAclarado.setChecked(False)
            self.radioCentrif.setChecked(False)
            self.radioDrenar.setChecked(False)
            print("Error or not any option chosen!")


    #Publishing ros message coming from the washing machine
    def callbackRosMsg(self,data):
        #for LCDs
        #self.lcdKg_2.display(float(data.Kilogram))
        self.lcdTemp_2.display(str(data.temperature))
        self.lcdCentr_2.display(str(data.Spin_speed))
        self.lcdHour_2.display(str(data.User_Select_Delay_Time))
        self.lcdSpeed_2.display(str(data.User_select_Rinse_Nums))
        #for Symbols
        if data.NormalTM2L7 == True:
            self.chkbxNormalTM2L7_2.setChecked(True)
        else:
            self.chkbxNormalTM2L7_2.setChecked(False)
        #######
        if data.DailyTM2L7 == True:
            self.chkbxDailyTM2L7_2.setChecked(True)
        else:
            self.chkbxDailyTM2L7_2.setChecked(False)
        #######
        if data.SuperQuickTM2L7 == True:
            self.chkbxSuperQuickTM2L7_2.setChecked(True)
        else:
            self.chkbxSuperQuickTM2L7_2.setChecked(False)
        #######
        if data.Economy == True:
            self.chkbxEconomy_2.setChecked(True)
        else:
            self.chkbxEconomy_2.setChecked(False)
        #######
        if data.Prewash == True:
            self.chkbxPrewash_2.setChecked(True)
        else:
            self.chkbxPrewash_2.setChecked(False)
        #######
        if data.Stain == True:
            self.chkbxStain_2.setChecked(True)
        else:
            self.chkbxStain_2.setChecked(False)
        #######
        if data.ExtraRinse == True:
            self.chkbxExtraRinse_2.setChecked(True)
        else:
            self.chkbxExtraRinse_2.setChecked(False)
        #######
        if data.Washing == True:
            self.chkbxWashing_2.setChecked(True)
        else:
            self.chkbxWashing_2.setChecked(False)
        #######
        if data.Rinsing == True:
            self.chkbxRinsing_2.setChecked(True)
        else:
            self.chkbxRinsing_2.setChecked(False)
        #######
        if data.Spinning == True:
            self.chkbxSpinning_2.setChecked(True)
        else:
            self.chkbxSpinning_2.setChecked(False)
        if data.Soft == True:
            self.chkbxSoft_2.setChecked(True)
        else:
            self.chkbxSoft_2.setChecked(False)
        #######
        if data.Night_Cycle == True:
            self.chkbxNightCycle_2.setChecked(True)
        else:
            self.chkbxNightCycle_2.setChecked(False)
        #######
        if data.Rinse_Hold == True:
            self.chkbxRinseHold_2.setChecked(True)
        else:
            self.chkbxRinseHold_2.setChecked(False)
        #######
        if data.Steam_Anticrease == True:
            self.chkbxSteamAnticrease_2.setChecked(True)
        else:
            self.chkbxSteamAnticrease_2.setChecked(False)
        #######
        if data.Low_Steam == True:
            self.chkbxLowSteam_2.setChecked(True)
        else:
            self.chkbxLowSteam_2.setChecked(False)
        #######
        if data.Steaming == True:
            self.chkbxSteaming_2.setChecked(True)
        else:
            self.chkbxSteaming_2.setChecked(False)
        #######
        if data.Delay == True:
            self.chkbxDelay_2.setChecked(True)
        else:
            self.chkbxDelay_2.setChecked(False)
        #print ros message coming from the washing Machine

        #For Programme
        if data.User_Select_Program_ID == Programs.algod.value:
            self.radioAlgod_2.setChecked(True)
        elif data.User_Select_Program_ID == Programs.sinteticos.value:
            self.radioSinteticos_2.setChecked(True)
        elif data.User_Select_Program_ID == Programs.delicados.value:
            self.radioDelicados_2.setChecked(True)
        elif data.User_Select_Program_ID == Programs.lana.value:
            self.radioLana_2.setChecked(True)
        elif data.User_Select_Program_ID == Programs.vapor.value:
            self.radioVapor_2.setChecked(True)
        elif data.User_Select_Program_ID == Programs.okopower.value:
            self.radioOko_2.setChecked(True)
        elif data.User_Select_Program_ID == Programs.antialergia.value:
            self.radioAntiAlergia_2.setChecked(True)
        elif data.User_Select_Program_ID == Programs.twentymin.value:
            self.radio20min_2.setChecked(True)
        elif data.User_Select_Program_ID == Programs.outdoor.value:
            self.radioOutdoor_2.setChecked(True)
        elif data.User_Select_Program_ID == Programs.jeans.value:
            self.radioJeans_2.setChecked(True)
        else:
            self.radioAlgod_2.setChecked(False)
            self.radioOko_2.setChecked(False)
            self.radioOutdoor_2.setChecked(False)
            self.radioDelicados_2.setChecked(False)
            self.radioJeans_2.setChecked(False)
            self.radioLana_2.setChecked(False)
            self.radio20min_2.setChecked(False)
            self.radioSinteticos_2.setChecked(False)
            self.radioAntiAlergia_2.setChecked(False)
            print("Error or the washing machine is not working!")




        if self.lcdTemp.value() == self.lcdTemp_2.value():
            self.checkSameTemp.setChecked(True)
        else:
            self.checkSameTemp.setChecked(False)

        #if self.lcdKg.value() and self.lcdKg_2.value():
        #    self.checkSameKg.setChecked(True)
        #else:
        #    self.checkSameKg.setChecked(False)

        if self.lcdCentr.value() == self.lcdCentr_2.value():
            self.checkSameCentr.setChecked(True)
        else:
            self.checkSameCentr.setChecked(False)

        if self.lcdHour.value() == self.lcdHour_2.value():
            self.checkSameHour.setChecked(True)
        else:
            self.checkSameHour.setChecked(False)

        if self.lcdSpeed.value() == self.lcdSpeed_2.value():
            self.checkSameSpeed.setChecked(True)
        else:
            self.checkSameSpeed.setChecked(False)


        if self.chkbxNormalTM2L7.isChecked() == self.chkbxNormalTM2L7_2.isChecked():
            self.chkbxNormalTM2L7_5.setChecked(True)
        else:
            self.chkbxNormalTM2L7_5.setChecked(False)
        #########
        if self.chkbxLowSteam.isChecked() == self.chkbxLowSteam_2.isChecked():
            self.chkbxLowSteam_5.setChecked(True)
        else:
            self.chkbxLowSteam_5.setChecked(False)
        #########
        if self.chkbxMediumSteam.isChecked() == self.chkbxMediumSteam_2.isChecked():
            self.chkbxMediumSteam_5.setChecked(True)
        else:
            self.chkbxMediumSteam_5.setChecked(False)
        #########
        if self.chkbxDailyTM2L7.isChecked() == self.chkbxDailyTM2L7_2.isChecked():
            self.chkbxDailyTM2L7_5.setChecked(True)
        else:
            self.chkbxDailyTM2L7_5.setChecked(False)
        #########
        if self.chkbxPrewashing.isChecked() == self.chkbxPrewashing_2.isChecked():
            self.chkbxPrewashing_5.setChecked(True)
        else:
            self.chkbxPrewashing_5.setChecked(False)
        #########
        if self.chkbxSuperQuickTM2L7.isChecked() == self.chkbxSuperQuickTM2L7_2.isChecked():
            self.chkbxSuperQuickTM2L7_5.setChecked(True)
        else:
            self.chkbxSuperQuickTM2L7_5.setChecked(False)
        #########
        if self.chkbxWashing.isChecked() == self.chkbxWashing_2.isChecked():
            self.chkbxWashing_5.setChecked(True)
        else:
            self.chkbxWashing_5.setChecked(False)
        #########
        if self.chkbxEconomy.isChecked() == self.chkbxEconomy_2.isChecked():
            self.chkbxEconomy_5.setChecked(True)
        else:
            self.chkbxEconomy_5.setChecked(False)
        #########
        if self.chkbxRinsing.isChecked() == self.chkbxRinsing_2.isChecked():
            self.chkbxRinsing_5.setChecked(True)
        else:
            self.chkbxRinsing_5.setChecked(False)
        #########
        if self.chkbxPrewash.isChecked() == self.chkbxPrewash_2.isChecked():
            self.chkbxPrewash_5.setChecked(True)
        else:
            self.chkbxPrewash_5.setChecked(False)
        #########
        if self.chkbxSpinning.isChecked() == self.chkbxSpinning_2.isChecked():
            self.chkbxSpinning_5.setChecked(True)
        else:
            self.chkbxSpinning_5.setChecked(False)
        #########
        if self.chkbxStain.isChecked() == self.chkbxStain_2.isChecked():
            self.chkbxStain_5.setChecked(True)
        else:
            self.chkbxStain_5.setChecked(False)
        #########
        if self.chkbxExtraRinse.isChecked() == self.chkbxExtraRinse_2.isChecked():
            self.chkbxExtraRinse_5.setChecked(True)
        else:
            self.chkbxExtraRinse_5.setChecked(False)
        #########
        if self.chkbxDrying.isChecked() == self.chkbxDrying_2.isChecked():
            self.chkbxDrying_5.setChecked(True)
        else:
            self.chkbxDrying_5.setChecked(False)
        #########
        if self.chkbxSoft.isChecked() == self.chkbxSoft_2.isChecked():
            self.chkbxSoft_5.setChecked(True)
        else:
            self.chkbxSoft_5.setChecked(False)
        #########
        if self.chkbxDelay.isChecked() == self.chkbxDelay_2.isChecked():
            self.chkbxDelay_5.setChecked(True)
        else:
            self.chkbxDelay_5.setChecked(False)
        #########
        if self.chkbxSteaming.isChecked() == self.chkbxSteaming_2.isChecked():
            self.chkbxSteaming_5.setChecked(True)
        else:
            self.chkbxSteaming_5.setChecked(False)
        #########
        if self.chkbxNightCycle.isChecked() == self.chkbxNightCycle_2.isChecked():
            self.chkbxNightCycle_5.setChecked(True)
        else:
            self.chkbxNightCycle_5.setChecked(False)
        #########
        if self.chkbxRinseHold.isChecked() == self.chkbxRinseHold_2.isChecked():
            self.chkbxRinseHold_5.setChecked(True)
        else:
            self.chkbxRinseHold_5.setChecked(False)
        #########
        if self.chkbxAnticrease.isChecked() == self.chkbxAnticrease_2.isChecked():
            self.chkbxAnticrease_5.setChecked(True)
        else:
            self.chkbxAnticrease_5.setChecked(False)
        #########
        if self.chkbxSteamAnticrease.isChecked() == self.chkbxSteamAnticrease_2.isChecked():
            self.chkbxSteamAnticrease_5.setChecked(True)
        else:
            self.chkbxSteamAnticrease_5.setChecked(False)
        #########
        if self.radioAlgod.isChecked() == self.radioAlgod_2.isChecked():
            self.checkSameAlgod.setChecked(True)
        else:
            self.checkSameAlgod.setChecked(False)
        ##########
        if self.radioSinteticos.isChecked() == self.radioSinteticos_2.isChecked():
            self.checkSameSinteticos.setChecked(True)
        else:
            self.checkSameSinteticos.setChecked(False)
        #########
        if self.radioDelicados.isChecked() == self.radioDelicados_2.isChecked():
            self.checkSameDelicados.setChecked(True)
        else:
            self.checkSameDelicados.setChecked(False)
        #########
        if self.radioLana.isChecked() == self.radioLana_2.isChecked():
            self.checkSameLana.setChecked(True)
        else:
            self.checkSameLana.setChecked(False)
        #########
        if self.radioVapor.isChecked() == self.radioVapor_2.isChecked():
            self.checkSameVapor.setChecked(True)
        else:
            self.checkSameVapor.setChecked(False)
        #########
        if self.radioOko.isChecked() == self.radioOko_2.isChecked():
            self.checkSameOko.setChecked(True)
        else:
            self.checkSameOko.setChecked(False)
        #########
        if self.radioAntiAlergia.isChecked() == self.radioAntiAlergia_2.isChecked():
            self.checkSameAntiAlergia.setChecked(True)
        else:
            self.checkSameAntiAlergia.setChecked(False)
        #########
        if self.radio20min.isChecked() == self.radio20min_2.isChecked():
            self.checkSame20min.setChecked(True)
        else:
            self.checkSame20min.setChecked(False)
        #########
        if self.radioOutdoor.isChecked() == self.radioOutdoor_2.isChecked():
            self.checkSameOutdoor.setChecked(True)
        else:
            self.checkSameOutdoor.setChecked(False)
        #########
        if self.radioJeans.isChecked() == self.radioJeans_2.isChecked():
            self.checkSameJeans.setChecked(True)
        else:
            self.checkSameJeans.setChecked(False)

    ####DESIGNING GUI################################
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1019, 791)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.frmScreen = QtWidgets.QFrame(self.centralWidget)
        self.frmScreen.setAutoFillBackground(False)
        self.frmScreen.setGeometry(QtCore.QRect(10, 10, 561, 191))
        self.frmScreen.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frmScreen.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frmScreen.setObjectName("frmScreen")
        palette = QtGui.QPalette() #insantiate style
        brush = QtGui.QBrush(QtGui.QColor(0,0,0)) #instantiate colour
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active,QtGui.QPalette.Light,brush)
        palette.setBrush(QtGui.QPalette.Inactive,QtGui.QPalette.Light,brush)
        palette.setBrush(QtGui.QPalette.Disabled,QtGui.QPalette.Light,brush)
        #################ROS MESSAGE DIGITS###################
        self.layoutWidget = QtWidgets.QWidget(self.frmScreen)
        self.layoutWidget.setGeometry(QtCore.QRect(0,40,102,141))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(11,11,11,11)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        #-----------Label showing name of LCDs------------
        self.labelTemp = QtWidgets.QLabel(self.layoutWidget)
        self.labelTemp.setObjectName("labelTemp")
        self.verticalLayout_2.addWidget(self.labelTemp)

        self.labelKg = QtWidgets.QLabel(self.layoutWidget)
        self.labelKg.setObjectName("labelKg")
        self.verticalLayout_2.addWidget(self.labelKg)

        self.labelSpinSpeed = QtWidgets.QLabel(self.layoutWidget)
        self.labelSpinSpeed.setObjectName("labelSpinSpeed")
        self.verticalLayout_2.addWidget(self.labelSpinSpeed)

        self.labelHour = QtWidgets.QLabel(self.layoutWidget)
        self.labelHour.setObjectName("labelHour")
        self.verticalLayout_2.addWidget(self.labelHour)


        self.labelSpeed = QtWidgets.QLabel(self.frmScreen)
        self.labelSpeed.setObjectName("labelSpeed")
        self.verticalLayout_2.addWidget(self.labelSpeed)
        #-----------------ROS MESSAGE DIGITS------------------#
        self.layoutWidget_3 = QtWidgets.QWidget(self.frmScreen)
        self.layoutWidget_3.setGeometry(QtCore.QRect(300,40,161,141))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.layoutWidget_3)
        self.verticalLayout_5.setContentsMargins(11,11,11,11)
        self.verticalLayout_5.setSpacing(6)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.lcdTemp_2 = QtWidgets.QLCDNumber(self.layoutWidget_3)
        self.lcdTemp_2.setEnabled(True)
        self.lcdTemp_2.setAutoFillBackground(False)
        self.lcdTemp_2.setPalette(palette)
        self.lcdTemp_2.setObjectName("lcdTemp_2")
        self.verticalLayout_5.addWidget(self.lcdTemp_2)
        self.lcdKg_2 = QtWidgets.QLCDNumber(self.layoutWidget_3)
        self.lcdKg_2.setPalette(palette)
        self.lcdKg_2.setObjectName("lcdKg_2")
        self.verticalLayout_5.addWidget(self.lcdKg_2)
        self.lcdCentr_2 = QtWidgets.QLCDNumber(self.layoutWidget_3)
        self.lcdCentr_2.setPalette(palette)
        self.lcdCentr_2.setObjectName("lcdCentr_2")
        self.verticalLayout_5.addWidget(self.lcdCentr_2)
        self.lcdHour_2 = QtWidgets.QLCDNumber(self.layoutWidget_3)
        self.lcdHour_2.setPalette(palette)
        self.lcdHour_2.setObjectName("lcdHour_2")
        self.verticalLayout_5.addWidget(self.lcdHour_2)
        self.lcdSpeed_2  = QtWidgets.QLCDNumber(self.layoutWidget_3)
        self.lcdSpeed_2.setPalette(palette)
        self.lcdSpeed_2.setObjectName("lcdSpeed_2")
        self.verticalLayout_5.addWidget(self.lcdSpeed_2)
        #---------Headers of Classification Info and ROS Message Info
        self.labelClassification = QtWidgets.QLabel(self.frmScreen)
        self.labelClassification.setGeometry(QtCore.QRect(120,10,151,17))
        self.labelClassification.setAlignment(QtCore.Qt.AlignCenter)
        self.labelClassification.setObjectName("labelClassification")
        self.labelNetwork = QtWidgets.QLabel(self.frmScreen)
        self.labelNetwork.setGeometry(QtCore.QRect(300,10,151,17))
        self.labelNetwork.setAlignment(QtCore.Qt.AlignCenter)
        self.labelNetwork.setObjectName("labelNetwork")
        #------------------------------------------------------------
        self.widgetSameCheckBoxesOfLCDs = QtWidgets.QWidget(self.frmScreen)
        self.widgetSameCheckBoxesOfLCDs.setGeometry(QtCore.QRect(470,40,39,151))
        self.widgetSameCheckBoxesOfLCDs.setObjectName("widgetSameCheckBoxesOfLCDs")
        self.verticalLayout_SameLCDs = QtWidgets.QVBoxLayout(self.widgetSameCheckBoxesOfLCDs)
        self.verticalLayout_SameLCDs.setContentsMargins(10,0,10,0)
        self.verticalLayout_SameLCDs.setSpacing(0)
        self.verticalLayout_SameLCDs.setObjectName("verticalLayout_SameLCDs")
        self.checkSameTemp = QtWidgets.QCheckBox(self.widgetSameCheckBoxesOfLCDs)
        self.checkSameTemp.setText("")
        self.checkSameTemp.setObjectName("checkSameTemp")
        self.verticalLayout_SameLCDs.addWidget(self.checkSameTemp)
        self.checkSameKg = QtWidgets.QCheckBox(self.widgetSameCheckBoxesOfLCDs)
        self.checkSameKg.setText("")
        self.checkSameKg.setObjectName("checkSameKg")
        self.verticalLayout_SameLCDs.addWidget(self.checkSameKg)
        self.checkSameCentr = QtWidgets.QCheckBox(self.widgetSameCheckBoxesOfLCDs)
        self.checkSameCentr.setText("")
        self.checkSameCentr.setObjectName("checkSameCentr")
        self.verticalLayout_SameLCDs.addWidget(self.checkSameCentr)
        self.checkSameHour = QtWidgets.QCheckBox(self.widgetSameCheckBoxesOfLCDs)
        self.checkSameHour.setText("")
        self.checkSameHour.setObjectName("checkSameHour")
        self.verticalLayout_SameLCDs.addWidget(self.checkSameHour)
        self.checkSameSpeed = QtWidgets.QCheckBox(self.widgetSameCheckBoxesOfLCDs)
        self.checkSameSpeed.setText("")
        self.checkSameSpeed.setObjectName("checkSameSpeed")
        self.verticalLayout_SameLCDs.addWidget(self.checkSameSpeed)

        #####CLASSIFICATION LCDS##########################
        self.widget = QtWidgets.QWidget(self.frmScreen)
        self.widget.setGeometry(QtCore.QRect(120,40,151,141))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(11,11,11,11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.lcdTemp = QtWidgets.QLCDNumber(self.widget)
        self.lcdTemp.setObjectName("lcdTemp")
        self.lcdTemp.setEnabled(True)
        self.lcdTemp.setAutoFillBackground(False)
        self.lcdTemp.setPalette(palette) #add own palette to LCD Object
        self.verticalLayout.addWidget(self.lcdTemp)
        self.lcdKg = QtWidgets.QLCDNumber(self.widget)
        self.lcdKg.setObjectName("lcdKg")
        self.lcdKg.setPalette(palette) #add own pallette to lcdKg object
        self.verticalLayout.addWidget(self.lcdKg)
        self.lcdCentr = QtWidgets.QLCDNumber(self.widget)
        self.lcdCentr.setObjectName("lcdCentr")
        self.lcdCentr.setPalette(palette) #add own palette to lcdCentr object
        self.verticalLayout.addWidget(self.lcdCentr)

        self.lcdHour = QtWidgets.QLCDNumber(self.widget)
        self.lcdHour.setObjectName("lcdHour")
        self.lcdHour.setPalette(palette) #add own palette to lcdHour object
        self.verticalLayout.addWidget(self.lcdHour)

        self.lcdSpeed = QtWidgets.QLCDNumber(self.widget)
        self.lcdSpeed.setFrameShape(QtWidgets.QFrame.Box)
        self.lcdSpeed.setObjectName("lcdSpeed")
        self.lcdSpeed.setPalette(palette)
        self.verticalLayout.addWidget(self.lcdSpeed)

        self.layoutWidget.raise_()
        self.lcdTemp.raise_()
        self.labelClassification.raise_()
        self.labelNetwork.raise_()
        self.checkSameTemp.raise_()
        self.checkSameKg.raise_()
        self.checkSameCentr.raise_()
        self.checkSameHour.raise_()
        self.checkSameSpeed.raise_()
        #----------------Button-------------------------------
        self.btnRun = QtWidgets.QPushButton(self.centralWidget)
        self.btnRun.setEnabled(True)
        self.btnRun.setGeometry(QtCore.QRect(440, 710, 101, 29))
        self.btnRun.setObjectName("btnRun")


        #-----------------------Programs------------------------
        self.frameProgram = QtWidgets.QFrame(self.centralWidget)
        self.frameProgram.setGeometry(QtCore.QRect(0, 200,541, 371))
        self.frameProgram.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frameProgram.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameProgram.setObjectName("frameProgram")
        #Programs obtained from ROS Message
        self.layoutWidget_2 = QtWidgets.QWidget(self.frameProgram)
        self.layoutWidget_2.setGeometry(QtCore.QRect(180,80,140,276))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_4.setContentsMargins(11,11,11,11)
        self.verticalLayout_4.setSpacing(6)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.radioAlgod_2 = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radioAlgod_2.setObjectName("radioAlgod_2")
        self.verticalLayout_4.addWidget(self.radioAlgod_2)
        self.radioSinteticos_2 = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radioSinteticos_2.setObjectName("radioSinteticos_2")
        self.verticalLayout_4.addWidget(self.radioSinteticos_2)
        self.radioDelicados_2 = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radioDelicados_2.setObjectName("radioDelicados_2")
        self.verticalLayout_4.addWidget(self.radioDelicados_2)
        self.radioLana_2 = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radioLana_2.setObjectName("radioLana_2")
        self.verticalLayout_4.addWidget(self.radioLana_2)
        self.radioVapor_2 = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radioVapor_2.setObjectName("radioVapor_2")
        self.verticalLayout_4.addWidget(self.radioVapor_2)
        self.radioOko_2 = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radioOko_2.setObjectName("radioOko_2")
        self.verticalLayout_4.addWidget(self.radioOko_2)
        self.radioAntiAlergia_2 = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radioAntiAlergia_2.setObjectName("radioAntiAlergia_2")
        self.verticalLayout_4.addWidget(self.radioAntiAlergia_2)
        self.radio20min_2 = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radio20min_2.setObjectName("radio20min_2")
        self.verticalLayout_4.addWidget(self.radio20min_2)
        self.radioOutdoor_2 = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radioOutdoor_2.setObjectName("radioOutdoor_2")
        self.verticalLayout_4.addWidget(self.radioOutdoor_2)
        self.radioJeans_2 = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radioJeans_2.setObjectName("radioJeans_2")
        self.verticalLayout_4.addWidget(self.radioJeans_2)

        #Headers for Programs
        self.lblProgram = QtWidgets.QLabel(self.frameProgram)
        self.lblProgram.setGeometry(QtCore.QRect(0, 10, 101, 17))
        self.lblProgram.setAlignment(QtCore.Qt.AlignCenter)
        self.lblProgram.setWordWrap(False)
        self.lblProgram.setObjectName("lblProgram")
        self.labelClassification_2 = QtWidgets.QLabel(self.frameProgram)
        self.labelClassification_2.setGeometry(QtCore.QRect(10, 50, 151, 17))
        self.labelClassification_2.setAlignment(QtCore.Qt.AlignCenter)
        self.labelClassification_2.setObjectName("labelClassification_2")
        self.labelNetwork_2 = QtWidgets.QLabel(self.frameProgram)
        self.labelNetwork_2.setGeometry(QtCore.QRect(180, 50, 141, 17))
        self.labelNetwork_2.setAlignment(QtCore.Qt.AlignCenter)
        self.labelNetwork_2.setObjectName("labelNetwork_2")
        #Programs obtained from Classification
        self.widget_Class_Programs = QtWidgets.QWidget(self.frameProgram)
        self.widget_Class_Programs.setGeometry(QtCore.QRect(10, 80, 140, 276))
        self.widget_Class_Programs.setObjectName("widget1")
        self.verticalLayout_Class_Programs = QtWidgets.QVBoxLayout(self.widget_Class_Programs)
        self.verticalLayout_Class_Programs.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_Class_Programs.setSpacing(6)
        self.verticalLayout_Class_Programs.setObjectName("verticalLayout_Class_Programs")

        self.radioAlgod = QtWidgets.QRadioButton(self.widget_Class_Programs)
        self.radioAlgod.setObjectName("radioAlgod")
        self.verticalLayout_Class_Programs.addWidget(self.radioAlgod)

        self.radioSinteticos = QtWidgets.QRadioButton(self.widget_Class_Programs)
        self.radioSinteticos.setObjectName("radioSinteticos")
        self.verticalLayout_Class_Programs.addWidget(self.radioSinteticos)

        self.radioDelicados = QtWidgets.QRadioButton(self.widget_Class_Programs)
        self.radioDelicados.setObjectName("radioDelicados")
        self.verticalLayout_Class_Programs.addWidget(self.radioDelicados)

        self.radioLana = QtWidgets.QRadioButton(self.widget_Class_Programs)
        self.radioLana.setObjectName("radioLana")
        self.verticalLayout_Class_Programs.addWidget(self.radioLana)

        self.radioVapor = QtWidgets.QRadioButton(self.widget_Class_Programs)
        self.radioVapor.setObjectName("radioVapor")
        self.verticalLayout_Class_Programs.addWidget(self.radioVapor)

        self.radioOko = QtWidgets.QRadioButton(self.widget_Class_Programs)
        self.radioOko.setObjectName("radioOko")
        self.verticalLayout_Class_Programs.addWidget(self.radioOko)

        self.radioAntiAlergia = QtWidgets.QRadioButton(self.widget_Class_Programs)
        self.radioAntiAlergia.setObjectName("radioAntiAlergia")
        self.verticalLayout_Class_Programs.addWidget(self.radioAntiAlergia)

        self.radio20min = QtWidgets.QRadioButton(self.widget_Class_Programs)
        self.radio20min.setObjectName("radio20min")
        self.verticalLayout_Class_Programs.addWidget(self.radio20min)

        self.radioOutdoor = QtWidgets.QRadioButton(self.widget_Class_Programs)
        self.radioOutdoor.setObjectName("radioOutdoor")
        self.verticalLayout_Class_Programs.addWidget(self.radioOutdoor)

        self.radioJeans = QtWidgets.QRadioButton(self.widget_Class_Programs)
        self.radioJeans.setObjectName("radioJeans")
        self.verticalLayout_Class_Programs.addWidget(self.radioJeans)

        #compatiblity btw Programs
        self.widget_compatibility_Programs = QtWidgets.QWidget(self.frameProgram)
        self.widget_compatibility_Programs.setGeometry(QtCore.QRect(340, 80, 97, 281))
        self.widget_compatibility_Programs.setObjectName("widget_compatibility_Programs")
        self.verticalLayout_compatibility_Programs = QtWidgets.QVBoxLayout(self.widget_compatibility_Programs)
        self.verticalLayout_compatibility_Programs.setContentsMargins(15, 11, 15, 11)
        self.verticalLayout_compatibility_Programs.setSpacing(6)
        self.verticalLayout_compatibility_Programs.setObjectName("verticalLayout_compatibility_Programs")

        self.checkSameAlgod = QtWidgets.QCheckBox(self.widget_compatibility_Programs)
        self.checkSameAlgod.setText("")
        self.checkSameAlgod.setObjectName("checkSameAlgod")
        self.checkSameAlgod.setCheckable(True)
        self.verticalLayout_compatibility_Programs.addWidget(self.checkSameAlgod)

        self.checkSameSinteticos = QtWidgets.QCheckBox(self.widget_compatibility_Programs)
        self.checkSameSinteticos.setText("")
        self.checkSameSinteticos.setObjectName("checkSameSinteticos")
        self.checkSameSinteticos.setCheckable(True)
        self.verticalLayout_compatibility_Programs.addWidget(self.checkSameSinteticos)

        self.checkSameDelicados = QtWidgets.QCheckBox(self.widget_compatibility_Programs)
        self.checkSameDelicados.setText("")
        self.checkSameDelicados.setObjectName("checkSameDelicados")
        self.checkSameDelicados.setCheckable(True)
        self.verticalLayout_compatibility_Programs.addWidget(self.checkSameDelicados)

        self.checkSameLana = QtWidgets.QCheckBox(self.widget_compatibility_Programs)
        self.checkSameLana.setText("")
        self.checkSameLana.setObjectName("checkSameLana")
        self.checkSameLana.setCheckable(True)
        self.verticalLayout_compatibility_Programs.addWidget(self.checkSameLana)

        self.checkSameVapor = QtWidgets.QCheckBox(self.widget_compatibility_Programs)
        self.checkSameVapor.setText("")
        self.checkSameVapor.setObjectName("checkSameVapor")
        self.checkSameVapor.setCheckable(True)
        self.verticalLayout_compatibility_Programs.addWidget(self.checkSameVapor)

        self.checkSameOko = QtWidgets.QCheckBox(self.widget_compatibility_Programs)
        self.checkSameOko.setText("")
        self.checkSameOko.setObjectName("checkSameOko")
        self.checkSameOko.setCheckable(True)
        self.verticalLayout_compatibility_Programs.addWidget(self.checkSameOko)

        self.checkSameAntiAlergia = QtWidgets.QCheckBox(self.widget_compatibility_Programs)
        self.checkSameAntiAlergia.setText("")
        self.checkSameAntiAlergia.setObjectName("checkSameAntiAlergia")
        self.checkSameAntiAlergia.setCheckable(True)
        self.verticalLayout_compatibility_Programs.addWidget(self.checkSameAntiAlergia)

        self.checkSame20min = QtWidgets.QCheckBox(self.widget_compatibility_Programs)
        self.checkSame20min.setText("")
        self.checkSame20min.setObjectName("checkSame20min")
        self.checkSame20min.setCheckable(True)
        self.verticalLayout_compatibility_Programs.addWidget(self.checkSame20min)

        self.checkSameOutdoor = QtWidgets.QCheckBox(self.widget_compatibility_Programs)
        self.checkSameOutdoor.setText("")
        self.checkSameOutdoor.setObjectName("checkSameOutdoor")
        self.checkSameOutdoor.setCheckable(True)
        self.verticalLayout_compatibility_Programs.addWidget(self.checkSameOutdoor)

        self.checkSameJeans = QtWidgets.QCheckBox(self.widget_compatibility_Programs)
        self.checkSameJeans.setText("")
        self.checkSameJeans.setObjectName("checkSameJeans")
        self.checkSameJeans.setCheckable(True)
        self.verticalLayout_compatibility_Programs.addWidget(self.checkSameJeans)

        #---------------Options-------------------------------
        self.frameOption = QtWidgets.QFrame(self.centralWidget)
        self.frameOption.setGeometry(QtCore.QRect(0, 570, 411, 131))
        self.frameOption.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frameOption.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameOption.setObjectName("frameOption")
        self.labelOptions = QtWidgets.QLabel(self.frameOption)
        self.labelOptions.setGeometry(QtCore.QRect(20, 10, 67, 17))
        self.labelOptions.setObjectName("labelOptions")
        self.layoutWidget_4 = QtWidgets.QWidget(self.frameOption)
        self.layoutWidget_4.setGeometry(QtCore.QRect(180, 40, 141, 80))
        self.layoutWidget_4.setObjectName("layoutWidget_4")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.layoutWidget_4)
        self.verticalLayout_7.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_7.setSpacing(6)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.radioAclarado_2 = QtWidgets.QRadioButton(self.layoutWidget_4)
        self.radioAclarado_2.setObjectName("radioAclarado_2")
        self.verticalLayout_7.addWidget(self.radioAclarado_2)
        self.radioCentrif_2 = QtWidgets.QRadioButton(self.layoutWidget_4)
        self.radioCentrif_2.setObjectName("radioCentrif_2")
        self.verticalLayout_7.addWidget(self.radioCentrif_2)
        self.radioDrenar_2 = QtWidgets.QRadioButton(self.layoutWidget_4)
        self.radioDrenar_2.setObjectName("radioDrenar_2")
        self.verticalLayout_7.addWidget(self.radioDrenar_2)
        self.widget3 = QtWidgets.QWidget(self.frameOption)
        self.widget3.setGeometry(QtCore.QRect(10, 40, 141, 80))
        self.widget3.setObjectName("widget3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget3)
        self.verticalLayout_6.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_6.setSpacing(6)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.radioAclarado = QtWidgets.QRadioButton(self.widget3)
        self.radioAclarado.setObjectName("radioAclarado")
        self.verticalLayout_6.addWidget(self.radioAclarado)
        self.radioCentrif = QtWidgets.QRadioButton(self.widget3)
        self.radioCentrif.setObjectName("radioCentrif")
        self.verticalLayout_6.addWidget(self.radioCentrif)
        self.radioDrenar = QtWidgets.QRadioButton(self.widget3)
        self.radioDrenar.setObjectName("radioDrenar")
        self.verticalLayout_6.addWidget(self.radioDrenar)
        self.widget4 = QtWidgets.QWidget(self.frameOption)
        self.widget4.setGeometry(QtCore.QRect(340, 40, 49, 81))
        self.widget4.setObjectName("widget4")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.widget4)
        self.verticalLayout_15.setContentsMargins(15, 11, 15, 11)
        self.verticalLayout_15.setSpacing(6)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.checkSameAclarado = QtWidgets.QCheckBox(self.widget4)
        self.checkSameAclarado.setText("")
        self.checkSameAclarado.setObjectName("checkSameAclarado")
        self.verticalLayout_15.addWidget(self.checkSameAclarado)
        self.checkSameCentrif = QtWidgets.QCheckBox(self.widget4)
        self.checkSameCentrif.setText("")
        self.checkSameCentrif.setObjectName("checkSameCentrif")
        self.verticalLayout_15.addWidget(self.checkSameCentrif)
        self.checkSameDrenar = QtWidgets.QCheckBox(self.widget4)
        self.checkSameDrenar.setText("")
        self.checkSameDrenar.setObjectName("checkSameDrenar")
        self.verticalLayout_15.addWidget(self.checkSameDrenar)

        #-------------------STATUS-----------------------------#
        self.frameStatus = QtWidgets.QFrame(self.centralWidget)
        self.frameStatus.setGeometry(QtCore.QRect(570, 650, 541, 131))
        self.frameStatus.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frameStatus.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameStatus.setObjectName("frameStatus")
        self.widget5 = QtWidgets.QWidget(self.frameStatus)
        self.widget5.setGeometry(QtCore.QRect(20, 10, 127, 108))
        self.widget5.setObjectName("widget5")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.widget5)
        self.verticalLayout_9.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_9.setSpacing(6)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.checkboxOnOff = QtWidgets.QCheckBox(self.widget5)
        self.checkboxOnOff.setObjectName("checkboxOnOff")
        self.verticalLayout_9.addWidget(self.checkboxOnOff)
        self.checkboxVaporPlus = QtWidgets.QCheckBox(self.widget5)
        self.checkboxVaporPlus.setObjectName("checkboxVaporPlus")
        self.verticalLayout_9.addWidget(self.checkboxVaporPlus)
        self.checkboxInicioDiferido = QtWidgets.QCheckBox(self.widget5)
        self.checkboxInicioDiferido.setObjectName("checkboxInicioDiferido")
        self.verticalLayout_9.addWidget(self.checkboxInicioDiferido)
        self.checkboxInicioPause = QtWidgets.QCheckBox(self.widget5)
        self.checkboxInicioPause.setObjectName("checkboxInicioPause")
        self.verticalLayout_9.addWidget(self.checkboxInicioPause)

        #########################SYMBOLS############################
        self.frameSymbols = QtWidgets.QFrame(self.centralWidget)
        self.frameSymbols.setGeometry(QtCore.QRect(570, 10, 541, 631))
        self.frameSymbols.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frameSymbols.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameSymbols.setObjectName("frameSymbols")
        #ROS Message Symbols
        self.layoutWidget_5 = QtWidgets.QWidget(self.frameSymbols)
        self.layoutWidget_5.setGeometry(QtCore.QRect(190, 40, 156, 584))
        self.layoutWidget_5.setObjectName("layoutWidget_5")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.layoutWidget_5)
        self.verticalLayout_10.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_10.setSpacing(6)
        self.verticalLayout_10.setObjectName("verticalLayout_10")

        self.chkbxNormalTM2L7_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxNormalTM2L7_2.setObjectName("chkbxNormalTM2L7_2")
        self.verticalLayout_10.addWidget(self.chkbxNormalTM2L7_2)

        self.chkbxLowSteam_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxLowSteam_2.setObjectName("chkbxLowSteam_2")
        self.verticalLayout_10.addWidget(self.chkbxLowSteam_2)

        self.chkbxMediumSteam_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxMediumSteam_2.setObjectName("chkbxMediumSteam_2")
        self.verticalLayout_10.addWidget(self.chkbxMediumSteam_2)

        self.chkbxDailyTM2L7_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxDailyTM2L7_2.setObjectName("chkbxDailyTM2L7_2")
        self.verticalLayout_10.addWidget(self.chkbxDailyTM2L7_2)
        self.chkbxPrewashing_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxPrewashing_2.setObjectName("chkbxPrewashing_2")
        self.verticalLayout_10.addWidget(self.chkbxPrewashing_2)
        self.chkbxSuperQuickTM2L7_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxSuperQuickTM2L7_2.setObjectName("chkbxSuperQuickTM2L7_2")
        self.verticalLayout_10.addWidget(self.chkbxSuperQuickTM2L7_2)
        self.chkbxWashing_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxWashing_2.setObjectName("chkbxWashing_2")
        self.verticalLayout_10.addWidget(self.chkbxWashing_2)
        self.chkbxEconomy_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxEconomy_2.setObjectName("chkbxEconomy_2")
        self.verticalLayout_10.addWidget(self.chkbxEconomy_2)
        self.chkbxRinsing_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxRinsing_2.setObjectName("chkbxRinsing_2")
        self.verticalLayout_10.addWidget(self.chkbxRinsing_2)
        self.chkbxPrewash_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxPrewash_2.setObjectName("chkbxPrewash_2")
        self.verticalLayout_10.addWidget(self.chkbxPrewash_2)
        self.chkbxSpinning_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxSpinning_2.setObjectName("chkbxSpinning_2")
        self.verticalLayout_10.addWidget(self.chkbxSpinning_2)
        self.chkbxStain_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxStain_2.setObjectName("chkbxStain_2")
        self.verticalLayout_10.addWidget(self.chkbxStain_2)
        self.chkbxExtraRinse_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxExtraRinse_2.setObjectName("chkbxExtraRinse_2")
        self.verticalLayout_10.addWidget(self.chkbxExtraRinse_2)
        self.chkbxDrying_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxDrying_2.setObjectName("chkbxDrying_2")
        self.verticalLayout_10.addWidget(self.chkbxDrying_2)
        self.chkbxSoft_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxSoft_2.setObjectName("chkbxSoft_2")
        self.verticalLayout_10.addWidget(self.chkbxSoft_2)
        self.chkbxDelay_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxDelay_2.setObjectName("chkbxDelay_2")
        self.verticalLayout_10.addWidget(self.chkbxDelay_2)
        self.chkbxSteaming_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxSteaming_2.setObjectName("chkbxSteaming_2")
        self.verticalLayout_10.addWidget(self.chkbxSteaming_2)
        self.chkbxNightCycle_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxNightCycle_2.setObjectName("chkbxNightCycle_2")
        self.verticalLayout_10.addWidget(self.chkbxNightCycle_2)
        self.chkbxRinseHold_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxRinseHold_2.setObjectName("chkbxRinseHold_2")
        self.verticalLayout_10.addWidget(self.chkbxRinseHold_2)
        self.chkbxAnticrease_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxAnticrease_2.setObjectName("chkbxAnticrease_2")
        self.verticalLayout_10.addWidget(self.chkbxAnticrease_2)
        self.chkbxSteamAnticrease_2 = QtWidgets.QCheckBox(self.layoutWidget_5)
        self.chkbxSteamAnticrease_2.setObjectName("chkbxSteamAnticrease_2")
        self.verticalLayout_10.addWidget(self.chkbxSteamAnticrease_2)
        #Coresspondency btw symbols
        self.layoutWidget_8 = QtWidgets.QWidget(self.frameSymbols)
        self.layoutWidget_8.setGeometry(QtCore.QRect(360, 40, 49, 591))
        self.layoutWidget_8.setObjectName("layoutWidget_8")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.layoutWidget_8)
        self.verticalLayout_13.setContentsMargins(15, 11, 15, 11)
        self.verticalLayout_13.setSpacing(6)
        self.verticalLayout_13.setObjectName("verticalLayout_13")

        self.chkbxNormalTM2L7_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxNormalTM2L7_5.setText("")
        self.chkbxNormalTM2L7_5.setObjectName("chkbxNormalTM2L7_5")
        self.verticalLayout_13.addWidget(self.chkbxNormalTM2L7_5)

        self.chkbxLowSteam_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxLowSteam_5.setText("")
        self.chkbxLowSteam_5.setObjectName("chkbxLowSteam_5")
        self.verticalLayout_13.addWidget(self.chkbxLowSteam_5)

        self.chkbxMediumSteam_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxMediumSteam_5.setText("")
        self.chkbxMediumSteam_5.setObjectName("chkbxMediumSteam_5")
        self.verticalLayout_13.addWidget(self.chkbxMediumSteam_5)

        self.chkbxDailyTM2L7_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxDailyTM2L7_5.setText("")
        self.chkbxDailyTM2L7_5.setObjectName("chkbxDailyTM2L7_5")
        self.verticalLayout_13.addWidget(self.chkbxDailyTM2L7_5)

        self.chkbxPrewashing_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxPrewashing_5.setText("")
        self.chkbxPrewashing_5.setObjectName("chkbxPrewashing_5")
        self.verticalLayout_13.addWidget(self.chkbxPrewashing_5)

        self.chkbxSuperQuickTM2L7_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxSuperQuickTM2L7_5.setText("")
        self.chkbxSuperQuickTM2L7_5.setObjectName("chkbxSuperQuickTM2L7_5")
        self.verticalLayout_13.addWidget(self.chkbxSuperQuickTM2L7_5)

        self.chkbxWashing_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxWashing_5.setText("")
        self.chkbxWashing_5.setObjectName("chkbxWashing_5")
        self.verticalLayout_13.addWidget(self.chkbxWashing_5)

        self.chkbxEconomy_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxEconomy_5.setText("")
        self.chkbxEconomy_5.setObjectName("chkbxEconomy_5")
        self.verticalLayout_13.addWidget(self.chkbxEconomy_5)

        self.chkbxRinsing_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxRinsing_5.setText("")
        self.chkbxRinsing_5.setObjectName("chkbxRinsing_5")
        self.verticalLayout_13.addWidget(self.chkbxRinsing_5)

        self.chkbxPrewash_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxPrewash_5.setText("")
        self.chkbxPrewash_5.setObjectName("chkbxPrewash_5")
        self.verticalLayout_13.addWidget(self.chkbxPrewash_5)

        self.chkbxSpinning_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxSpinning_5.setText("")
        self.chkbxSpinning_5.setObjectName("chkbxSpinning_5")
        self.verticalLayout_13.addWidget(self.chkbxSpinning_5)

        self.chkbxStain_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxStain_5.setText("")
        self.chkbxStain_5.setObjectName("chkbxStain_5")
        self.verticalLayout_13.addWidget(self.chkbxStain_5)

        self.chkbxExtraRinse_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxExtraRinse_5.setText("")
        self.chkbxExtraRinse_5.setObjectName("chkbxExtraRinse_5")
        self.verticalLayout_13.addWidget(self.chkbxExtraRinse_5)

        self.chkbxDrying_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxDrying_5.setText("")
        self.chkbxDrying_5.setObjectName("chkbxDrying_5")
        self.verticalLayout_13.addWidget(self.chkbxDrying_5)

        self.chkbxSoft_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxSoft_5.setText("")
        self.chkbxSoft_5.setObjectName("chkbxSoft_5")
        self.verticalLayout_13.addWidget(self.chkbxSoft_5)

        self.chkbxDelay_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxDelay_5.setText("")
        self.chkbxDelay_5.setObjectName("chkbxDelay_5")
        self.verticalLayout_13.addWidget(self.chkbxDelay_5)

        self.chkbxSteaming_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxSteaming_5.setText("")
        self.chkbxSteaming_5.setObjectName("chkbxSteaming_5")
        self.verticalLayout_13.addWidget(self.chkbxSteaming_5)

        self.chkbxNightCycle_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxNightCycle_5.setText("")
        self.chkbxNightCycle_5.setObjectName("chkbxNightCycle_5")
        self.verticalLayout_13.addWidget(self.chkbxNightCycle_5)

        self.chkbxRinseHold_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxRinseHold_5.setText("")
        self.chkbxRinseHold_5.setObjectName("chkbxRinseHold_5")
        self.verticalLayout_13.addWidget(self.chkbxRinseHold_5)

        self.chkbxAnticrease_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxAnticrease_5.setText("")
        self.chkbxAnticrease_5.setObjectName("chkbxAnticrease_5")
        self.verticalLayout_13.addWidget(self.chkbxAnticrease_5)

        self.chkbxSteamAnticrease_5 = QtWidgets.QCheckBox(self.layoutWidget_8)
        self.chkbxSteamAnticrease_5.setText("")
        self.chkbxSteamAnticrease_5.setObjectName("chkbxSteamAnticrease_5")
        self.verticalLayout_13.addWidget(self.chkbxSteamAnticrease_5)
        #Titles of Symbol Section
        self.labelClassification_3 = QtWidgets.QLabel(self.frameSymbols)
        self.labelClassification_3.setGeometry(QtCore.QRect(20, 10, 151, 17))
        self.labelClassification_3.setAlignment(QtCore.Qt.AlignCenter)
        self.labelClassification_3.setObjectName("labelClassification_3")
        self.labelNetwork_3 = QtWidgets.QLabel(self.frameSymbols)
        self.labelNetwork_3.setGeometry(QtCore.QRect(190, 10, 151, 17))
        self.labelNetwork_3.setAlignment(QtCore.Qt.AlignCenter)
        self.labelNetwork_3.setObjectName("labelNetwork_3")
        #Classification Symbols
        self.widget6 = QtWidgets.QWidget(self.frameSymbols)
        self.widget6.setGeometry(QtCore.QRect(20, 40, 156, 584))
        self.widget6.setObjectName("widget6")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.widget6)
        self.verticalLayout_8.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_8.setSpacing(6)
        self.verticalLayout_8.setObjectName("verticalLayout_8")

        self.chkbxNormalTM2L7 = QtWidgets.QCheckBox(self.widget6)
        self.chkbxNormalTM2L7.setObjectName("chkbxNormalTM2L7")
        self.verticalLayout_8.addWidget(self.chkbxNormalTM2L7)

        self.chkbxLowSteam = QtWidgets.QCheckBox(self.widget6)
        self.chkbxLowSteam.setObjectName("chkbxLowSteam")
        self.verticalLayout_8.addWidget(self.chkbxLowSteam)

        self.chkbxMediumSteam = QtWidgets.QCheckBox(self.widget6)
        self.chkbxMediumSteam.setObjectName("chkbxMediumSteam")
        self.verticalLayout_8.addWidget(self.chkbxMediumSteam)

        self.chkbxDailyTM2L7 = QtWidgets.QCheckBox(self.widget6)
        self.chkbxDailyTM2L7.setObjectName("chkbxDailyTM2L7")
        self.verticalLayout_8.addWidget(self.chkbxDailyTM2L7)

        self.chkbxPrewashing = QtWidgets.QCheckBox(self.widget6)
        self.chkbxPrewashing.setObjectName("chkbxPrewashing")
        self.verticalLayout_8.addWidget(self.chkbxPrewashing)

        self.chkbxSuperQuickTM2L7 = QtWidgets.QCheckBox(self.widget6)
        self.chkbxSuperQuickTM2L7.setObjectName("chkbxSuperQuickTM2L7")
        self.verticalLayout_8.addWidget(self.chkbxSuperQuickTM2L7)

        self.chkbxWashing = QtWidgets.QCheckBox(self.widget6)
        self.chkbxWashing.setObjectName("chkbxWashing")
        self.verticalLayout_8.addWidget(self.chkbxWashing)

        self.chkbxEconomy = QtWidgets.QCheckBox(self.widget6)
        self.chkbxEconomy.setObjectName("chkbxEconomy")
        self.verticalLayout_8.addWidget(self.chkbxEconomy)

        self.chkbxRinsing = QtWidgets.QCheckBox(self.widget6)
        self.chkbxRinsing.setObjectName("chkbxRinsing")
        self.verticalLayout_8.addWidget(self.chkbxRinsing)

        self.chkbxPrewash = QtWidgets.QCheckBox(self.widget6)
        self.chkbxPrewash.setObjectName("chkbxPrewash")
        self.verticalLayout_8.addWidget(self.chkbxPrewash)

        self.chkbxSpinning = QtWidgets.QCheckBox(self.widget6)
        self.chkbxSpinning.setObjectName("chkbxSpinning")
        self.verticalLayout_8.addWidget(self.chkbxSpinning)

        self.chkbxStain = QtWidgets.QCheckBox(self.widget6)
        self.chkbxStain.setObjectName("chkbxStain")
        self.verticalLayout_8.addWidget(self.chkbxStain)

        self.chkbxExtraRinse = QtWidgets.QCheckBox(self.widget6)
        self.chkbxExtraRinse.setObjectName("chkbxExtraRinse")
        self.verticalLayout_8.addWidget(self.chkbxExtraRinse)

        self.chkbxDrying = QtWidgets.QCheckBox(self.widget6)
        self.chkbxDrying.setObjectName("chkbxDrying")
        self.verticalLayout_8.addWidget(self.chkbxDrying)

        self.chkbxSoft = QtWidgets.QCheckBox(self.widget6)
        self.chkbxSoft.setObjectName("chkbxSoft")
        self.verticalLayout_8.addWidget(self.chkbxSoft)

        self.chkbxDelay = QtWidgets.QCheckBox(self.widget6)
        self.chkbxDelay.setObjectName("chkbxDelay")
        self.verticalLayout_8.addWidget(self.chkbxDelay)

        self.chkbxSteaming = QtWidgets.QCheckBox(self.widget6)
        self.chkbxSteaming.setObjectName("chkbxSteaming")
        self.verticalLayout_8.addWidget(self.chkbxSteaming)

        self.chkbxNightCycle = QtWidgets.QCheckBox(self.widget6)
        self.chkbxNightCycle.setObjectName("chkbxNightCycle")
        self.verticalLayout_8.addWidget(self.chkbxNightCycle)

        self.chkbxRinseHold = QtWidgets.QCheckBox(self.widget6)
        self.chkbxRinseHold.setObjectName("chkbxRinseHold")
        self.verticalLayout_8.addWidget(self.chkbxRinseHold)

        self.chkbxAnticrease = QtWidgets.QCheckBox(self.widget6)
        self.chkbxAnticrease.setObjectName("chkbxAnticrease")
        self.verticalLayout_8.addWidget(self.chkbxAnticrease)

        self.chkbxSteamAnticrease = QtWidgets.QCheckBox(self.widget6)
        self.chkbxSteamAnticrease.setObjectName("chkbxSteamAnticrease")
        self.verticalLayout_8.addWidget(self.chkbxSteamAnticrease)


        self.frameProgram.raise_()
        self.frmScreen.raise_()
        self.btnRun.raise_()
        self.frameOption.raise_()
        self.frameStatus.raise_()
        self.frameSymbols.raise_()
        MainWindow.setCentralWidget(self.centralWidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        self.btnRun.clicked.connect(self.showInfo)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    #information receiver function
    def showInfo(self):
        if not rospy.is_shutdown():
            coordinate_sub_digits = rospy.Subscriber("user_interface_info", screen, self.callbackUserInterface)
            coordinate_symbol_program_option = rospy.Subscriber("symbol_program_option_info",screen,self.callbackSymbolProgramOption)
            wash_mash_ros_msg = rospy.Subscriber("wm_state",WmState,self.callbackRosMsg)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Washing Machine User Interface"))
        self.labelTemp.setText(_translate("MainWindow", "Temperature"))
        self.labelKg.setText(_translate("MainWindow", "Kilogram"))
        self.labelSpinSpeed.setText(_translate("MainWindow", "Centrifugation"))
        self.labelHour.setText(_translate("MainWindow", "Hour"))
        self.labelSpeed.setText(_translate("MainWindow", "Speed"))
        self.labelClassification.setText(_translate("MainWindow", "Classification"))
        self.labelNetwork.setText(_translate("MainWindow", "ROS Message "))
        self.btnRun.setText(_translate("MainWindow", "Run"))
        self.radioAlgod_2.setText(_translate("MainWindow", "Algod"))
        self.radioSinteticos_2.setText(_translate("MainWindow", "Sinteticos"))
        self.radioDelicados_2.setText(_translate("MainWindow", "Delicados"))
        self.radioLana_2.setText(_translate("MainWindow", "Lana/Las + Seda"))
        self.radioVapor_2.setText(_translate("MainWindow", "Vapor"))
        self.radioOko_2.setText(_translate("MainWindow", "OKOPower"))
        self.radioAntiAlergia_2.setText(_translate("MainWindow", "Anti-Alergia"))
        self.radio20min_2.setText(_translate("MainWindow", "20min - 3kg"))
        self.radioOutdoor_2.setText(_translate("MainWindow", "Outdoor"))
        self.radioJeans_2.setText(_translate("MainWindow", "Jeans"))
        self.lblProgram.setText(_translate("MainWindow", "Program"))
        self.labelClassification_2.setText(_translate("MainWindow", "Classification"))
        self.labelNetwork_2.setText(_translate("MainWindow", "ROS Message "))
        self.radioAlgod.setText(_translate("MainWindow", "Algod"))
        self.radioSinteticos.setText(_translate("MainWindow", "Sinteticos"))
        self.radioDelicados.setText(_translate("MainWindow", "Delicados"))
        self.radioLana.setText(_translate("MainWindow", "Lana/Las + Seda"))
        self.radioVapor.setText(_translate("MainWindow", "Vapor"))
        self.radioOko.setText(_translate("MainWindow", "OKOPower"))
        self.radioAntiAlergia.setText(_translate("MainWindow", "Anti-Alergia"))
        self.radio20min.setText(_translate("MainWindow", "20min - 3kg"))
        self.radioOutdoor.setText(_translate("MainWindow", "Outdoor"))
        self.radioJeans.setText(_translate("MainWindow", "Jeans"))
        self.labelOptions.setText(_translate("MainWindow", "Options"))
        self.radioAclarado_2.setText(_translate("MainWindow", "Aclarado/Enxag"))
        self.radioCentrif_2.setText(_translate("MainWindow", "Centrif"))
        self.radioDrenar_2.setText(_translate("MainWindow", "Drenar"))
        self.radioAclarado.setText(_translate("MainWindow", "Aclarado/Enxag"))
        self.radioCentrif.setText(_translate("MainWindow", "Centrif"))
        self.radioDrenar.setText(_translate("MainWindow", "Drenar"))
        self.checkboxOnOff.setText(_translate("MainWindow", "On/Off"))
        self.checkboxVaporPlus.setText(_translate("MainWindow", "Vapor Plus"))
        self.checkboxInicioDiferido.setText(_translate("MainWindow", "Inicio Diferido"))
        self.checkboxInicioPause.setText(_translate("MainWindow", "Inicio/Pause"))
        self.chkbxNormalTM2L7_2.setText(_translate("MainWindow", "NormalTM2L7"))
        self.chkbxLowSteam_2.setText(_translate("MainWindow", "Low_Steam"))
        self.chkbxMediumSteam_2.setText(_translate("MainWindow", "Medium_Steam"))
        self.chkbxDailyTM2L7_2.setText(_translate("MainWindow", "DailyTM2L7"))
        self.chkbxPrewashing_2.setText(_translate("MainWindow", "Prewashing"))
        self.chkbxSuperQuickTM2L7_2.setText(_translate("MainWindow", "SuperQuickTM2L7"))
        self.chkbxWashing_2.setText(_translate("MainWindow", "Washing"))
        self.chkbxEconomy_2.setText(_translate("MainWindow", "Economy"))
        self.chkbxRinsing_2.setText(_translate("MainWindow", "Rinsing"))
        self.chkbxPrewash_2.setText(_translate("MainWindow", "Prewash"))
        self.chkbxSpinning_2.setText(_translate("MainWindow", "Spinning"))
        self.chkbxStain_2.setText(_translate("MainWindow", "Stain"))
        self.chkbxExtraRinse_2.setText(_translate("MainWindow", "ExtraRinse"))
        self.chkbxDrying_2.setText(_translate("MainWindow", "Drying"))
        self.chkbxSoft_2.setText(_translate("MainWindow", "Soft"))
        self.chkbxDelay_2.setText(_translate("MainWindow", "Delay"))
        self.chkbxSteaming_2.setText(_translate("MainWindow", "Steaming"))
        self.chkbxNightCycle_2.setText(_translate("MainWindow", "Night_Cycle"))
        self.chkbxRinseHold_2.setText(_translate("MainWindow", "Rinse_Hold"))
        self.chkbxAnticrease_2.setText(_translate("MainWindow", "Anticrease"))
        self.chkbxSteamAnticrease_2.setText(_translate("MainWindow", "Steam_Anticrease"))
        self.labelClassification_3.setText(_translate("MainWindow", "Classification"))
        self.labelNetwork_3.setText(_translate("MainWindow", "ROS Message "))
        self.chkbxNormalTM2L7.setText(_translate("MainWindow", "NormalTM2L7"))
        self.chkbxLowSteam.setText(_translate("MainWindow", "Low_Steam"))
        self.chkbxMediumSteam.setText(_translate("MainWindow", "Medium_Steam"))
        self.chkbxDailyTM2L7.setText(_translate("MainWindow", "DailyTM2L7"))
        self.chkbxPrewashing.setText(_translate("MainWindow", "Prewashing"))
        self.chkbxSuperQuickTM2L7.setText(_translate("MainWindow", "SuperQuickTM2L7"))
        self.chkbxWashing.setText(_translate("MainWindow", "Washing"))
        self.chkbxEconomy.setText(_translate("MainWindow", "Economy"))
        self.chkbxRinsing.setText(_translate("MainWindow", "Rinsing"))
        self.chkbxPrewash.setText(_translate("MainWindow", "Prewash"))
        self.chkbxSpinning.setText(_translate("MainWindow", "Spinning"))
        self.chkbxStain.setText(_translate("MainWindow", "Stain"))
        self.chkbxExtraRinse.setText(_translate("MainWindow", "ExtraRinse"))
        self.chkbxDrying.setText(_translate("MainWindow", "Drying"))
        self.chkbxSoft.setText(_translate("MainWindow", "Soft"))
        self.chkbxDelay.setText(_translate("MainWindow", "Delay"))
        self.chkbxSteaming.setText(_translate("MainWindow", "Steaming"))
        self.chkbxNightCycle.setText(_translate("MainWindow", "Night_Cycle"))
        self.chkbxRinseHold.setText(_translate("MainWindow", "Rinse_Hold"))
        self.chkbxAnticrease.setText(_translate("MainWindow", "Anticrease"))
        self.chkbxSteamAnticrease.setText(_translate("MainWindow", "Steam_Anticrease"))


if __name__ == "__main__":
    rospy.init_node("Washing_Machine_Info_Indicator", anonymous= True)
    msg = screen() #reference for classification message
    rosmsg = WmState() #reference for ros message
    #rosmsg =
    import sys
    app = QtWidgets.QApplication(sys.argv)
    try:
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
    except:
        pass
