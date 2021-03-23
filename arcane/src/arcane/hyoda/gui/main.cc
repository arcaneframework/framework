// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <QFileInfo>
#include <QApplication>
#include <QtNetwork/QHostInfo>

#include <QHyoda.h>
#include <QOpenGLFunctions>

// ****************************************************************************
// * main: on informe en premier la partie Hyoda d'Arcane
// *       afin qu'elle se connecte dès le début
// *       Ceci se fait en settant les variables ARCANE_HYODA,
// *       ARCANE_HYODA_ADRS & ARCANE_HYODA_PORT
// ****************************************************************************
int main(int argc, char *argv[]){
  QApplication qt(argc,argv);
  QString localHostName=QHostInfo::localHostName();
  qputenv("ARCANE_HYODA", QByteArray());
  qputenv("ARCANE_HYODA_HOST",localHostName.toLocal8Bit());
  QHostInfo info=QHostInfo::fromName(localHostName);
  if (!info.addresses().isEmpty()){
    QString adrs=QString("%1").arg(info.addresses().first().toIPv4Address());
    //qDebug() << "[main] ARCANE_HYODA_ADRS:"<<adrs;
    qputenv("ARCANE_HYODA_ADRS", adrs.toLocal8Bit());
  }else qFatal("Empty list of IP addresses associated with hostName()");
  //qDebug() << "[main] ARCANE_HYODA_PORT:"<<3889;
  qputenv("ARCANE_HYODA_PORT", QString("%1").arg(3889).toLocal8Bit());
      
  QHyoda hyoda(argc,argv);
  return qt.exec();
  
}
