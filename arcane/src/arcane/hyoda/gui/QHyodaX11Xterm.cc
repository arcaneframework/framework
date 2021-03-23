// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "QHyodaX11Xterm.h"

/******************************************************************************
 * QHyodaX11Xterm
 *****************************************************************************/
QHyodaX11Xterm::QHyodaX11Xterm(QWidget *widget, QString tabName):QHyodaX11(widget,tabName)
{
  qDebug() << "[NEW] QHyodaX11Xterm";
}


/***********************************************
 * ~QHyodaX11Xterm
 ***********************************************/
QHyodaX11Xterm::~QHyodaX11Xterm(void){ 
/*  qDebug() << "~QHyodaX11Xterm";
  if (xterm!=NULL){
    if (xterm->state()==QProcess::Running) {
      qDebug() << "~QHyodaX11Xterm xterm";
      xterm->close(); // Cela appelle notre finished
      //delete xterm;
    }
  }
  if (xtermContainer!=NULL){
    qDebug() << "~QHyodaX11Xterm xtermContainer";
    //delete xtermContainer;
    }*/
  //if(layout!=NULL) delete layout;
}


/**********************************************************
 * embedIsEmbedded
 **********************************************************/
void QHyodaX11Xterm::clientIsEmbedded(){
  qDebug()<<"[QHyodaX11Xterm::clientIsEmbedded]";
}

/**********************************************************
 * embedClosed
 **********************************************************/
void QHyodaX11Xterm::clientClosed(){
  qDebug() << "[QHyodaX11Xterm::embedClosed] Container clientClosed!";
}
  
/**********************************************************
 * xtermError
 **********************************************************/
/*void QHyodaX11Xterm::error(QX11EmbedContainer::Error error){
  qDebug() << "[QHyodaX11Xterm::error]"<<error;
  }*/


bool QHyodaX11Xterm::close()
{
  qDebug() << "[QHyodaX11Xterm::close]";
  return true;
}


void QHyodaX11Xterm::started(){
  qDebug() << "[QHyodaX11Xterm::started]";
}


/**********************************************************
 * launchXtermInWindow
 **********************************************************/
void QHyodaX11Xterm::launch(QWidget *widget){  
  //X11Container.append(new QHyodaX11Embed(widget, this));
  splitter->addWidget(this);
  
  X11Process.append(new QProcess(widget));
  X11Process.last()->setProcessChannelMode(QProcess::MergedChannels);
  //connect(xterm, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(finished(int, QProcess::ExitStatus)));
  //connect(xterm, SIGNAL(readyReadStandardOutput()), this, SLOT(readyReadStandardOutput()));
  //connect(xterm, SIGNAL(stateChanged(QProcess::ProcessState)), this, SLOT(stateChanged(QProcess::ProcessState)));
  //connect(xterm, SIGNAL(started()), this, SLOT(started()));
  QString executable("/usr/bin/xterm");
  QStringList arguments=QStringList() << "-fg" << "cyan"
                                      << "-bg" << "black"
                                      << "-vb" // Visual bell is preferred over an audible one
                                      << "-sb" << "-sl" << "8192"
                                      << "-bc" // Turn on text cursor blinking
                                      << "+l"  // Turn logging off
                                      << "+lc" // Turn off support of automatic selection of locale encodings
                                      << "-s"  // xterm may scroll asynchronously
                                      << "+aw" // auto-wraparound should not be allowed
                                      << "-ut" // Do not write a record into the the system utmp log file
                                      << "-into" << QString::number(this->winId()); 
  X11Process.last()->start(executable, arguments);
  if (!X11Process.last()->waitForStarted()) qFatal("QHyodaX11Xterm::refresh NOT started!");
}


void QHyodaX11Xterm::stateChanged(QProcess::ProcessState newState){
  qDebug() << "[QHyodaX11Xterm::stateChanged]"<<newState;
}


void QHyodaX11Xterm::finished(int exitCode, QProcess::ExitStatus exitStatus){
  qDebug() << "[QHyodaX11Xterm] xterm has finished with exitCode="<<exitCode<<", and exitStatus="<<exitStatus;
  // Flush des channels
  if (X11Process.last()!=NULL){
    X11Process.last()->closeWriteChannel();
    if (X11Process.last()->waitForFinished()) qFatal("xterm NOT closed!");
  }
//  xtermContainer->hide();
  
  // On enlève le tab s'il existe: on arrive avec le CTRL-Q alors qu'il est deleté!  
  //int idx=tabParent->indexOf(this);
  //qDebug() << "[QHyodaX11Xterm] idx="<<idx;
  //if (idx!=-1) tabParent->removeTab(idx);
  
  // exitStatus à 0 pour le CTRL-D ou destructeur
  // exitStatus à 1 depuis la croix
/*  if (exitStatus==0){
    qDebug() << "[QHyodaX11Xterm] From CTRL-D ou destructeur!";
    delete xterm; xterm=NULL;
    delete layout; layout=NULL;
    delete xtermContainer; xtermContainer=NULL;
  }
  if (exitStatus==1){
   qDebug() << "[QHyodaX11Xterm] From X tab!";
  }
*/
}

void QHyodaX11Xterm::readyReadStandardOutput(void){
  qDebug()<<X11Process.last()->readAllStandardOutput();
}


