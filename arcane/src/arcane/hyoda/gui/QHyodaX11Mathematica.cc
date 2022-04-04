// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
//#include <QHyoda.h>
#include <unistd.h>
#include <QHyodaX11Mathematica.h>


// ****************************************************************************
// * QHyodaX11Mathematica
// ****************************************************************************
QHyodaX11Mathematica::QHyodaX11Mathematica(QWidget *tabWidgetParent,
                                           QFileInfo fInfo):
  QHyodaX11(tabWidgetParent,fInfo.fileName()),
  fileInfo(QFileInfo(fInfo)),
  cmdMathematica(QString("xterm")),
  //cmdMathematica(QString("mathematica")),
  argsMathematica(QStringList()),
  cmdXwininfo(QString("xwininfo")),
  argsXwininfo(QStringList()),
  winId(0)
{
  //qDebug() << "[1;36m[QHyodaX11Mathematica] NEW[0m";
  argsMathematica<< "-T" << "xtermMath";
  //argsMathematica << "/tmp/110308-Laplace.nb";
  //<< "--singlelaunch" << fInfo.absoluteFilePath();
  argsXwininfo<< "-name" << "xtermMath";
  //argsXwininfo<< "-name" << fileInfo.fileName().append(" - Wolfram Mathematica 10.0");
  //argsXwininfo << "-name" << "/tmp/110308-Laplace.nb - Wolfram Mathematica 10.0";
  qDebug() << "[1;36m[QHyodaX11Mathematica] "<<cmdMathematica << argsMathematica<<"[0m";
  qDebug() << "[1;36m[QHyodaX11Mathematica] "<<cmdXwininfo << argsXwininfo<<"[0m";
  
    // Process Mathematica
  X11Process.append(new QProcess());
  X11Process.last()->setProcessChannelMode(QProcess::MergedChannels);
  connect(X11Process.last(),SIGNAL(finished(int,QProcess::ExitStatus)),this,SLOT(mathFinished(int,QProcess::ExitStatus)));
  connect(X11Process.last(),SIGNAL(readyReadStandardOutput()),this,SLOT(mathReadyReadStandardOutput()));
  X11Process.last()->start(cmdMathematica, argsMathematica);
  if (!X11Process.last()->waitForStarted())
    qFatal("[1;36mQHyodaX11Mathematica::mathematica NOT started![0m");

  // Process xwin pour récupérer l'ID de la fenêtre
  X11Process.append(new QProcess());
  X11Process.last()->setProcessChannelMode(QProcess::MergedChannels);
  connect(X11Process.last(),SIGNAL(readyReadStandardOutput()),this,SLOT(xwinReadyReadStandardOutput()));
  X11Process.last()->start(cmdXwininfo, argsXwininfo);
  if (!X11Process.last()->waitForStarted())
    qFatal("QHyodaX11Mathematica::xlsclients NOT started!");

  layout->addWidget(this);
}

// ****************************************************************************
// * ~QHyodaX11Mathematica
// * pgrep -u camierjs -P 15507 -l '.*'
// * pkill -SIGKILL 'Math.*'
// ****************************************************************************
QHyodaX11Mathematica::~QHyodaX11Mathematica(void){
  qDebug() << "[1;36m~QHyodaX11Mathematica"<<"[0m";
  qDebug()<< "[1;36m~QHyodaX11Mathematica clossing X11Process x"<<X11Process.count()<<"[0m";
  {
    QProcess pkill(this);
    QString pkill_cmd("/usr/bin/pkill");
    QStringList pkill_args=QStringList()<<"-SIGKILL" << "'Math.*'";
    qDebug()<<pkill_cmd<<pkill_args;
    pkill.start(pkill_cmd, pkill_args);
    if (!pkill.waitForStarted())
      qFatal("QHyodaX11Mathematica::mathFinished pkill not started!");
    pkill.close();
    if (pkill.waitForFinished())
      qFatal("pkill could not be closed!");
  }
  foreach(QProcess *prcs, X11Process){
    qDebug()<< "[1;36m~QHyodaX11Mathematica clossing "<<prcs<<"[0m";
    prcs->close();
    if (prcs->waitForFinished()) qFatal("A process could not be closed!");
  }
}

// ****************************************************************************
// * embedIsEmbedded
// ****************************************************************************
void QHyodaX11Mathematica::clientIsEmbedded(){
  qDebug()<<"[1;36m[QHyodaX11Mathematica::clientIsEmbedded]"<<"[0m";
}

// ****************************************************************************
// * embedClosed
// ****************************************************************************
void QHyodaX11Mathematica::clientClosed(){
  qDebug() << "[1;36m[QHyodaX11Mathematica::embedClosed] Container clientClosed!"<<"[0m";
}

// ****************************************************************************
// * error
// ****************************************************************************
/*void QHyodaX11Mathematica::error(QX11EmbedContainer::Error error){
  qDebug() << "[1;36m[QHyodaX11Mathematica::error] Container error:"<<error<<"[0m";
  }*/


// ****************************************************************************
// * launchMathematicaInWindow
// ****************************************************************************
void QHyodaX11Mathematica::launch(QWidget *widget){
  //tabParent=widget;
  qDebug() << "[1;36m[QHyodaX11Mathematica::launch] Now launching...[0m";
  //X11Container.append(new QHyodaX11Embed(widget, this));
  //splitter->addWidget(this);


  
  qDebug() << "[1;36m[QHyodaX11Mathematica::launch] Launched![0m";
  
  //qDebug() << "[1;36m[QHyodaX11Mathematica::launch] Launched, waitting for winId![0m";
  //while (winId==0) usleep(100);
  //qDebug() << "[1;36m[QHyodaX11Mathematica::launch] Got IT![0m";
}


// ****************************************************************************
// * mathFinished
// ****************************************************************************
void QHyodaX11Mathematica::mathFinished(int exitCode,
                                        QProcess::ExitStatus exitStatus){
  qDebug() << "[1;36m[QHyodaX11Mathematica::mathFinished] mathematica has finished with exitCode="<<exitCode<<", and exitStatus="<<exitStatus<<"[0m";
  // On enlève le tab s'il existe: on arrive avec le CTRL-Q alors qu'il est deleté!  
  //int idx=tabParent->indexOf(this);
  //qDebug() << "[QHyodaX11Mathematica] idx="<<idx;
  //if (idx!=-1) tabParent->removeTab(idx);
  
  // exitStatus à 0 pour le CTRL-D ou destructeur
  // exitStatus à 1 depuis la croix
/*  if (exitStatus==0){
    qDebug() << "[QHyodaX11Mathematica] From CTRL-D ou destructeur!";
    delete mathematica; mathematica=NULL;
    delete layout; layout=NULL;
    delete mathematicaContainer; mathematicaContainer=NULL;
  }
  if (exitStatus==1){
   qDebug() << "[QHyodaX11Mathematica] From X tab!";
  }
*/
}


// ****************************************************************************
// * mathReadyReadStandardOutput
// ****************************************************************************
void QHyodaX11Mathematica::mathReadyReadStandardOutput(void){
  qDebug()<<"[1;36mmathReadyReadStandardOutput[0m";
  //qDebug()<<X11Process.last()->readAllStandardOutput();
}


// ****************************************************************************
// * xwinReadyReadStandardOutput
// ****************************************************************************
void QHyodaX11Mathematica::xwinReadyReadStandardOutput(void){
  qDebug()<<"[1;36mxwinReadyReadStandardOutput"<<"[0m";
  const QStringList lines=QString(X11Process.last()->readAllStandardOutput()).split(QRegExp("\\s+"));
  //foreach(QString line, lines) qDebug()<<line;
  
  // Si on a déjà récupéré l'id de la fenêtre, il n'y a plus rien à faire
  if (winId!=0) {
    qDebug()<<"[1;36m[xwinReadyReadStandardOutput] winId!=0, returning!"<<"[0m";
    return;
  }

  // On scrute le retour pour voir si l'on peut récupérer l'id
  if (lines.at(4).startsWith("0x")){
    bool ok;
    winId = lines.at(4).toLong(&ok, 16);
    qDebug()<<"[1;36m[xwinReadyReadStandardOutput] winId="<<winId<<"[0m";
    if (!ok) qFatal("Could not play with WinID");
    
    QWindow *math_win=QWindow::fromWinId(winId);
    QWidget *container=QWidget::createWindowContainer(math_win);//,this,Qt::FramelessWindowHint);
    //math_win->setParent(tabParent->windowHandle());
    //math_win->setFlags(Qt::FramelessWindowHint);
    layout->addWidget(container);
    //show();
    //math_win->setParent(parentWidget->windowHandle());
   //layout->addWidget(container);
    return;
  }

  // Sinon, on ferme pour recommencer
  X11Process.last()->close();
  if (X11Process.last()->waitForFinished()) qFatal("xwininfo NOT closed!");
  usleep(100ul);
  X11Process.last()->start(cmdXwininfo, argsXwininfo);
  if (!X11Process.last()->waitForStarted())
    qFatal("QHyodaX11Mathematica::xwininfo NOT started!");
}

