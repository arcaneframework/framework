// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "QHyodaGdb.h" 
#include "QHyodaX11Emacs.h"
#include "QHyodaX11Xterm.h"

//#include "ui_hyoda.h"

//#include <QtWidgets>
//#include <QProcess>
//#include <QX11EmbedContainer>
//#include <QX11Info>

//#include <unistd.h>

/******************************************************************************
 * QHyodaX11Emacs
 *****************************************************************************/
QHyodaX11Emacs::QHyodaX11Emacs(QWidget *parent,
                               QString title): QHyodaX11(parent, title),
                                               serverUp(false)
{
  qDebug() << "[NEW] QHyodaX11Emacs";
}


/***********************************************
 * ~QHyodaX11Emacs
 ***********************************************/
QHyodaX11Emacs::~QHyodaX11Emacs(void){ 
  qDebug() << "~QHyodaX11Emacs";
/*
  if (xterm!=NULL){
    qDebug() << "~QHyodaX11Emacs closing xterm";
    xterm->close();
    xterm->closeWriteChannel();
    if (xterm->waitForFinished()) qFatal("xterm NOT closed!");
    delete xterm;
  }
  
  if (emacs!=NULL){
    qDebug() << "~QHyodaX11Emacs closing emacs";
    emacs->close();
    emacs->closeWriteChannel();
    if (emacs->waitForFinished()) qFatal("QProcessEmacs NOT closed!");
    delete emacs;
  }
  
  if (emacsClientProcess==NULL) return;
  qDebug() << "~QHyodaX11Emacs closing emacsClientProcess";
  emacsClientProcess->close();
  emacsClientProcess->closeWriteChannel();
  if (emacsClientProcess->waitForFinished()) qFatal("QProcessEmacs NOT closed!");
  delete emacsClientProcess;*/
}



/**********************************************************
 * embedIsEmbedded
 **********************************************************/
void QHyodaX11Emacs::clientIsEmbedded(){
  qDebug()<<"[QHyodaX11Emacs::clientIsEmbedded]";
  //#warning Work around for emacs to show correctly
  //usleep(1500ul*1000ul);
}

/**********************************************************
 * embedClosed
 **********************************************************/
void QHyodaX11Emacs::clientClosed(){
  qDebug() << "[QHyodaX11Emacs::clientClosed] Container clientClosed!";
//#warning Should remove tab
}
  
/**********************************************************
 * emacsError
 **********************************************************/
/*void QHyodaX11Emacs::error(QX11EmbedContainer::Error error){
  qDebug() << "[QHyodaX11Emacs::error] Container error:"<<error;
  }*/



/***********************************************
 * launchEmacsInWindow
 ***********************************************/
void QHyodaX11Emacs::launchEmacsInWindow(QWidget *widget, QString fileNameToOpen){
  if (!serverUp)
    launchServerInWindow(widget, fileNameToOpen);
  else
    launchClient(fileNameToOpen);
}


/**********************************************************
 * launchServerInWindow
 **********************************************************/
void QHyodaX11Emacs::launchServerInWindow(QWidget *widget, QString fileName){
  qDebug() << "[QHyodaX11Emacs::launchServerInWindow]";
  // Now create our container
  //X11Container.append(new QHyodaX11Embed(widget, this));
   // Set initial stretch factor
  splitter->setStretchFactor(0, 3);
  splitter->addWidget(this);

// Now launching server
  X11Process.append(new QProcess(widget));
  X11Process.last()->setProcessChannelMode(QProcess::MergedChannels);
  connect(X11Process.last(), SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(serverFinished(int, QProcess::ExitStatus)));
  connect(X11Process.last(), SIGNAL(readyReadStandardOutput()), this, SLOT(serverReadyReadStandardOutput()));
  
  QString emacs_path("emacs");
  QString setqEmacsName("(setq server-name \"");
  setqEmacsName.append(title);
  setqEmacsName.append("\")");

  QStringList daemonArgs=QStringList()
    //<< "-Q"
    //<< "--maximized"
    //<< "--geometry"<< "20x60"
    //<< "--geometry"<< geoString
    //<< "--internal-border" << 0
    << "--eval" << setqEmacsName
    << "--eval" << "(setq server-raise-frame nil)"
    << "--eval" << "(server-force-delete)"
    << "--eval" << "(server-start)"
    << "--eval" << "(hl-line-mode)"
    //<< "--fullwidth"
    //<< "--fullheight"
    //<< "--fullscreen"
    << "--parent-id" << QString::number(this->winId());
  if (fileName!=NULL) daemonArgs << fileName;
  qDebug()<<"daemonArgs:"<<daemonArgs;
  X11Process.last()->start(emacs_path, daemonArgs);
  if (!X11Process.last()->waitForStarted()) qFatal("emacs NOT started!");
  qDebug()<<"serverUp!";
  serverUp=true;
}

/**********************************************************
 * launchXtermInWindow
 **********************************************************/
void QHyodaX11Emacs::launchXtermInWindow(QWidget *widget){
  // Now add an xterm south
  //X11Container.append(new QHyodaX11Embed(widget,this));
  splitter->addWidget(this);
  
  X11Process.append(new QProcess(widget));
  X11Process.last()->setProcessChannelMode(QProcess::MergedChannels);
  connect(X11Process.last(), SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(xtermFinished(int, QProcess::ExitStatus)));
  connect(X11Process.last(), SIGNAL(readyReadStandardOutput()), this, SLOT(xtermReadyReadStandardOutput()));
  QString executable("/usr/bin/xterm");
  QStringList arguments=QStringList() << "-fg" << "cyan"
                                      << "-bg" << "black"
                                      << "-vb" // Visual bell is preferred over an audible one
                                      << "-sb" << "-sl" << "8192"
                                      << "-bc" // Turn on text cursor blinking
                                      << "+l"  // Turn logging off
                                      << "+lc" // Turn off support of automatic selection of locale encodings
                                      << "-s"  // This option indicates that xterm may scroll asynchronously
                                      << "+aw" // This option indicates that auto-wraparound should not be allowed
    //<< "+tb" // This option indicates that xterm should not set up a toolbar
                                      << "-ut" // This option indicates that xterm should not write a record into the the system utmp log file
                                      << "-into" << QString::number(this->winId()); 
  X11Process.last()->start(executable, arguments);
  if (!X11Process.last()->waitForStarted()) qFatal("QHyodaX11Emacs::refresh NOT started!");
  splitter->setStretchFactor(1,1);
}


/**********************************************************
 * launchClient
 **********************************************************/
void QHyodaX11Emacs::launchClient(QString fileName){
  qDebug()<<"launchClient"<<fileName;
  X11Process.append(new QProcess(this));
  X11Process.last()->setProcessChannelMode(QProcess::MergedChannels);
  connect(X11Process.last(), SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(clientFinished(int, QProcess::ExitStatus)));
  connect(X11Process.last(), SIGNAL(readyReadStandardOutput()), this, SLOT(clientReadyReadStandardOutput()));
  QString emacsclient("emacsclient");
  QStringList clientArgs=QStringList() << "-n" << "-s" << title;//emacsName;
  // Dans le cas du tab pour visualiser les 'src', on ne passe pas d'argument fileName
  if (fileName!=NULL) clientArgs << fileName;
  qDebug()<<"clientArgs:"<<clientArgs;
  X11Process.last()->start(emacsclient, clientArgs);
  if (!X11Process.last()->waitForStarted()) qFatal("QProcessEmacs NOT started!");
}




// Process slots
void QHyodaX11Emacs::serverFinished(int exitCode, QProcess::ExitStatus exitStatus){
  qDebug() << "[QHyodaX11Emacs::serverFinished] emacs finished with exitCode="<<exitCode<<", and exitStatus="<<exitStatus;
  //X11Container.at(0)->hide();
}
void QHyodaX11Emacs::serverReadyReadStandardOutput(void){
  const QString output=QString(X11Process.at(0)->readAllStandardOutput());
  qDebug()<<output;
  if (output.contains("Starting Emacs daemon")){
    serverUp=true;
    qDebug()<<"emacs has started!";
    //launchClient(fileName);
  }
}


void QHyodaX11Emacs::clientFinished(int exitCode, QProcess::ExitStatus exitStatus){
  qDebug() << "[QHyodaX11Emacs::clientFinished] with exitCode="<<exitCode<<", and exitStatus="<<exitStatus;
  if (splitter->count()==0) return;
  splitter->setStretchFactor(1,1);
}
void QHyodaX11Emacs::clientReadyReadStandardOutput(void){
  qDebug()<<X11Process.last()->readAllStandardOutput();
}


void QHyodaX11Emacs::xtermFinished(int exitCode, QProcess::ExitStatus exitStatus){
  qDebug() << "[QHyodaX11Emacs::xtermFinished] finished with exitCode="<<exitCode<<", and exitStatus="<<exitStatus;
  //xtermContainer->hide();
}
void QHyodaX11Emacs::xtermReadyReadStandardOutput(void){
  qDebug()<<X11Process.at(1)->readAllStandardOutput();
}


