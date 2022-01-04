// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <unistd.h>
#include <QWindow>
#include <QGroupBox> 
#include <QHyodaX11XtermLog.h>
#include <QProcess>

// ****************************************************************************
// * QHyodaX11XtermLog
// ****************************************************************************
QHyodaX11XtermLog::QHyodaX11XtermLog(QHyodaJob *j,
                                     QWidget *parent,
                                     QString tabName):
  QWidget(parent),
  job(j),
  xterm(new QProcess()),
  xwininfo(new QProcess()),
  xdotool(new QProcess()),
  xWidget(NULL),
  xtermTitle(QString("xtermHyoda0x%1").arg(qHash(QUuid::createUuid()),0,16).toLocal8Bit().constData()),
  WinId(0),
  size(QSize())
{
  qDebug() << "\33[32m[QHyodaX11XtermLog::QHyodaX11XtermLog] NEW"<<"\33[m";
  // Mise en place de l'UI
  setupUI(parent);
  // Connexion du signal indiquant la fin de l'xterm
  connect(xterm,  SIGNAL(finished(int,QProcess::ExitStatus)),
          this,SLOT(xtermFinished(int,QProcess::ExitStatus)));
  // On prÃ©pare l'xterm
  QString executable("xterm");
  QStringList arguments=
    QStringList() << "-fg" << "green" << "-bg" << "black"
                  << "-maximized"
                  << "-title" << xtermTitle
                  << "-j" // indicates that xterm should do jump scrolling
                  << "-b" << "2" // size of the inner border, in pixels
                  << "-vb" // Visual bell is preferred over an audible one
                  << "-sb" << "-sl" << "8192"
                  << "+cm" // Enables recognition of ANSI color-change escape sequences
                  << "-bc" // Turn on text cursor blinking
                  << "+l"  // Turn logging off
                  << "+lc" // Turn off support of automatic selection of locale encodings
    //<< "-s"  // xterm may scroll asynchronously
                  << "+aw" // auto-wraparound should not be allowed
                  << "-ut" // xterm should not write a record into the the system utmp log file
                  << "-into" << QString::number(xWidget->winId())
                  << "-e";
  // On prÃ©pare le programme Ã  exÃ©cuter dans cet xterm
  QString xterm_execute;
  for(int i=1;i<job->argc;i+=1)
    xterm_execute.append(QString("%1%2").arg((i>1)?" ":"",job->argv[i]));
  xterm_execute.append(";/bin/bash");
  arguments << xterm_execute;
  qDebug() << "\33[32m[QHyodaX11XtermLog] arguments: "<<arguments<<"\33[m";
  // Lancement de l'xterm
  xterm->start(executable,arguments);
  if (!xterm->waitForStarted()) qFatal("QHyodaX11XtermLog::refresh NOT started!");
  // Lancement du xwininfo
  xwininfo->setProcessChannelMode(QProcess::MergedChannels);
  connect(xwininfo,SIGNAL(readyReadStandardOutput()),this,SLOT(xwininfoOutput()));
  xwininfoStart();
}

// ****************************************************************************
// * setupUI
// ****************************************************************************
void QHyodaX11XtermLog::setupUI(QWidget *parent){
  QGroupBox *groupBox = new QGroupBox(QString("Log"),parent);
  QVBoxLayout *layout = new QVBoxLayout(this);
  QVBoxLayout *vbox = new QVBoxLayout;
  vbox->addWidget(xWidget=new QWidget(this));
  groupBox->setLayout(vbox);
  layout->addWidget(groupBox);
  setLayout(layout);
}

// ****************************************************************************
// * xwininfoStart
// ****************************************************************************
void QHyodaX11XtermLog::xwininfoStart(){
  xwininfo->start(QString("xwininfo"), QStringList()<< "-name" << xtermTitle);
  if (!xwininfo->waitForStarted()) qFatal("xlsclients NOT started!");  
}

// ****************************************************************************
// * setXtermSizeWithXdotool
// ****************************************************************************
void QHyodaX11XtermLog::setXtermSizeWithXdotool(const QSize size){
  if (WinId==0) return;
  if (xdotool->state()==QProcess::Running) return;
  xdotool->start(QString("xdotool"),
                 QStringList()
                 << "windowsize"
                 << QString::number(WinId)
                 << QString::number(size.width())
                 << QString::number(size.height()));
  if (!xdotool->waitForStarted()) qFatal("xlsclients NOT started!");
}

// ****************************************************************************
// * resizeEvent
// ****************************************************************************
void QHyodaX11XtermLog::resizeEvent(QResizeEvent *e){
  setXtermSizeWithXdotool(size=e->size());
}

// ****************************************************************************
// * xtermFinished
// ****************************************************************************
void QHyodaX11XtermLog::xtermFinished(int exitCode, QProcess::ExitStatus exitStatus){
  qDebug() << "\33[32m[QHyodaX11XtermLog] xterm has finished with exitCode"
           <<exitCode<<", and exitStatus"<<exitStatus<<"\33[m";
  xterm->closeWriteChannel();
  if (xterm->waitForFinished()) qFatal("xterm NOT finished!");
  if (exitStatus==QProcess::NormalExit) job->quit();
}

// ****************************************************************************
// * xwininfoOutput
// ****************************************************************************
void QHyodaX11XtermLog::xwininfoOutput(){
  // Si on a dÃ©jÃ  rÃ©cupÃ©rÃ© l'id de la fenÃªtre, il n'y a plus rien Ã  faire
  if (WinId!=0) return;
  const QStringList lines=QString(xwininfo->readAllStandardOutput()).split(QRegExp("\\s+"));
  // On scrute le retour pour voir si l'on peut rÃ©cupÃ©rer l'id
  if (lines.size()>=4 && lines.at(4).startsWith("0x")){
    bool ok;
    WinId = lines.at(4).toLong(&ok, 16);
    qDebug()<<"[1;32m[xwininfoOutput] WinId is"
            <<QString("0x%1").arg(WinId,0,16).toLocal8Bit().constData()<<"[0m";
    if (!ok) qFatal("Could not play with WinID");
    setXtermSizeWithXdotool(size);
    return;
  }
  
  xwininfo->close();
  if (xwininfo->waitForFinished()) qFatal("xwininfo NOT closed!");
  xwininfoStart();
}
