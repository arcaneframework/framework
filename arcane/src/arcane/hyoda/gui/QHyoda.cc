// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/**
 * /usr/local/qt/5.7/gcc_64/bin/assistant
 *
 * /usr/local/qt/5.7/gcc_64/bin/designer
 *
 * xpdf gdb.pdf
 *
 * ARCANE_HYODA_MESH_PPM
 * ARCANE_HYODA_MATRIX_PPM
 *
 * tput reset && rm -f /tmp/*.ppm && pkill gdbserver && pkill mpiexec && pkill mono && pkill ctest; ARCANE_HYODA_MATRIX_RENDER=1 m ha1
 *
 * tput reset && rm -f /tmp/*.ppm && pkill gdbserver && pkill mpiexec && pkill mono && pkill ctest; m hyoda1
 *
 **/

#include <unistd.h>
#include <signal.h>
#include <QFileInfo>

#include <QHyoda.h>
#include <QHyodaJob.h>
#include <QHyodaIceT.h>
#include <QHyodaMachine.h>
#include <QtNetwork/QtNetwork>

// ****************************************************************************
// * QHyoda class
// ****************************************************************************
QHyoda::QHyoda(int argc, char *argv[]):hyodaQuitRequest(false),
                                       mainSplitter(NULL),
                                       QTabLeft(NULL),
                                       QJob(NULL),
                                       QTabRight(NULL){
  const QString localHostName = QHostInfo::localHostName(); 
  QHyodaMachine *localhost=new QHyodaMachine(this);
  hyodaUi();
  QJob->layout()->addWidget(localhost);
  QHyodaJob *job = new QHyodaJob(localhost,argc,argv);
  localhost->layout()->addWidget(job);
  job->setEnabled(true);
  setHandlerException();
}

// ****************************************************************************
// * ~QHyoda
// ****************************************************************************
QHyoda::~QHyoda(void){
  qDebug() << "QHyoda::~QHyoda DELETE";
  delete QJob;
  delete QTabLeft;
  delete QTabRight;
  delete mainSplitter;
}

// ****************************************************************************
// * setupUi
// ****************************************************************************
void QHyoda::hyodaUi(){
  setupUi(this);
  resize(2048,1152);
  //resize(2240,1260);
  mainSplitter = new QSplitter(Qt::Horizontal, mainWidget);
  mainSplitter->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  verticalLayout->addWidget(mainSplitter);
  // Le Tab left permet de récupérer l'IHM Mathematica, par exemple
  QTabLeft = new QTabWidget(mainWidget);
  QTabLeft->setTabsClosable(true);
  QTabLeft->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
  QTabLeft->hide();
  // Notre central
  QJob = new QWidget(mainWidget);
  QVBoxLayout *vbox = new QVBoxLayout;
  QJob->setLayout(vbox);
  // Le Tab Right permet de capter les xterms, emacs & co
  QTabRight = new QTabWidget(mainWidget);
  QTabRight->setTabsClosable(true);
  QTabRight->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
  QTabRight->hide();
  mainSplitter->addWidget(QTabLeft);
  mainSplitter->addWidget(QJob);
  mainSplitter->addWidget(QTabRight);
  mainSplitter->setSizes(QList<int>()<<128<<512<<128); 
  show();
}


// ****************************************************************************
// * quit
// ****************************************************************************
void QHyoda::quit(void){
//   if (hyodaQuitRequest==false) return;
/*  bool jobed=false;
  qDebug() << "QHyoda::quit";
  for(int machineIdx=ui.machines->count();machineIdx>0;--machineIdx){
    QHyodaMachine *machine=static_cast<QHyodaMachine*>(ui.machines->widget(machineIdx-1));
    QTabWidget *jobs=(static_cast<QHyodaMachine*>(ui.machines->widget(machineIdx-1)))->jobs;
    qDebug() << "QHyoda::quit machine"<<machine->name<<":"<<jobs->count()<<"job(s)";
    for(int jobsIdx=jobs->count();jobsIdx>0;--jobsIdx){
      QHyodaJob *job= static_cast<QHyodaJob*>(jobs->widget(jobsIdx-1));
      jobed|=true;
    }
  }
  if (jobed==false){
    qDebug() << "QHyoda::quit now quit";
    this->close();
  }else{
    qDebug() << "QHyoda::quit not quitting, still people";
    }*/
  this->close();
}

// ****************************************************************************
// * sigaction & restart
// ****************************************************************************
static char *sigAtExitCommand=NULL;
static void hyodaRestart(void){execlp(sigAtExitCommand,sigAtExitCommand,NULL);}
void QHyoda::restart(void){
  atexit(hyodaRestart);
  QCoreApplication::exit(0);
}

// ****************************************************************************
// * sigSegvHandler
// ****************************************************************************
static void sigHandler(int sig) {
  switch (sig){
  case SIGHUP: qDebug() << "Hangup (POSIX)"; break;
  case SIGINT:
    qDebug() << "Interrupt (ANSI)";
    exit(SIGINT);
  case SIGQUIT:
    qDebug() << "Quit (POSIX)";
    exit(SIGQUIT);
  case SIGILL:    qDebug() << "Illegal instruction (ANSI)"; break;
  case SIGTRAP:   qDebug() << "Trace trap (POSIX)"; break;
  case SIGABRT:   qDebug() << "Abort (ANSI)"; break;
  case SIGBUS:    qDebug() << "BUS error (4.2 BSD)"; break;
  case SIGFPE:    qDebug() << "Floating-point exception (ANSI)"; break;
  case SIGKILL:   qDebug() << "Kill, unblockable (POSIX)"; break;
  case SIGUSR1:   qDebug() << "User-defined signal 1 (POSIX)"; break;
  case SIGSEGV:   qDebug() << "Segmentation violation (ANSI)";  break;
  case SIGUSR2:   qDebug() << "User-defined signal 2 (POSIX)"; break;
  case SIGPIPE:   qDebug() << "Broken pipe (POSIX)"; break;
  case SIGALRM:   qDebug() << " Alarm clock (POSIX)"; break;
  case SIGTERM:   qDebug() << "Termination (ANSI)"; break;
  case SIGSTKFLT: qDebug() << "Stack fault"; break;
  case SIGCHLD:   qDebug() << "Child status has changed (POSIX)"; break;
  case SIGCONT:   qDebug() << "Continue (POSIX)"; break;
  case SIGSTOP:   qDebug() << "Stop, unblockable (POSIX)"; break;
  case SIGTSTP:   qDebug() << "Keyboard stop (POSIX)"; break;
  case SIGTTIN:   qDebug() << "Background read from tty (POSIX)"; break;
  case SIGTTOU:   qDebug() << "Background write to tty (POSIX)"; break;
  case SIGURG:    qDebug() << "Urgent condition on socket (4.2 BSD)"; break;
  case SIGXCPU:   qDebug() << "CPU limit exceeded (4.2 BSD)"; break;
  case SIGXFSZ:   qDebug() << "File size limit exceeded (4.2 BSD)"; break;
  case SIGVTALRM: qDebug() << "Virtual alarm clock (4.2 BSD)"; break;
  case SIGPROF:   qDebug() << "Profiling alarm clock (4.2 BSD)"; break;
  case SIGWINCH:  qDebug() << "Window size change (4.3 BSD, Sun)"; break;
  case SIGPOLL:   qDebug() << " Pollable event occurred (System V)"; break;
  case SIGPWR:    qDebug() << "Power failure restart (System V)"; break;
  case SIGSYS:    qDebug() << "Bad system call"; break;
  default:        qDebug() << "Unknown signal";
  }
  exit(EXIT_FAILURE);
}

// ****************************************************************************
// * Setting our handler exception
// ****************************************************************************
void QHyoda::setHandlerException(){
  struct sigaction sigAction;
  sigemptyset(&sigAction.sa_mask);
  sigAction.sa_handler = &sigHandler;
  sigAction.sa_flags = 0;
  sigaction(SIGSEGV, &sigAction, 0);
  sigaction(SIGINT, &sigAction, 0);
}
