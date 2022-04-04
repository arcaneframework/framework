// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <QHyodaTcp.h>
#include <QHyodaMachine.h>
#include "QHyodaPapi.h"


/******************************************************************************
 * STATIC DECLARATIONS
 *****************************************************************************/
static double rawStringToDouble(QString);
static unsigned long long rawStringToULongLong(QString);


/******************************************************************************
 * CONSTRUCTOR
 *****************************************************************************/
QHyodaJob::QHyodaJob(QHyodaMachine *_machine,
                     int _argc,char *_argv[]):
  machine(_machine),
  argc(_argc),
  argv(_argv),
  id(-1),
  has_been_broadcasted(false),
  tcpAdrs(0x100007Ful),
  tcpPort(3889),
  tcpPyld(8*1024),
  iceWidthHeight(0x04000300ul),
  gdb(new QHyodaGdb(this,machine,0x100007Ful,tcpPort,tcpPyld)),
  gdbserver(new QProcess()),
  tack_toggle(0),
  data(NULL),
  tcp(new QHyodaTcp(this)),
  arcane_variables_names(new QStringList()){
  qDebug() << "\33[36m[QHyodaJob::QHyodaJob] NEW @"<<machine->localHostName<<"\33[m";
  setupUi(this);

  // Ratio entre la partie gdb et les tools
  jobSplitter->setSizes(QList<int>()<<128<<1024);
  // Ratio entre l'output gdbserver, celui de gdb et les common variables d'Arcane
  gdbSplitter->setSizes(QList<int>()<<128<<1024);
  
  // Ratio entre les tools elles mêmes
  toolsSplitter->setSizes(QList<int>()<<512<<256);
  topToolsSplitter->setSizes(QList<int>()<<512<<512);
  bottomToolsSplitter->setSizes(QList<int>()<<512<<512);
 
  topLeftTools->hide();      // Matrix
  topRightTools->hide();     // IceT mesh
  bottomLeftTools->hide();   // xterm log 
  bottomRightTools->hide();  // papi,cell
  
  host=machine->localHostName;
        
  topLeftTools->setJob(this);
  topRightTools->setJob(this);
  bottomLeftTools->setJob(this);
  bottomRightTools->setJob(this);

  // On prépare les connections  
  connect(tackButton,SIGNAL(clicked()),this,SLOT(tack_slot()));
  connect(startButton,SIGNAL(clicked()),this,SLOT(start_slot()));
  connect(stepButton,SIGNAL(clicked()),this,SLOT(step_slot()));
  connect(stopButton,SIGNAL(clicked()),this,SLOT(stop_slot()));

  connect(matrixButton,SIGNAL(clicked()),this,SLOT(matrix_slot()));
  connect(logButton,SIGNAL(clicked()),this,SLOT(log_slot()));
  connect(srcButton,SIGNAL(clicked()),this,SLOT(src_slot()));
  connect(meshButton,SIGNAL(clicked()),this,SLOT(mesh_slot()));
  connect(cellButton,SIGNAL(clicked()),this,SLOT(cell_slot()));
  connect(papiButton,SIGNAL(clicked()),this,SLOT(papi_slot()));

  if ((data=(hyoda_taxi_data*)calloc(1,sizeof(hyoda_taxi_data)))==NULL)
    qFatal("\33[36m[QHyodaJob::QHyodaJob] Could not allocate data space for hook\33[m");

  // C'est le log slot qui va lancer l'xterm et la commande
  bottomLeftTools->show();
  bottomLeftTools->add_log();

  //bottomRightTools->show();bottomRightTools->add_papi();
  //bottomRightTools->add_cell();
  //bottomRightTools->add_papi();
  
  //topRightTools->show();topRightTools->add_mesh();
  
  //topLeftTools->show();topLeftTools->add_matrix();
}



/******************************************************************************
 * QHyodaJob::~QHyodaJob
 *****************************************************************************/
QHyodaJob::~QHyodaJob(void){
  qDebug() << "\33[36m[QHyodaJob::~QHyodaJob] DELETE\33[m";
  delete tcp;
  delete gdb;
  delete gdbserver;
  delete arcane_variables_names;
  free(data);  
}


/******************************************************************************
 * GDBSERVER process
 *****************************************************************************/
void QHyodaJob::gdbserver_hook(void){
  gdbServerGroupBox->setEnabled(true);
  qDebug() << "\33[36m[QHyodaJob::gdbserver_hook] #### LOCAL gdbserver ####\33[m";
  //QString command("/usr/local/bin/gdbserver");
  QString command("/usr/bin/gdbserver");
  QStringList args = QStringList() << "--attach"
                                   << ":3883"
                                   << pid;
  gdbserver->setProcessChannelMode(QProcess::MergedChannels);
  connect(gdbserver,SIGNAL(readyReadStandardOutput()),this,SLOT(gdbserver_slot()));
  gdbserver->start(command, args);
  if (!gdbserver->waitForStarted())
    qFatal("\33[36m[QHyodaJob::gdbserver_hook] NOT started!\33[m");
}



/******************************************************************************
 * GDB SERVER process
 * C'est ici qu'on lance le gdbserver coté rank0
 *****************************************************************************/
void QHyodaJob::gdbserver_slot(void){
  QString output=QString(gdbserver->readAllStandardOutput());
  //qDebug() << "\33[36m[QHyodaJob::gdbserver_slot] output="<<output<<"\33[m";
  if (output.endsWith('\n'))
    output.replace(output.length()-1,1,' ');
  gdbServerTextEdit->append(output);
  if (output.contains("Listening")){
    qDebug() << "\33[33m[QHyodaJob::gdbserver_slot] Launching gdb_hook\33[m";
    gdb_hook();
  }
}

/******************************************************************************
 * GDB process
 * C'est ici qu'on setup les breapoints avec lesquels nous allons jouer
 *****************************************************************************/
void QHyodaJob::gdb_hook(void){
  gdb->cmdline=(has_been_broadcasted==true)?broadcasted_cmdline:cmdline;
  qDebug() << "\33[36m[QHyodaJob::gdb_hook] Now launching\33[m";
  gdb->launch();
  gdbGroupBox->setEnabled(true);
  jobButtons->setEnabled(true);
}

/******************************************************************************
 * TOOLBAR SLOTS
 *****************************************************************************/
void QHyodaJob::stop_slot(void){
  gdb->state=QHyodaGdb::Interrupt;
  startButton->setEnabled(true);
  stepButton->setEnabled(true);
}

void QHyodaJob::step_slot(void){
  stepButton->setEnabled(false);
  startButton->setEnabled(false);
  tackButton->setEnabled(false);
  if (gdb->state!=QHyodaGdb::Step)
    gdb->state=QHyodaGdb::Step;
  else
    gdb->enqueue("-exec-continue");
}

void QHyodaJob::start_slot(void){
  gdb->state=QHyodaGdb::Continue;
  gdb->enqueue("-break-disable 3"); // On enlève le softbreak
  gdb->enqueue("-break-disable 4"); // On enlève les entryPoints
  gdb->enqueue("-exec-continue");
  tackButton->setEnabled(true);
  startButton->setEnabled(false);
}

void QHyodaJob::tack_slot(void){
  tack_toggle=(tack_toggle+1)%2;
  if (tack_toggle!=0){
    gdb->state=QHyodaGdb::Untack;
    arcaneCommonVariablesGroupBox->setEnabled(false);
    //gdbInputLineEdit->setEnabled(false);
    stopButton->setEnabled(false);
    stepButton->setEnabled(false);
  }else{
    gdb->state=QHyodaGdb::Retack;
    gdb->enqueue("-exec-interrupt");
    arcaneCommonVariablesGroupBox->setEnabled(true);
    //gdbInputLineEdit->setEnabled(true);
    stopButton->setEnabled(true);
    stepButton->setEnabled(true);
  }
}



void QHyodaJob::log_slot(void){
  //qDebug() << "\33[36m[QHyodaJob::log_slot]\33[m";
  if (bottomLeftTools->isHidden())
    bottomLeftTools->show();
  else
    bottomLeftTools->hide();
}
void QHyodaJob::cell_slot(void){
  qDebug() << "\33[36m[QHyodaJob::cell_slot]\33[m";
  if (!bottomRightTools->cell) bottomRightTools->add_cell();
  bottomRightTools->show();
  //if (bottomRightTools->isHidden()) bottomRightTools->show();  else bottomRightTools->hide();
}

void QHyodaJob::mesh_slot(void){
  qDebug() << "\33[36m[QHyodaJob::mesh_slot]\33[m";
  if (!topRightTools->mesh) topRightTools->add_mesh();
  if (topRightTools->isHidden()) topRightTools->show();
  else topRightTools->hide();
}

void QHyodaJob::src_slot(void){
  qDebug() << "\33[36m[QHyodaJob::src_slot]\33[m";
  topLeftTools->show();
  topLeftTools->add_src();
}

void QHyodaJob::matrix_slot(void){
  qDebug() << "\33[36m[QHyodaJob::matrix_slot]\33[m";
  topLeftTools->show();
  topLeftTools->add_matrix();
}

void QHyodaJob::papi_slot(void){
  if (!bottomRightTools->papi) bottomRightTools->add_papi();
  if (bottomRightTools->isHidden()) bottomRightTools->show();
  else bottomRightTools->hide();
}


/**
 * detach
 **/
void QHyodaJob::detach(void){
  qDebug() << "\33[36m[QHyodaJob::detach] Queuing Detach and interrupting job\33[m";
  gdb->state=QHyodaGdb::Detach;
  gdb->enqueue("-exec-interrupt");
}


/**
 * quit
 **/
void QHyodaJob::quit(void){
  qDebug() << "\33[36m[QHyodaJob::quit]\33[m";
  gdb->process->close();
  if (gdb->process->state()!=QProcess::NotRunning)
    qDebug() << "\33[36m[QHyodaJob::quit] GDB NOT closed\33[m";
  else
    qDebug() << "\33[36m[QHyodaJob::quit] GDB closed\33[m";
  
  gdbserver->close();
  if (gdbserver->state()!=QProcess::NotRunning)
    qDebug() << "\33[36m[QHyodaJob::quit] GDBSERVER NOT closed\33[m";
  else
    qDebug() << "\33[36m[QHyodaJob::quit] GDBSERVER closed\33[m";

  this->close();
  machine->hyoda->quit();
}


/******************************************************************************
 * REFRESH'MENTS
 *****************************************************************************/
void QHyodaJob::refresh_common_variables(const QStringList &splitted_output){ 
  //qDebug() << "QHyodaJob::jobid_read_gdb splitted_output="<<splitted_output;
  
  quint64 global_iteration=rawStringToULongLong(splitted_output.at(17));
  globalIterationLine->setText(QString("%1").arg(global_iteration));
  
  double global_time=rawStringToDouble(splitted_output.at(19));
  globalTimeLine->setText(QString("%1").arg(global_time));
  
  double global_deltat=rawStringToDouble(splitted_output.at(21));
  globalDeltaTLine->setText(QString("%1").arg(global_deltat));
  
  double global_cpu_time=rawStringToDouble(splitted_output.at(23));
  globalCPUTimeLine->setText(QString("%1").arg(global_cpu_time));
  
  quint64 mesh_total_nb_cell=rawStringToULongLong(splitted_output.at(25));
  if (mesh_total_nb_cell!=data->mesh_total_nb_cell){
    data->mesh_total_nb_cell=mesh_total_nb_cell;
    //qDebug() << "QHyodaJob::refresh_common_variables new mesh_total_nb_cell="<<mesh_total_nb_cell;
  }
   unsigned long long got_target_cell_nb_nodes=rawStringToULongLong(splitted_output.at(31));
}






/******************************************************************************
 * STATIC TOOLS
 *****************************************************************************/
static double rawStringToDouble(QString qStr){
  const char *s=qStr.toLocal8Bit().constData();
  double ret=0.;
  sscanf(s, "%llx", (unsigned long long *)&ret);
  return ret;
}

static unsigned long long rawStringToULongLong(QString qStr){
  const char *s=qStr.toLocal8Bit().constData();
  unsigned long long ret=0;
  sscanf(s, "%llx", &ret);
  return ret;
}

