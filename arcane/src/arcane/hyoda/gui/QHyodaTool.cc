// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "QHyodaX11Emacs.h"
#include "QHyodaX11XtermLog.h"
#include "QHyodaToolCell.h"
#include "QHyodaToolMesh.h" 
#include "QHyodaToolMatrix.h" 
#include "QHyodaPapi.h" 


QHyodaTool::QHyodaTool(QSplitter *parent):QTabWidget(parent),
                                          job(NULL),
                                          mesh(NULL),
                                          matrix(NULL),
                                          cell(NULL),
                                          xlog(NULL),
                                          src(NULL),
                                          papi(NULL)
{
  setObjectName(QString::fromUtf8("tools"));
  setEnabled(true);
  setTabsClosable(true);
  setMovable(true);
  setCurrentIndex(-1);
  connect(this,SIGNAL(tabCloseRequested(int)),this,SLOT(tools_close_requested_slot(int)));
  connect(this,SIGNAL(currentChanged(int)),this,SLOT(currentChanged(int)));
}

void QHyodaTool::currentChanged(int idx){
  //qDebug() << "\33[7m[QHyodaTool::currentChanged] idx="<<idx<<"\33[m";
  //setCurrentIndex(idx);
}

QHyodaTool::~QHyodaTool(void){
  qDebug() << "[QHyodaTool::~QHyodaTool] "<< objectName();
  delete mesh;
  delete cell;
  delete xlog;
  delete matrix;
  delete papi;
}

void QHyodaTool::setJob(QHyodaJob *thisJob){
  job=thisJob;
}

void QHyodaTool::add_matrix(void){
  addTab(matrix=new QHyodaToolMatrix(this),"Matrix");
}

void QHyodaTool::add_mesh(void){
  mesh=new QHyodaToolMesh(this);
  setCurrentIndex(addTab(mesh,"Mesh"));
  job->meshButton->setEnabled(false);
  qDebug() << "[QHyodaTool::add_mesh] \33[7mAdding variables:\33[m";
  for(int i=0; i<job->arcane_variables_names->size(); ++i){
    qDebug() << "[QHyodaTool::add_mesh] Adding variable "<<job->arcane_variables_names->at(i);
    mesh->variablesComboBox->addItem(job->arcane_variables_names->at(i));
  }
  connect(mesh->variablesComboBox,SIGNAL(currentIndexChanged(int)),
          this,SLOT(tools_variable_index_change(int)));
}
void QHyodaTool::tools_variable_index_change(int index){
  qDebug() << "[QHyodaTool] tools_variable_index_change:"<<index;
}

void QHyodaTool::add_cell(void){
  cell=new QHyodaToolCell(job, this);
  setCurrentIndex(addTab(cell,"Cell"));
  cell->setRange(job->data->mesh_total_nb_cell);
  job->cellButton->setEnabled(false);
}

void QHyodaTool::add_log(void){
  xlog=new QHyodaX11XtermLog(job,this,"Log");
  setCurrentIndex(addTab(xlog,"Log"));
  //job->logButton->setEnabled(true);
}

void QHyodaTool::add_src(void){
  /*QString daemonIdStr("src");
  QHyodaX11Emacs *emacs=new QHyodaX11Emacs(this, daemonIdStr);
  int idx=addTab(emacs, "Src");
  emacs->launchEmacsInWindow(widget(idx), NULL);
  setCurrentIndex(idx);
  job->srcButton->setEnabled(false);*/
}

void QHyodaTool::add_papi(void){
  papi=new QHyodaPapi(this);
  setCurrentIndex(addTab(papi,"Papi"));
}

void QHyodaTool::tools_close_requested_slot(int index){
  qDebug() << "[QHyodaTool::tools_close_requested_slot]";
  if (tabText(index)=="Mesh"){
    mesh->close();
    job->meshButton->setEnabled(true);
  }
  if (tabText(index)=="Cell"){
    cell->close();
    job->cellButton->setEnabled(true);
  }
  if (tabText(index)=="Log"){
    if (xlog) xlog->close();
    job->logButton->setEnabled(true);
  }
  if (tabText(index)=="Src")
    job->srcButton->setEnabled(true);

  if (tabText(index)=="Papi")
    job->papiButton->setEnabled(true);
  
  removeTab(index);
  
  if (count()==0) hide();
}
