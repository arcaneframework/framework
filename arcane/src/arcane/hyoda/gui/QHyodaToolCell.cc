// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "QHyodaJob.h"
#include "QHyodaToolCell.h"
#include "QHyodaGLCell.h"


/******************************************************************************
 * STATIC DECLARATIONS
 *****************************************************************************/
static double rawStringToDouble(QString qStr);
static unsigned long long rawStringToULongLong(QString qStr);




/******************************************************************************
 * QHyodaToolCell
 *****************************************************************************/
QHyodaToolCell::QHyodaToolCell(QHyodaJob *_job,
                               QHyodaTool *_tools):job(_job),
                                                   tools(_tools),
                                                   cell(new QHyodaGLCell(this)){
  qDebug() << "QHyodaToolCell::QHyodaToolCell";
  setupUi(this); 
  groupBox->layout()->addWidget(cell);
  vLayout->setStretch(1,1);
  connect(targetCellNumLineEdit, SIGNAL(returnPressed()), this, SLOT(targetCellUidLineEditSlot()));
  connect(cellHSlider,SIGNAL(valueChanged(int)), this, SLOT(targetCellUidSliderSlot(int)));
  targetCellNumLineEdit->setEnabled(true); 
  cellHSlider->setEnabled(true);
  qDebug() << "QHyodaToolCell::QHyodaToolCell done";
}


/******************************************************************************
 * ~QHyodaToolCell
 *****************************************************************************/
QHyodaToolCell::~QHyodaToolCell(){
  qDebug() << "QHyodaToolCell::~QHyodaToolCell";
  delete cell;
}


void QHyodaToolCell::close(){
  qDebug() << "QHyodaToolCell::cell";
}


/******************************************************************************
 * setRange
 *****************************************************************************/
void QHyodaToolCell::setRange(quint64 mesh_total_nb_cell){
  //qDebug()<<"QHyodaToolCell::setRange"<<mesh_total_nb_cell;
  cellHSlider->setMaximum(mesh_total_nb_cell-1);
  cellHSlider->setTickInterval(mesh_total_nb_cell>>2);
  cellHSlider->setPageStep(mesh_total_nb_cell>>3);
  cellHSlider->setSingleStep(mesh_total_nb_cell>>8);
  cellHSlider->setValue((mesh_total_nb_cell>>1)-1);
  //qDebug()<<"QHyodaToolCell::setRange done";
}

static unsigned int l2g(unsigned int i){
  if (i==0) return 0;
  if (i==1) return 1;
  if (i==2) return 2;
  if (i==3) return 3;
  if (i==4) return 4;
  if (i==5) return 5;
  if (i==6) return 6;
  if (i==7) return 7;
  qFatal("l2g INVALID code!");
}

/******************************************************************************
 * refresh
 *****************************************************************************/
void QHyodaToolCell::refresh(const QStringList &splitted_output){
  //qDebug()<<"\33[7m[QHyodaToolCell::refresh] splitted_output=" << splitted_output << "\33[m";
  //qDebug()<<"QHyodaToolCell::refresh target_cell_uid"<<job->data->target_cell_uid;
  
  unsigned long long got_target_cell_rank=rawStringToULongLong(splitted_output.at(29));
  uidRankLabel->setText(QString("%1").arg(got_target_cell_rank));
  
  cell->cell_nb_nodes=rawStringToULongLong(splitted_output.at(31));
  //qDebug()<<"\33[7m[QHyodaToolCell::refresh] got_target_cell_nb_nodes="<<cell->cell_nb_nodes<< "\33[m";

  const int offset=33;

  // Flush current coords
  for(unsigned int i=0;i<cell->cell_nb_nodes; ++i)
    coords[i][0]=coords[i][1]=coords[i][2]=0.0;

  // On flush toute la structure géométrique
  cell->clear();

  // On fetch le noeud 0, mais on l'impose à 0.0 (on le translate à l'origine)
  coords[0][0]=rawStringToDouble(splitted_output.at(offset+0));
  coords[0][1]=rawStringToDouble(splitted_output.at(offset+2));
  coords[0][2]=rawStringToDouble(splitted_output.at(offset+4));
  // On le pousse dans la géométrie comme référence
  cell->add_node(QVector3D(0.0,0.0,0.0), QColor(Qt::blue));

  for(unsigned int iNode=1, iAt=offset+6; iNode<cell->cell_nb_nodes; ++iNode,iAt+=6){
    coords[iNode][0]=rawStringToDouble(splitted_output.at(iAt))-coords[0][0];
    coords[iNode][1]=rawStringToDouble(splitted_output.at(iAt+2))-coords[0][1];
    coords[iNode][2]=rawStringToDouble(splitted_output.at(iAt+4))-coords[0][2];
    //qDebug()<<"QHyodaToolCell::refresh "<<coords[iNode][0]<<coords[iNode][1]<<coords[iNode][2];
    cell->add_node(QVector3D(coords[iNode][0],coords[iNode][1],coords[iNode][2]), QColor(Qt::cyan));
  }
  cell->update();
}


/******************************************************************************
 * targetCellUidLineEditSlot
 *****************************************************************************/
void QHyodaToolCell::targetCellUidLineEditSlot(void){
  bool ok;
  unsigned long long text2ull=targetCellNumLineEdit->text().toULongLong(&ok,10);
  if ((!ok) || (text2ull>=job->data->mesh_total_nb_cell)){
    qDebug()<<"QHyodaToolCell::targetCellUidLineEditSlot ERROR while converting targetCellUid";
    return;
  }
  job->data->target_cell_uid=text2ull;
  cellHSlider->setValue(text2ull);
  job->gdb->state=QHyodaGdb::TargetCell;
}


/******************************************************************************
 * targetCellUidSliderSlot
 *****************************************************************************/
void QHyodaToolCell::targetCellUidSliderSlot(int value){
//  qDebug()<<"QHyodaJob::targetCellUidSliderSlot targetCellUid"<<cellHSlider->value();
  targetCellNumLineEdit->setText(QString("%1").arg(value));
  job->data->target_cell_uid=value;
  job->gdb->state=QHyodaGdb::TargetCell;
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
