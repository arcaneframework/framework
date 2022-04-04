// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
//#include <QtGui>
#include <QtWidgets>

#include "QHyoda.h"
#include "QHyodaPapi.h"
#include "QHyodaTool.h"

#include "ui_hyodaPapi.h"

#ifdef __GNUG__
#include <cxxabi.h>
#endif

// tput reset && pkill gdbserver && pkill mpiexec && pkill mono && pkill ctest && pkill glvis ; m hyoda1


// ****************************************************************************
// * Constructeur/Destructeur
// ****************************************************************************
QHyodaPapi::QHyodaPapi(QHyodaTool *QTool):tool(QTool){
  qDebug() << "[QHyodaPapi::QHyodaPapi]";
  setupUi(this);
}

QHyodaPapi::~QHyodaPapi(){
  qDebug() << "[QHyodaPapi::~QHyodaPapi]";
}


// ****************************************************************************
// * Initialisation des progress bar et group boxes
// ****************************************************************************
void QHyodaPapi::ini(){
  QVBoxLayout *profilerLayout = tool->papi->profilerLayout;
  if (profilerLayout->count()==0){
    for(int i=0;i<max_nb_func_to_profile;i+=1){
      QGroupBox *box = new QGroupBox;      
      QVBoxLayout *vbox = new QVBoxLayout;
      QProgressBar *bar = new QProgressBar;
      vbox->addWidget(bar);
      box->setLayout(vbox);
      profilerLayout->addWidget(box);
    }
  }
}


// ****************************************************************************
// * Update et lancement de l'init si besoin
// ****************************************************************************
void QHyodaPapi::update(QByteArray *read){
  QVBoxLayout *profilerLayout = tool->papi->profilerLayout;
  
  // Test s'il faut lancer l'init
  if (profilerLayout->count()==0) ini();
  
  //qDebug() << "\n[QHyodaPapi::update]";

  // Et on dÃ©pile le TCP
  const char *data=read->data();
  const qint64 *pkt=(qint64*)&data[0];
  const qint64 size=pkt[0];
  const int capacity=read->capacity();
  
  //qDebug() << "[QHyodaPapi::update] capacity: "<< capacity;
  //printf("[QHyodaPapi::update] size %d\n",size);
  
  if (capacity!=size<<3)
    qFatal("[QHyodaPapi::update] capacity missmatch size !");
  
  const qint64 total_event = pkt[1];
  const qint64 total_fp = pkt[2];
  const qint64 m_period = pkt[3];
  const qint64 index = pkt[4];

  //printf("[QHyodaPapi::update] total_event %d\n",total_event);
  //printf("[QHyodaPapi::update] total_fp %d\n",total_fp);
  //printf("[QHyodaPapi::update] m_period %d\n",m_period);
  //qDebug() << "[QHyodaPapi::update] index"<<index;

  int of7=5;
  char demangled_func_name[8192];
  for(int i=0;i<index;i+=1){
    const qint64 nb_event = pkt[of7++];
    const qint64 total_percent = pkt[of7++];
    const qint64 m_counters_0 = pkt[of7++]; // PAPI_TOT_CYC Total cycles
    const qint64 m_counters_1 = pkt[of7++]; // PAPI_RES_STL Cycles stalled on any resource
    const qint64 m_counters_2 = pkt[of7++]; // PAPI_FP_INS  Floating point instructions
    const qint64 mx = pkt[of7++];
    const char* func_name = (char*)&pkt[of7];
        
    //qDebug() <<"[QHyodaPapi::update]        nb_event %d\n", nb_event);
    //printf("[QHyodaPapi::update]    Total cycles %d\n", m_counters_0);
    //printf("[QHyodaPapi::update]  Stalled cycles %d\n", m_counters_1);
    //printf("[QHyodaPapi::update] FP instructions %d\n", m_counters_2);
    //qDebug() << "[QHyodaPapi::update] mx="<<mx<<", func_name="<<func_name;
    
    size_t len = 8192;
    int dstatus = 0;
    const char* buf = 0; 
#ifdef __GNUG__
    buf = abi::__cxa_demangle(func_name,demangled_func_name,&len,&dstatus);
#endif
    if (!buf) memcpy(&demangled_func_name[0],func_name,strlen(func_name)+1);
    
    //qDebug() <<"[ProfInfos::getInfos] demangled_func_name="<<demangled_func_name;
    //printf("[QHyodaPapi::update] func_name %s\n", demangled_func_name);

    QGroupBox *box = static_cast<QGroupBox*>(profilerLayout->itemAt(i)->widget());
    
    if (box->title() != demangled_func_name) box->setTitle(demangled_func_name);
    
    QProgressBar *bar = box->findChild<QProgressBar*>();
    int value =  (nb_event*100)/total_event;
    //qDebug() << "[QHyodaPapi::update] bar min= "<< bar->minimum() << "bar max= "<< bar->maximum();
    if (bar->value() != value) bar->setValue(value);
    
    of7+=1+(mx>>3);
    //qDebug()<<"[ProfInfos::getInfos] nbPktForName="<<(mx>>3);

    // On ne propose que les max_nb_func_to_profile premiÃ¨res fonction max
    if (i==max_nb_func_to_profile-1){
      //qDebug() << "[QHyodaPapi::update] max_nb_func_to_profile!, breaking!";
      break;
    }
  }
  delete read;
}
