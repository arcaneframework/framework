// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <QHyodaJob.h>
#include <QHyodaMachine.h>
#include <QHyodaGdb.h>

#include <QtNetwork/QtNetwork>





/******************************************************************************
 * CONSTRUCTOR
 *****************************************************************************/
QHyodaMachine::QHyodaMachine(QHyoda *parent):hyoda(parent),
                                             localHostName(QHostInfo::localHostName()){
  qDebug() << "[QHyodaMachine::QHyodaMachine] NEW QHyodaMachine on "<<localHostName;
  setLayout(new QVBoxLayout());
  show();
}

QHyodaMachine::~QHyodaMachine(void){
  qDebug() << "[QHyodaMachine::QHyodaMachine] DELETE tab";
}
