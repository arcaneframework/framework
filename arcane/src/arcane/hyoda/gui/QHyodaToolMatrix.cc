// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "QHyodaJob.h"
#include "QHyodaTool.h"
#include "QHyodaTcp.h"
#include "QHyodaIceT.h"
#include "QHyodaToolMatrix.h"

#include <QVector4D> 

// ****************************************************************************
// * QHyodaToolMatrix
// ****************************************************************************
QHyodaToolMatrix::QHyodaToolMatrix(QTabWidget *tab)
{
  qDebug() << "\33[1;32m[QHyodaToolMatrix::QHyodaToolMatrix]\33[0m";
  setupUi(this);
  ice->setPov(QVector4D(0,0,0,1));
}


QHyodaToolMatrix::~QHyodaToolMatrix(){
  qDebug() << "\33[1;32m[QHyodaToolMatrix::~QHyodaToolMatrix]\33[0m";
}
