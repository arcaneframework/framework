// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <QtWidgets>
//#include <QtGui>
#include "QHyodaJob.h"
#include "QHyodaTool.h"
#include "QHyodaTcp.h"
#include "QHyodaIceT.h"
#include "QHyodaToolMesh.h"

#include "QHyodaGL.h"

#include "ui_hyodaMesh.h"


/******************************************************************************
 * QHyodaToolMesh
 *****************************************************************************/
QHyodaToolMesh::QHyodaToolMesh(QTabWidget *tab)
{
  qDebug() << "[QHyodaToolMesh::QHyodaToolMesh]";
  setupUi(this);
}


QHyodaToolMesh::~QHyodaToolMesh(){
  qDebug() << "[QHyodaToolMesh::~QHyodaToolMesh]";
}
