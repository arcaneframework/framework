// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* QHyodaToolMesh.h                                            (C) 2000-2022 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef Q_HYODA_JOB_TOOL_MESH_H
#define Q_HYODA_JOB_TOOL_MESH_H

#include <QtWidgets>
#include "QHyodaIceT.h"
#include "ui_hyodaMesh.h"

class QHyodaToolMesh: public QWidget, public  Ui::toolMeshWidget{
  Q_OBJECT
public:
  QHyodaToolMesh(QTabWidget*);
  ~QHyodaToolMesh();
};
#endif //  Q_HYODA_JOB_TOOL_MESH_H
