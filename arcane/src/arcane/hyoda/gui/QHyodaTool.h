// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_TOOL_H
#define Q_HYODA_TOOL_H

#include <QtWidgets>

class QHyodaJob;
class QHyodaToolLog;
class QHyodaToolSrc;
class QHyodaToolCell;
class QHyodaToolMesh;
class QHyodaToolMatrix;
class QHyodaX11XtermLog;
class QHyodaPapi;

class QHyodaTool:public QTabWidget{
  Q_OBJECT
public:
  QHyodaTool(QSplitter*);
  ~QHyodaTool();
public:
  void setJob(QHyodaJob *thisJob);
public:
  void add_mesh();
  void add_matrix();
  void add_cell();
  void add_src();
  void add_log();
  void add_papi();
public slots:
  void currentChanged(int);
  void tools_close_requested_slot(int);
  void tools_variable_index_change(int);
public:
  QHyodaJob *job;
public:
  QHyodaToolMesh *mesh;
  QHyodaToolMatrix *matrix;
  QHyodaToolCell *cell;
  QHyodaX11XtermLog *xlog;
  QHyodaToolSrc *src;
  QHyodaPapi *papi;
};

#endif // Q_HYODA_TOOL_H
