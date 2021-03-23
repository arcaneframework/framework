// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_JOB_TOOL_CELL_H
#define Q_HYODA_JOB_TOOL_CELL_H

#include <QtWidgets>
#include <ui_hyodaCell.h>

class QHyodaJob;
class QHyodaTool;
class QHyodaGLCell;

class QHyodaToolCell: public QWidget, public  Ui::toolCellWidget{
  Q_OBJECT
public:
  QHyodaToolCell(QHyodaJob*,QHyodaTool*);
  ~QHyodaToolCell();
public:
  void setRange(quint64);
  void refresh(const QStringList &);
  void close();
public slots:
  void targetCellUidLineEditSlot(void);
  void targetCellUidSliderSlot(int value);
public:
  QHyodaJob *job;
  QHyodaTool *tools;
  QHyodaGLCell *cell;
private:
  double coords[8][3];
};
#endif //  Q_HYODA_JOB_TOOL_CELL_H
