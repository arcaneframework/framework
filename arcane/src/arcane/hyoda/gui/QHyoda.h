// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_H
#define Q_HYODA_H

#include <ui_hyoda.h>
#include <QFileInfo>
#include <QSplitter>

class QHyoda: public QMainWindow, public Ui::hyodaWindow{
  Q_OBJECT
public:
  QHyoda(int,char**);
  ~QHyoda();
public:
  void quit();
  void hyodaUi();
  void setHandlerException();
public slots:
  //void detach();
  void restart();
public:
  void toys();
public:
  bool hyodaQuitRequest;
public:
  QSplitter *mainSplitter;
  QTabWidget *QTabLeft;
  QWidget *QJob;
  QTabWidget *QTabRight;
};

#endif // Q_HYODA_H
