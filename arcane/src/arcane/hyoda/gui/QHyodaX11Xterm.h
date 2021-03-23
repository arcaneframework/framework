// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_XTERM_H
#define Q_HYODA_XTERM_H

#include <QtWidgets>
#include "QHyoda.h"
#include "QHyodaX11.h"


/****************************************************/
class QHyodaX11Xterm: public QHyodaX11{
  Q_OBJECT
public:
  QHyodaX11Xterm(QWidget*,QString);
  ~QHyodaX11Xterm(void);
  void launch(QWidget *widget);
public slots:  
  bool close();
  void started ();
  void finished(int, QProcess::ExitStatus);
  void readyReadStandardOutput(void);
  void stateChanged(QProcess::ProcessState);
public slots:
  void clientClosed();
  void clientIsEmbedded();
  //void error(QX11EmbedContainer::Error);
};



#endif // Q_HYODA_XTERM_H
