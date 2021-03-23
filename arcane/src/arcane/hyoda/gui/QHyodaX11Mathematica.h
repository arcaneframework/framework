// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_MATHEMATICA_H
#define Q_HYODA_MATHEMATICA_H

#include <QtWidgets>
#include "QHyoda.h"
#include "QHyodaX11.h"


/****************************************************/
class QHyodaX11Mathematica: public QHyodaX11{
  Q_OBJECT
public:
  QHyodaX11Mathematica(QWidget*,QFileInfo);
  ~QHyodaX11Mathematica(void);
  void launch(QWidget*);
public slots:  
  void mathFinished(int, QProcess::ExitStatus);
  void mathReadyReadStandardOutput(void);
  void xwinReadyReadStandardOutput(void);
public slots:
  void clientClosed();
  void clientIsEmbedded();
  //void error(QX11EmbedContainer::Error);
public:
  QFileInfo fileInfo;
  QString cmdMathematica;
  QStringList argsMathematica;
  QString cmdXwininfo;
  QStringList argsXwininfo;
  long winId;
  QWidget *win;
  //QWidget *tabParent;
};

#endif // Q_HYODA_EMACS_H
