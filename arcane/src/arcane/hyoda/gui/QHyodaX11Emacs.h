// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_EMACS_H
#define Q_HYODA_EMACS_H

#include <QtWidgets>
#include "QHyoda.h"
#include "QHyodaX11.h"

/****************************************************/
class QHyodaX11Emacs: public QHyodaX11{
  Q_OBJECT
public:
  QHyodaX11Emacs(QWidget*,QString);
  ~QHyodaX11Emacs(void);
public:
  void launchEmacsInWindow(QWidget*, QString);
  void launchXtermInWindow(QWidget *widget);
private:
  void launchServerInWindow(QWidget*, QString);
  void launchClient(QString);
public slots:  
  void serverFinished(int, QProcess::ExitStatus);
  void serverReadyReadStandardOutput(void);
  void clientFinished(int, QProcess::ExitStatus);
  void clientReadyReadStandardOutput(void);
  void xtermFinished(int, QProcess::ExitStatus);
  void xtermReadyReadStandardOutput(void);
public slots:
  void clientClosed();
  void clientIsEmbedded();
  //void error(QX11EmbedContainer::Error);
private:
  bool serverUp;
};

#endif // Q_HYODA_EMACS_H
