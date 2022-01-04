// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_XTERM_LOG_H
#define Q_HYODA_XTERM_LOG_H

#include <QtWidgets>
#include <QWindow>
#include "QHyoda.h"
#include "QHyodaJob.h"
#include "QHyodaX11.h"
#include <QResizeEvent>

class QHyodaX11XtermLog: public QWidget{
  Q_OBJECT
 public:
  QHyodaX11XtermLog(QHyodaJob*,QWidget*,QString);
  ~QHyodaX11XtermLog(void){}
public:
  void resizeEvent(QResizeEvent*);
  void xwininfoStart();
public slots:  
  void xtermFinished(int, QProcess::ExitStatus);
  void xwininfoOutput();
private:
  void setupUI(QWidget*);
  void setXtermSizeWithXdotool(const QSize);
private:
  QHyodaJob *job;
  QProcess *xterm;
  QProcess *xwininfo;
  QProcess *xdotool;
  QWidget *xWidget;
  QString xtermTitle;
  WId WinId;
  QSize size;
};

#endif // Q_HYODA_XTERM_LOG_H
