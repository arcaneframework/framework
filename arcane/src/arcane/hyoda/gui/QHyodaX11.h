// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_X11_H
#define Q_HYODA_X11_H

#include "QHyoda.h"
#include "QHyodaX11.h"
#include "QHyodaX11Embed.h"

#include <QtWidgets>
#include <QString>
#include <QWidget>
#include <QProcess>
#include <QTabWidget>
//#include <QX11EmbedContainer>


/******************************************************************************
 * QHyodaX11
 ******************************************************************************/
class QHyodaX11:public QWidget{
  Q_OBJECT
public:
  QHyodaX11(QWidget*,QString);
  ~QHyodaX11(void);
public:
  virtual void clientClosed()=0;
  virtual void clientIsEmbedded()=0;
  //virtual void error(QX11EmbedContainer::Error)=0;
public:
  QWidget *parentWidget;
  QString title;
  QVBoxLayout *layout;
  QSplitter *splitter;
  QList<QProcess*> X11Process;
  //QList<QHyodaX11Embed*> X11Container;
};

#endif // Q_HYODA_X11_H
