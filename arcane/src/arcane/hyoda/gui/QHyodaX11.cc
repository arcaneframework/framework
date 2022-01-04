// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "QHyoda.h"
#include "QHyodaX11.h"
#include "QHyodaX11Emacs.h"

#include <QtWidgets>
#include <QProcess>

/*
 * xwininfo
 * xwininfo -root -tree|grep XMathematica
 * xlsclients -l|tr -d \'|sed -ne 's/^ *Window *\(0x[0-9a-f]*\):$/\1/p'
 * xlsclients -l|tr -d \'|sed -ne 's/^ *Instance\/Class: *\(.*\)\//\1/p'
 * xlsclients -l|tr -d \'|sed -ne 's/^ *Name: *\(.*\)/\1/p'
 * xlsclients -l|tr -d :\'|awk '/Window/ {winid=$2};  $1=="Name"{name=$2}; $1=="Instance/Class" && $2=="XMathematica/" && name!="" {printf winid "\n"}'
 */


/******************************************************************************
 * QHyodaX11
 *****************************************************************************/
QHyodaX11::QHyodaX11(QWidget *tabWidgetParent,QString ttl):
  QWidget(tabWidgetParent),
  parentWidget(tabWidgetParent),
  title(ttl),
  layout(new QVBoxLayout),
  //splitter(new QSplitter(Qt::Vertical,this)),
  X11Process(QList<QProcess*>())
  //X11Container(QList<QHyodaX11Embed*>())
{
  qDebug() << "[QHyodaX11::QHyodaX11] New";
  
  //QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  //sizePolicy.setHorizontalStretch(1);
  //sizePolicy.setVerticalStretch(1);
  //sizePolicy.setHeightForWidth(splitter->sizePolicy().hasHeightForWidth());

  //splitter->setSizePolicy(sizePolicy);
  //layout->addWidget(splitter);
  setLayout(layout);
  show();
}


/***********************************************
 * ~QHyodaX11
 ***********************************************/
QHyodaX11::~QHyodaX11(void){ 
  qDebug() << "[QHyodaX11::~QHyodaX11]";
  while (!X11Process.isEmpty())
    X11Process.takeFirst()->close();
}

