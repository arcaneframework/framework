// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "QHyodaX11.h"
#include "QHyodaX11Embed.h"

/**********************************************************
 * QHyodaX11Embed class
 **********************************************************/
QHyodaX11Embed::QHyodaX11Embed(QWidget *wdgt,
                               QHyodaX11 *X11):QWidget(wdgt)
{
  qDebug()<<"[QHyodaX11Embed::QHyodaX11Embed] NEW"; 
  connect(this,SIGNAL(clientIsEmbedded()),X11, SLOT(clientIsEmbedded()));
  connect(this,SIGNAL(clientClosed()),X11, SLOT(clientClosed()));
  //connect(this,SIGNAL(error(QX11EmbedContainer::Error)),X11,SLOT(error(QX11EmbedContainer::Error)));
  show();
}

/**********************************************************
 * ~QHyodaX11Embed
 **********************************************************/
QHyodaX11Embed::~QHyodaX11Embed(void){
  qDebug()<<"[QHyodaX11Embed::~QHyodaX11Embed]";
  close();
}
