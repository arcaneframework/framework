// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* QHyodaX11Embed.h                                            (C) 2000-2022 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef Q_HYODA_X11_EMBED_H
#define Q_HYODA_X11_EMBED_H

#include <QtWidgets>
//#include <QX11EmbedContainer>
class QHyodaX11;

class QHyodaX11Embed: public QWidget{
  Q_OBJECT
public:
  QHyodaX11Embed(QWidget*, QHyodaX11*);
  ~QHyodaX11Embed(void);
};

#endif // Q_HYODA_X11_EMBED_H
