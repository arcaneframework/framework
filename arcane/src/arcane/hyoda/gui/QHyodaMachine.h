// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_MACHINE_H
#define Q_HYODA_MACHINE_H

#include "QHyoda.h"
#include "QHyodaJob.h"
#include "QHyodaMachine.h"

#include <QList>
#include <QThread>
#include <QProcess>
#include <QStringList>
#include <QSignalMapper>

#include <QtWidgets>

#include <QtCore/QVariant>

class QHyodaJob;


/******************************************************************************
 * Classe QHyodaMachine
 *****************************************************************************/
class QHyodaMachine: public QWidget {
  Q_OBJECT
public:
  QHyodaMachine(QHyoda*);
  ~QHyodaMachine();
public:
  QHyoda *hyoda;
  QString localHostName;
};

#endif // Q_HYODA_MACHINE_H
