// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_GDB_H
#define Q_HYODA_GDB_H

#include <QObject>
#include <QQueue>
#include <QProcess>

class QHyodaJob;
class QHyodaMachine;

class QHyodaGdb:public QObject{
  Q_OBJECT
public:
  enum QHyodaGdbCommand{
    None=0,
    Quit,
    Detach,
    Interrupt,
    Step,
    Continue,
    Untack,
    Retack,
    TargetCell
  };
  QHyodaGdb(QHyodaJob*,QHyodaMachine*,quint32,quint32,quint32);
  ~QHyodaGdb(void);
public:
  bool launch(void);
  void enqueue(const QString & t);
  void dequeue(void);
public slots:
  void gdbmi(void);
public:
  void showViaEmacsClient(QString, QString);
public:
  QHyodaJob *job;
  QHyodaMachine *tab;
  quint32 tcpAdrs;
  quint32 tcpPort;
  quint32 tcpPyld;
  QProcess *process;
  QString cmdline;
  QQueue<QString> commands;
  QQueue<QString> outputs;
  QHyodaGdbCommand state;
  QString data_read_memory;
  QList<QString> entryPoints;
  QList<QString> entryPointsFile;
  QList<QString> entryPointsLine;
};

#endif // Q_HYODA_GDB_H
