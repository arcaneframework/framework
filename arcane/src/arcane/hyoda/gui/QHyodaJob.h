// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_JOB_H
#define Q_HYODA_JOB_H

#include <QtWidgets>
#include <QStringList>

#include "QHyodaGdb.h"
#include "ui_hyodaJob.h"

class QHyodaTcp;
class QHyodaTool;
class QHyodaProcess;
class QHyodaMachine;
class QHyodaStateMachine;
class QHyodaSsh;


/******************************************************************************
 * Structure partagée par gdb
 *****************************************************************************/
struct hyoda_taxi_data{
  quint64 global_iteration;
  double global_time;
  double global_deltat;
  double global_cpu_time;
  quint64 mesh_total_nb_cell;
  quint64 target_cell_uid;
  quint64 target_cell_rank;
  double coords[8][3];
};

/**
 *
 **/
class QHyodaJob:public QWidget, public Ui::jobWidget {
  Q_OBJECT
public:
  QHyodaJob(QHyodaMachine*,int,char**);
  ~QHyodaJob();
public: 
  void refresh_common_variables(const QStringList&);
public:
  void gdbserver_hook();
  void gdb_hook();
  void detach();
  void quit();
public slots:
  void gdbserver_slot();
public slots:  // Toolbar buttons
  void stop_slot();
  void step_slot();
  void tack_slot();
  void start_slot();
  void log_slot();
  void src_slot();
  void cell_slot();
  void mesh_slot();
  void papi_slot();
  void matrix_slot();
public:
  QHyodaMachine *machine;
  int argc;
  char **argv;
  uint id;
  bool has_been_broadcasted;
  quint32 tcpAdrs;
  quint32 tcpPort;
  quint32 tcpPyld;
  quint32 iceWidthHeight;
  QHyodaGdb *gdb;
  QProcess *gdbserver;
  QString pid;
  QString cmd;
  QString cmdline;
  QString host;
  QString broadcasted_cmdline;  
  int tack_toggle;
  int fdm = 0;
  int fds = 0;
  // La structure partagée
  struct hyoda_taxi_data *data;
  QHyodaTcp *tcp;
  QStringList *arcane_variables_names;
};

#endif
