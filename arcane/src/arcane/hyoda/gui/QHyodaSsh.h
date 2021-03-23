// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_SSH_H
#define Q_HYODA_SSH_H

#include <QThread>
#include <QProcess>

/******************************************************************************
 * Thread qui va gérer le process de port forwarding
 *****************************************************************************/

class QHyodaSsh: public QThread{
  Q_OBJECT
public:
  QHyodaSsh(QString,QString,quint32,quint32,bool=false);
  ~QHyodaSsh();
  void run();
  void setHost(QString);
  quint32 getTcpAdrs(void){return tcpAdrs;}
public slots:
  void tunnelReadyReadStandardOutput(void);
  void ceaHostReadyReadStandardOutput(void);
private:
  quint32 inetAton(const char *cp);
private:
  QString client_name;
  QString inter_adrs;
  QString inter_name;
  QString rank_zero;
  QProcess *tunnelProcess;
  QProcess *ceaHostProcess;
  bool via_interactive_hop;
  quint32 tcpAdrs;
  quint32 tcpPort;
  bool singleShot;
};

#endif // Q_HYODA_SSH_H
