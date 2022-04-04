// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * QHyodaTcp.h                                                      (C) 2012 *
 *****************************************************************************/
#ifndef Q_HYODA_TCP_H
#define Q_HYODA_TCP_H

#include <QObject>
#include <QtNetwork/QtNetwork>


class QHyodaJob;

class QHyodaTcp:public QObject {
  Q_OBJECT
public:
  //Mathematica: BaseForm [Hash[ "...", "CRC32"], 16]
  enum QHyodaTcpSwitch{
    Sleeping,
    MeshIceTHeader   = 0xcbce69bcul,
    MeshIceTImage    = 0x73491278ul,
    VariableName     = 0xca6cd6f0ul,
    HandShake        = 0xe73b2e9cul,
    Papi             = 0xb80dd1a3ul,
    MatrixIceTHeader = 0x78f78f67ul,
    MatrixIceTImage  = 0x2cd5e780ul
  };
public:
  QHyodaTcp(QHyodaJob*);
  ~QHyodaTcp();
public slots:
  void acceptConnection();
  void tcpReadyRead();
  void displayError(QAbstractSocket::SocketError socketError);
private:
  void tcpSleeping();
  void tcpHandShake();
  void tcpVariableName();
  void tcpMeshIceTHeader();
  void tcpMeshIceTImage();
  void tcpPapi();
  void tcpMatrixIceTHeader();
  void tcpMatrixIceTImage();
private:
  qint64 sendPacket(const char *data, qint64 maxSize);
  void sendAcknowledgePacket(void);
  void recvAcknowledgePacket(void);
private:
  QHyodaJob *job;
  QTcpServer tcpServer;
  QTcpSocket *tcpServerConnection;
  quint32 iceWdth;
  quint32 iceHght;
  quint32 tcpImageBytes;
  QImage *image,*matrix;
  QByteArray *byteArray,*matrixArray;
  QHyodaTcpSwitch state;
  quint32 tcpPacketLength;
};

#endif // Q_HYODA_TCP_H
