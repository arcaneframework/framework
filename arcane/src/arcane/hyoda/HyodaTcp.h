// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * HyodaTcp.h                                                 (C) 2000-2012 *
 *                                                                           *
 * Header du debugger hybrid.                                                *
 *****************************************************************************/
#ifndef ARCANE_HYODA_TCP_H
#define ARCANE_HYODA_TCP_H

#include <netdb.h>
#include <poll.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Hyoda;

/******************************************************************************
 * Hyoda CLASS
 *****************************************************************************/
class HyodaTcp: public TraceAccessor{
public:
  HyodaTcp(Hyoda*, ISubDomain*, ITraceMng*, Integer, Integer, Integer,bool=false);
  ~HyodaTcp();
public:
  void send(const void *,size_t);
  void recvPov(double*);
  void recvPov(double*,int);
  void waitForAcknowledgment(void);
  void sendAcknowledgmentPacket(void);
  void recvPacket(char *pov, int maxSize, int ms_timeout);
public:  
  void disconect(void);
  Integer payload(void){return m_payload;}
  void handshake(void);
  void getVariableCollectionAndSendItToHost(void);
private:
  ssize_t wData(int sockd, const void *vptr, size_t n);
  ssize_t rData(int sockd, void *vptr, size_t maxlen);
  int checkTcpError(int error);
private:
  Hyoda *hyoda;
  ISubDomain *m_sub_domain;
  size_t m_payload;
  int m_sockfd;
  struct sockaddr_in *m_servaddr;
  nfds_t m_nfds;
  struct pollfd *m_fds;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  // ARCANE_HYODA_TCP_H
