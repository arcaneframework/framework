// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * HyodaTcp.cc                                                 (C) 2012~2016 *
 *****************************************************************************/
#include <poll.h>
#include <errno.h>
#include <sys/socket.h>
#include <unistd.h>

#include "arcane/IMesh.h"
#include "arcane/IApplication.h"
#include "arcane/IParallelMng.h"
#include "arcane/IVariableMng.h"
#include "arcane/IApplication.h"
#include "arcane/FactoryService.h"
#include "arcane/ServiceFinder2.h"
#include "arcane/SharedVariable.h"
#include "arcane/CommonVariables.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/AbstractService.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IOnlineDebuggerService.h"
#include "arcane/ITransferValuesParallelOperation.h"
#include "arcane/VariableCollection.h"

#include "arcane/hyoda/HyodaArc.h"
#include "arcane/hyoda/HyodaTcp.h"
#define VARIABLE_PACKET_MAX_LENGTH 4*1024


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/******************************************************************************
 * HyodaTcp
 *****************************************************************************/
HyodaTcp::
HyodaTcp(Hyoda *_hyoda,ISubDomain *sd,ITraceMng *tm,Integer adrs,
         Integer port, Integer pyld, bool break_at_startup)
: TraceAccessor(tm)
, hyoda(_hyoda)
, m_sub_domain(sd)
, m_payload(pyld)
, m_sockfd(0)
, m_servaddr(nullptr)
, m_nfds(0)
, m_fds(nullptr)
{
  ARCANE_UNUSED(break_at_startup);
  if ((m_servaddr=(sockaddr_in *)malloc(sizeof(struct sockaddr_in)))==NULL)
    ARCANE_FATAL("Could not allocate data space for sockaddr");
  if ((m_fds=(pollfd *)malloc(sizeof(struct pollfd)))==NULL)
    ARCANE_FATAL("Could not allocate data space for pollfd");
  // Il y en a qu'un qui tente la connection
  if (m_sub_domain->parallelMng()->commRank()!=0) return;
  // internet protocol, pseudo protocol number
  if ((m_sockfd=socket(PF_INET, SOCK_STREAM, 0))<0)
    fatal()<<"[HyodaTcp::HyodaTcp] Error creating socket !";
  memset(m_servaddr, 0, sizeof(struct sockaddr_in));
  m_servaddr->sin_family=AF_INET;
  m_servaddr->sin_port=htons(port);
  //inet_aton("127.0.0.1", &m_servaddr->sin_addr);
  m_servaddr->sin_addr.s_addr=adrs;
  debug() << "[HyodaTcp::HyodaTcp] \33[7m @ " << m_servaddr->sin_addr.s_addr << " ?\33[m";
  if (checkTcpError(connect(m_sockfd, (struct sockaddr *) m_servaddr, sizeof(struct sockaddr_in)))!=0)
    fatal()<<"[HyodaTcp::HyodaTcp] Error connecting to socket!";
  // Poll preparation
  debug()<<"[HyodaTcp::HyodaTcp] \33[7mConnected!\33[m";
  memset(m_fds, 0, sizeof(struct pollfd));
  m_fds->fd=m_sockfd;
  m_fds->events=POLLIN;
  m_nfds=1;
  // Un packet de handshaking est envoyé
  handshake();
  // Puis un packet des variables
  getVariableCollectionAndSendItToHost();
}


/******************************************************************************
 * disconect
 *****************************************************************************/
void HyodaTcp::disconect(void){
  debug()<<"\33[7m[HyodaTcp::disconect]\33[m";
  //#warning HyodaTcp close should be done while disconnecting
  //if (m_sockfd!=0) ::close(m_sockfd);
}

// ******************************************************************************
// * Envoi des information de connection
// * BaseForm[Hash["handshake", "CRC32"], 16] = e73b2e9c
// *****************************************************************************
void HyodaTcp::handshake(void){
  int of7=4+4; // Header + taille
  char pkt[VARIABLE_PACKET_MAX_LENGTH];
 
  debug() << "[HyodaTcp::handshake]";
  // On écrit le header 'HandShake' de QHyodaTcpSwitch
  *(unsigned int*)&pkt[0]=0xe73b2e9cul;

  // On pousse notre pid
  *(unsigned int*)&pkt[of7]=platform::getProcessId();
  debug() << "[HyodaTcp::handshake] pid=" << platform::getProcessId();//<<", pid in packet="<<*(unsigned int*)&pkt[of7];
  of7+=4;

  // On pousse notre hostname
  // On écrit le nombre de caractères du hostname
  String hostname=platform::getHostName();//.localstr();
  *(unsigned int*)&pkt[of7]=hostname.len()+1;
  if (sprintf(&pkt[of7+4],"%s%c",hostname.localstr(),'\0')!=(hostname.len()+1))
    fatal()<<"Error pushing hostname into packet !";
  debug() << "[HyodaTcp::handshake] hostname=" << platform::getHostName();
  of7+=4+hostname.len()+1;
  
  // On trouve la ligne de commande, que l'on soit broadcasté ou pas
  String hyoda_bridge_broadcast_original_cmd=platform::getEnvironmentVariable("BRIDGE_BROADCAST_ORIGINAL_CMD");
  if (hyoda_bridge_broadcast_original_cmd!=NULL){
    debug() << "[HyodaTcp::handshake] Bridged from:"<<hyoda_bridge_broadcast_original_cmd;
    *(unsigned int*)&pkt[of7]=hyoda_bridge_broadcast_original_cmd.len()+1;
    if (sprintf(&pkt[of7+4], "%s%c",hyoda_bridge_broadcast_original_cmd.localstr(),'\0')
        !=(hyoda_bridge_broadcast_original_cmd.len()+1))
      fatal()<<"Error pushing hyoda_bridge_broadcast_original_cmd into packet !";
    of7+=hyoda_bridge_broadcast_original_cmd.len()+1;
  }else{
    const ApplicationInfo& app_info = hyoda->application()->applicationInfo();
    //int argc = *app_info.commandLineArgc();
    char** argv = *app_info.commandLineArgv();
    //for(int i=0;i<argc;++i) debug()<<argv[i];
    *(unsigned int*)&pkt[of7]=strlen(argv[0])+1;
    if (sprintf(&pkt[of7+4],"%s%c",argv[0],'\0')!=(strlen(argv[0])+1))
      fatal()<<"Error pushing commandLineArgv into packet !";
    debug() << "[HyodaTcp::handshake] command line:"<<argv[0];
    of7+=strlen(argv[0])+1;
  }
  of7+=4;
  
  // On récupère le SLURM_ID
  String slurm_job_id=platform::getEnvironmentVariable("SLURM_JOB_ID");
  if (slurm_job_id!=NULL){
    Integer sjobid=0;
    if (!builtInGetValue(sjobid,slurm_job_id)){
      if (m_sub_domain->parallelMng()->commRank()==0) debug()<<"\33[7m[Hyoda] slurm_job_id="<<sjobid<<"\33[m";
      debug() << "[HyodaTcp::handshake] SLURM_JOB_ID=" << sjobid;
      *(unsigned int*)&pkt[of7]=sjobid;
    }else{
      debug() << "[HyodaTcp::handshake] SLURM_JOB_ID but no builtInGetValue";
      *(unsigned int*)&pkt[of7]=0;
    }
  }else{
    debug() << "[HyodaTcp::handshake] not Slurm'ed";
    *(unsigned int*)&pkt[of7]=0;
  }
  of7+=4;
  
  // On pousse la taille du paquet
  *(unsigned int*)&pkt[4]=of7;

  // On envoi le packet
   send(pkt,of7);
   
   // Et on attend la réponse de QHyoda
   waitForAcknowledgment(); // venant du QHyodaTcp::Sleeping
   //waitForAcknowledgment(); // puis du QHyodaTcp::HandShake
}


/******************************************************************************
 * Récupération de la liste des variables affichables
 *****************************************************************************/
void HyodaTcp::
getVariableCollectionAndSendItToHost(void)
{
  int varPacketOffset=4+4;
  char varPacket[VARIABLE_PACKET_MAX_LENGTH];
  VariableCollection variables = m_sub_domain->variableMng()->usedVariables();
  // On écrit le 'VariableName' QHyodaTcpSwitch
  *(unsigned int*)&varPacket[0]=0xca6cd6f0ul;
  debug()<< "Variables count=" << variables.count();
  for(VariableCollection::Enumerator ivar(variables); ++ivar; ){
    IVariable* var = *ivar;
    debug() << "[HyodaTcp::getVariableCollectionAndSendItToHost]"
            << "Focusing variable"
            << "name=" << var->name().localstr();
    // Pas de références, pas de variable
    if (var->nbReference()==0) {debug() << "No nbReference"; continue;}
    // Pas sur le bon support, pas de variable
    if (var->itemKind()!=IK_Node &&
        var->itemKind()!=IK_Cell &&
        var->itemKind()!=IK_Face &&
        var->itemKind() != IK_Particle) continue;
    // Pas réclamée en tant que PostProcess'able, pas de variable
    if (var->itemKind()!=IK_Particle && (!var->hasTag("PostProcessing")))
      {debug() << "No PostProcessing"; continue;}
    // Pas de type non supportés
    //if (var->dataType()>=DT_String) continue;
    if (var->dataType()!=DT_Real) continue;
    debug() << "[HyodaTcp::getVariableCollectionAndSendItToHost] Found variable"
           << " name=" <<var->name().localstr()
           << ", dataType=" <<dataTypeName(var->dataType())
           << ", fullName=" <<var->fullName()
           << ", family=" <<var->itemFamilyName()
           << ", isUsed=" << var->isUsed()
           << ", nbReference=" << var->nbReference()
           << ", dimension=" << var->dimension()
           << ", isPartial=" <<var->isPartial()
           << ", var.hasTag(\"PostProcessing\")"<<var->hasTag("PostProcessing");
    // On écrit le nombre de caractères du nom de la variable
    //debug() << "[getVariableCollectionAndSendItToHost] var->name().len()+1="<<var->name().len()+1;
    *(unsigned int*)&varPacket[varPacketOffset]=var->name().len()+1;
    // On écrit le nom de la variable
    debug() << "[HyodaTcp::getVariableCollectionAndSendItToHost] \33[7mVariable: "<<var->name()<<"\33[m";
    if (sprintf(&varPacket[varPacketOffset+4],"%s%c",var->name().localstr(),'\0')!=(var->name().len()+1))
      fatal()<<"Error pushing variable name into packet !";
    ARCANE_ASSERT(varPacketOffset+4+var->name().len()+1<VARIABLE_PACKET_MAX_LENGTH, ("VARIABLE_PACKET_MAX_LENGTH"));
    varPacketOffset+=4+var->name().len()+1;
    //debug() << "number_of_characters_printed="<<number_of_characters_printed;
    //for(int i=0;i<varPacketHeaderLen;++i) debug()<<"\tvariablePacket["<<i<<"]="<<varPacketHeader[i];
  }
  // On pousse la taille du paquet
  *(unsigned int*)&varPacket[4]=varPacketOffset;
  //debug() << "[getVariableCollectionAndSendItToHost] varPacketLength="<<varPacketOffset<<"o";
  if (m_sub_domain->parallelMng()->commRank()==0)
    send(varPacket,varPacketOffset);
  waitForAcknowledgment(); // venant du QHyodaTcp::VariableNameleeping
}




/******************************************************************************
 * send
 *****************************************************************************/
void HyodaTcp::
send(const void *data, size_t nleft)
{
  size_t payload = m_payload;
  size_t offset = 0;
  debug() << "\33[7m[HyodaTcp::send] sending "<< nleft << " bytes of data, payload=" <<m_payload << "\33[m";
  while (nleft>0){
    if (nleft<payload) payload=nleft;
    if (wData(m_sockfd, (unsigned char*)data+offset, payload)!=payload)
      fatal()<<"Error sending into socket !";
    nleft  -= payload;
    offset += payload;
  }
  ARCANE_ASSERT(nleft==0, ("[send] nleft!=0"));
  debug() << "\33[7m[HyodaTcp::send] done\33[m";
  ::fsync(m_sockfd);
}


/******************************************************************************
 * Write a line to a socket 
 *****************************************************************************/
ssize_t HyodaTcp::
wData(int sockd, const void *vptr, size_t n)
{
  size_t nleft=n;
  ssize_t nwritten;
  const char *buffer=(char*)vptr;
  while (nleft>0) {
    if ((nwritten = write(sockd, buffer, nleft))<=0){
      if (errno==EINTR) nwritten=0;
      else return -1;
    }
    nleft  -= nwritten;
    buffer += nwritten;
  }
  return n;
}


// ******************************************************************************
// * recv
// * The timeout argument specifies an upper limit on the time for which poll()
// * will block, in milliseconds.
// * Specifying a negative value in timeout means an infinite timeout.
// *****************************************************************************
void HyodaTcp::recvPov(double *pov){
  recvPacket((char*)pov, 8*(1+3+1+1), -1);
  debug() << "\33[7m[HyodaTcp::recvPov] ok\33[m";
  //sendAcknowledgmentPacket();
}

void HyodaTcp::recvPov(double *pov, int ms_timeout){
  recvPacket((char*)pov, 8*(1+3+1+1), ms_timeout);
  debug() <<"\33[7m[HyodaTcp::recv] pov "
         << " scale=" << pov[0]
         << " rot_x=" << pov[1]
         << " rot_y=" << pov[2]
         << " rot_z=" << pov[3]
         << " idx="   << pov[4]
         << " plg="   << pov[5]
         <<"\33[m";
}

void HyodaTcp::sendAcknowledgmentPacket(void){
  // Il faut au moins 8o pour que le Sleeping state se déclenche
  char ack[8];
  *(unsigned int*)&ack[0]=0x3e9ff203ul;
  *(unsigned int*)&ack[4]=0;
  debug() << "\33[7m[HyodaTcp::sendAcknowledgmentPacket] ...\33[m";
  send(ack, 8);
  debug() << "\33[7m[HyodaTcp::sendAcknowledgmentPacket] !\33[m";
}

void HyodaTcp::waitForAcknowledgment(void){
  char ack[4];
  debug() << "\33[7m[HyodaTcp::waitForAcknowledgment] ?\33[m";
  recvPacket((char*)&ack[0], 4, -1);
  debug() << "\33[7m[HyodaTcp::waitForAcknowledgment] !\33[m";
  if ((*(unsigned int*)&ack[0]) != 0x3e9ff203ul)
    fatal() << "HyodaTcp::waitForAcknowledgment, ack[0]="<<*(unsigned int*)&ack[0];
  debug() << "\33[7m[HyodaTcp::waitForAcknowledgment] ok\33[m";
}

void HyodaTcp::recvPacket(char *pov, int maxSize, int ms_timeout){
  int returned_events;
  debug() << "\33[7m[HyodaTcp::recvPacket] ...\33[m";
  do{
    returned_events=poll(m_fds, m_nfds, ms_timeout);
  } while(returned_events==-1 && errno == EINTR);
  if (returned_events==-1) fatal()<<"Error polling socket !";
  if (returned_events==0) debug()<<"Timeout polling socket !";
  if (!(m_fds->revents && POLLIN)) return;
  rData(m_sockfd, (char*)&pov[0], maxSize);
  debug() << "\33[7m[HyodaTcp::recvPacket] !\33[m";
}


/******************************************************************************
 * Read a line from a socket 
 *****************************************************************************/
ssize_t HyodaTcp::rData(int sockd, void *vptr, size_t maxlen){
  ssize_t n, rc;
  char c, *buffer=(char*)vptr;
  for(n=0; n<maxlen; n++){
    if ((rc = read(sockd, &c, 1))==1){
      //debug() << "got char '" << c <<"'";
      *buffer++ = c;
    }
    else if (rc==0) { // end of file
      //debug() << "eof";
      if (n==0){
        //debug() << "n==0, returning 0";
        return 0;
      }
      else break;
    }
    else {
      if (errno==EINTR){
        //debug() << "EINTR, continue";
        continue; // The call was interrupted by a signal before any data was read.
      }
      //debug() << "returning -1";
      return -1;
    }
  }
  // Attention, dans le cas exact, il ne faut pas déborder!
  //*buffer = 0;
  return n;
}


/******************************************************************************
 * checkTcpError
 *****************************************************************************/
int HyodaTcp::checkTcpError(int error){
  if (error>=0) return error;
  switch (errno) {
  case EACCES:
    debug() << "\33[7m" << "EACCES" << "\33[m:" << "Write permission is denied on the socket.";
    break;
  case EPERM:
    debug() << "\33[7m" << "EPERM" << "\33[m:"
           << "The  user tried to connect to a broadcast address without having the socket broadcast flag enabled"
           << "or the connection request failed because of a local firewall rule.";
    break;
  case EADDRINUSE:
    debug() << "\33[7m" << "EADDRINUSE" << "\33[m:"
           << "Local address is already in use.";
    break;
  case EAFNOSUPPORT:
    debug() << "\33[7m" << "EAFNOSUPPORT" << "\33[m:"
           <<"The passed address didn't have the correct address family in its sa_family field.";
    break;
  case EADDRNOTAVAIL:
    debug() << "\33[7m" << "EADDRNOTAVAIL" << "\33[m:"
           << "Non-existent interface was requested or the requested address was not local.";
    break;
  case EALREADY:
    debug() << "\33[7m" << "EALREADY" << "\33[m:"
           << "The socket is non-blocking and a previous connection attempt has not yet been completed.";
    break;
  case EBADF:
    debug() << "\33[7m" << "EBADF" << "\33[m:"
           << " The file descriptor is not a valid index in the descriptor table.";
    break;
  case ECONNREFUSED:
    debug() << "\33[7m" << "ECONNREFUSED" << "\33[m:"
           << "No one listening on the remote address.";
    break;
  case EFAULT:
    debug() << "\33[7m" << "EFAULT" << "\33[m:"
           <<"The socket structure address is outside the user's address space.";
    break;
  case EINPROGRESS:
    debug() << "\33[7m" << "EINPROGRESS" << "\33[m:"
           <<"The socket is non-blocking and the connection cannot be completed immediately.";
    break;
  case EINTR:
    debug() << "\33[7m" << "EINTR" << "\33[m:"
           << "The system call was interrupted by a signal that was caught.";
    break;
  case EISCONN:
    debug() << "\33[7m" << "EISCONN" << "\33[m:"
           <<"The socket is already connected.";
    break;
  case ENETUNREACH:
    debug() << "\33[7m" << "ENETUNREACH" << "\33[m:"
           << "Network is unreachable.";
    break;
  case ENOTSOCK:
    debug() << "\33[7m" << "ENOTSOCK" << "\33[m:"
           << "The file descriptor is not associated with a socket.";
    break;
  case ETIMEDOUT:
    debug() << "\33[7m" << "ETIMEDOUT" << "\33[m:"
           <<"Timeout while attempting connection.";
  default: debug()<<"## UNKNOWN ERROR CODE error="<<error<<", errno="<<errno;
  }
  return error;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
