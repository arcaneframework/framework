// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <QHyodaTcp.h>
#include <QHyodaMachine.h>
#include <QHyodaToolMesh.h>
#include <QHyodaToolMatrix.h>
#include <QHyodaPapi.h>

// ****************************************************************************
// * QHyodaTcp class
// ****************************************************************************
QHyodaTcp::
QHyodaTcp(QHyodaJob *jb)
: job(jb)
, tcpServerConnection(nullptr)
, iceWdth((job->iceWidthHeight>>16)&0xFFFFul)
, iceHght((job->iceWidthHeight)&0xFFFFul)
, tcpImageBytes(iceWdth*iceHght*sizeof(uint32_t))
, image(nullptr)
, matrix(nullptr)
, byteArray(nullptr)
, matrixArray(nullptr)
, state(QHyodaTcp::Sleeping)
, tcpPacketLength(0)
{
  connect(&tcpServer, SIGNAL(newConnection()), this, SLOT(acceptConnection()));
  while (!tcpServer.isListening() &&
         !tcpServer.listen(QHostAddress::Any,job->tcpPort));//usleep(100);
  qDebug() << "\33[37m[QHyodaTcp::QHyodaTcp] Listening @ port" << tcpServer.serverPort()<<"\33[m";
}

// ****************************************************************************
// * ~QHyodaTcp
// ****************************************************************************
QHyodaTcp::~QHyodaTcp(){
  qDebug() << "\33[37m[QHyodaTcp::~QHyodaTcp]\33[m";
  if (tcpServerConnection){
    tcpServerConnection->close();
    delete tcpServerConnection;
  }
  tcpServer.close();
  delete image;
  //free(data);
}

// ****************************************************************************
// * acceptConnection
// ****************************************************************************
void QHyodaTcp::acceptConnection(){
  // On arrive là lorsque HyodaArc créé son HyodaTcp
  qDebug() << "\33[37m[QHyodaTcp::acceptConnection] NEW Connection detected!\33[m";
  tcpServerConnection = tcpServer.nextPendingConnection();
  //Seeks to the start of input for random-access devices
  //tcpServerConnection->reset();
  connect(tcpServerConnection, SIGNAL(readyRead()),this,SLOT(tcpReadyRead()));
  connect(tcpServerConnection, SIGNAL(error(QAbstractSocket::SocketError)),
          this, SLOT(displayError(QAbstractSocket::SocketError)));
  tcpServer.close();
}

// ****************************************************************************
// * tcpSleeping
// ****************************************************************************
void QHyodaTcp::tcpSleeping(){
  char pktHeader[8];
  // On attend au moins le QHyodaTcpSwitch de 4+4 octets
  if (tcpServerConnection->bytesAvailable() < 8) return;
  //qDebug()<<"\33[37m[QHyodaTcp::tcpSleeping] Sleeping\33[m";
  // On lit alors le header
  if (tcpServerConnection->read(pktHeader,8)!=8)
    qFatal("\33[37m[QHyodaTcp::tcpReadyRead] Could not fetch packet header!\33[m");
  // On récupère la taille tout de suite
  tcpPacketLength=*(unsigned int*)&pktHeader[4];
  //qDebug()<<"[QHyodaTcp] tcpPacketLength="<<tcpPacketLength;
  //qDebug()<<"[QHyodaTcp] Header code="<<*(unsigned int*)&pktHeader[0]<<", tcpPacketLength="<<tcpPacketLength;
  if (*(unsigned int*)&pktHeader[0]==0xe73b2e9cul) state=QHyodaTcp::HandShake;
  if (*(unsigned int*)&pktHeader[0]==0xca6cd6f0ul) state=QHyodaTcp::VariableName;
  if (*(unsigned int*)&pktHeader[0]==0xcbce69bcul) state=QHyodaTcp::MeshIceTHeader;
  if (*(unsigned int*)&pktHeader[0]==0x73491278ul) state=QHyodaTcp::MeshIceTImage;
  if (*(unsigned int*)&pktHeader[0]==0xb80dd1a3ul) state=QHyodaTcp::Papi;
  if (*(unsigned int*)&pktHeader[0]==0x78f78f67ul) state=QHyodaTcp::MatrixIceTHeader;
  if (*(unsigned int*)&pktHeader[0]==0x2cd5e780ul) state=QHyodaTcp::MatrixIceTImage;
  // et on reprovoque pour aller traiter ce qu'on a éventuellement switché
  tcpReadyRead();
}

// ****************************************************************************
// * tcpHandShake
// ****************************************************************************
void QHyodaTcp::tcpHandShake(){
  char data[1024];
  quint32 pid;
  const int tcpPacketLengthLeft=tcpPacketLength-8;
  if (tcpServerConnection->bytesAvailable() < tcpPacketLengthLeft) return;
  qDebug()<<"\33[37m[QHyodaTcp::tcpHandShake] QHyodaTcp::HandShake\33[m";
  // On récupère le pid
  qDebug()<<"\33[37m[QHyodaTcp::tcpHandShake] @ HandShake: On recupere le pid\33[m";
  if (tcpServerConnection->read(data,4)!=4)
    qFatal("\33[37m[QHyodaTcp::tcpHandShake] Could not fetch pid!\33[m");
  pid=*(unsigned int*)&data[0];
  qDebug()<<"\33[37m[QHyodaTcp::tcpHandShake] \33[7m@ HandShake: pid="<<pid<<"\33[m";
  job->pid=QString("%1").arg(pid);
  // On récupère le nom de la machine
  if (tcpServerConnection->read(data,4)!=4)
    qFatal("\33[37m[QHyodaTcp::tcpHandShake] Could not fetch hostname len!\33[m");
  unsigned int len=*(unsigned int*)&data[0];
  qDebug()<<"\33[37m[QHyodaTcp::tcpHandShake] @ HandShake: hostname's len="<<len<<"\33[m";
  if (tcpServerConnection->read(data,len)!=len)
    qFatal("\33[37m[QHyodaTcp] Could not fetch hostname!\33[m");
  qDebug()<<"\33[37m[QHyodaTcp::tcpHandShake] \33[7m@ HandShake: rankZero is"<<data<<"\33[m";
  job->host=QString(data);
  // On récupère le nom de la ligne de commande
  if (tcpServerConnection->read(data,4)!=4)
    qFatal("\33[37m[QHyodaTcp::tcpHandShake] Could not fetch command line len!\33[m");
  len=*(unsigned int*)&data[0];
  qDebug()<<"\33[37m[QHyodaTcp::tcpHandShake] @ HandShake: COMMAND LINE's len="<<len<<"\33[m";
  if (tcpServerConnection->read(data,len)!=len)
    qFatal("\33[37m[QHyodaTcp] Could not fetch COMMAND LINE!\33[m");
  qDebug()<<"\33[37m[QHyodaTcp::tcpHandShake] \33[7m@ HandShake: COMMAND LINE is"<<data<<"\33[m";
  job->broadcasted_cmdline=QString(data);
  job->has_been_broadcasted=true;
  // On récupère le SLURM_JOB_ID
  if (tcpServerConnection->read(data,4)!=4)
    qFatal("\33[37m[QHyodaTcp::tcpHandShake] Could not fetch SLURM_JOB_ID!\33[m");
  job->id=*(unsigned int*)&data[0];
  qDebug()<<"\33[37m[QHyodaTcp::tcpHandShake] \33[7m@ HandShake: SLURM_JOB_ID is"<<job->id<<"\33[m";
  // On acknowlege le handshake
  sendAcknowledgePacket();
  // Si on s'est hand-shaké, on se met en doze
  state=QHyodaTcp::Sleeping;
}

// ****************************************************************************
// * tcpVariableName
// ****************************************************************************
void QHyodaTcp::tcpVariableName(){
  char varName[4*1024];
  const int tcpPacketLengthLeft=tcpPacketLength-8;
  qDebug()<<"\33[37m[QHyodaTcp::tcpVariableName] state @ QHyodaTcp::\33[7mVariableName\33[m";
  // Tant qu'on a pas tout le nom de la variable qui est arrivé, on à rien à faire
  if (tcpServerConnection->bytesAvailable() < tcpPacketLengthLeft) return;
  qDebug()<<"\33[37m[QHyodaTcp::tcpVariableName] QHyodaTcp::VariableName\33[m";
  // On récupère les noms des variables
  if (tcpServerConnection->read(varName,tcpPacketLengthLeft)!=tcpPacketLengthLeft)
    qFatal("\33[37m[QHyodaTcp::tcpVariableName] Could not fetch variable name!\33[m");
  for(int offset=0;offset<tcpPacketLengthLeft;){
    //int var_name_len=varName[offset];
    unsigned int var_name_len=*(unsigned int*)&varName[offset]; // 4o
    // On saute les 4o de len
    offset+=4;
    //qDebug()<<"\t[QHyodaTcp::tcpReadyRead]"<<var_name_len<<" bytes for this name";
    qDebug()<<"\33[37m\t[QHyodaTcp::tcpVariableName] Variable \33[7m"<<&varName[offset]<<"\33[m";
    // On rajoute cette variable arcane à la liste des noms
    *job->arcane_variables_names<<&varName[offset];
    //job->topRightTools->mesh->variablesComboBox->addItem(&varName[offset]);
    //qDebug()<<"[QHyodaTcp::tcpReadyRead] arcane_variables_names are now:"<<*job->arcane_variables_names;
    offset+=var_name_len;
  }
  // Et on retourne en mode doze
  state=QHyodaTcp::Sleeping;
  // Maintenant qu'on a tout récupéré, on accroche gdb
  qDebug()<<"\33[37m[QHyodaTcp::tcpVariableName] Maintenant qu'on a tout recupere, on accroche GDB\33[m";
  //job->machine->tabJobs->setTabText(job->machine->tabJobs->currentIndex(),job->host);
  job->gdbserver_hook();
  // On acknowledge le getVariableCollectionAndSendItToHost
  //qDebug()<<"\33[7m[QHyodaTcp::tcpReadyRead] On acknowledge le getVariableCollectionAndSendItToHost\33[m";
  sendAcknowledgePacket();
}

// ****************************************************************************
// * tcpMeshIceTHeader
// ****************************************************************************
void QHyodaTcp::tcpMeshIceTHeader(){
  //qDebug()<<"\33[7m[QHyodaTcp::tcpMeshIceTHeader]\33[m";
  sendAcknowledgePacket();
  state=QHyodaTcp::MeshIceTImage;
}

// ****************************************************************************
// * tcpMeshIceTImage
// ****************************************************************************
void QHyodaTcp::tcpMeshIceTImage(){
  // Tant qu'on a pas tout reçu, on revient plus tard
  if (tcpServerConnection->bytesAvailable() < tcpImageBytes) return;
  //qDebug()<<"\33[7m[QHyodaTcp::tcpMeshIceTImage]\33[m";
  // On fait le ménage avant de récupérer la nouvelle image
  if (byteArray) delete byteArray;
  byteArray=new QByteArray(tcpServerConnection->read(tcpImageBytes));
  sendAcknowledgePacket();
  // On a tout reçu, on peut envoyer une réponse
  // On va envoyer le POV et l'index de la variable
  double pov[6]={0.,0.,0.,0.,0.,0.};
  if (job->topRightTools->mesh!=NULL){
    job->topRightTools->mesh->ice->sxyz(&pov[0]);
    pov[4]=job->topRightTools->mesh->variablesComboBox->currentIndex();
    pov[5]=job->topRightTools->mesh->hPluginComboBox->currentIndex();
  }
  sendPacket((char*)&pov[0],8*(4+1+1));
  recvAcknowledgePacket();
  // On retourne par défaut en mode doze dès qu'on a reçu une image
  state=QHyodaTcp::Sleeping;
  // Tant qu'on a pas demandé le tab QHyodaToolMesh, on ne fait rien
  if (job->topRightTools->mesh==NULL) return;
  // Tant qu'on a pas cliqué sur le bouton
  if (job->meshButton->isEnabled()) return;
  // On recréé l'image à chaque fois
  if (image!=NULL)  delete image;
  image=new QImage((uchar*)byteArray->data(), iceWdth, iceHght, QImage::Format_ARGB32);// ARGB32 vs RGBA8888
  job->topRightTools->mesh->ice->setImage(image);
  // Et on  update le GL
  //job->topRightTools->mesh->ice->update();
}

// ****************************************************************************
// * tcpPapi
// ****************************************************************************
void QHyodaTcp::tcpPapi(){  
  const int bytesAvailable=tcpServerConnection->bytesAvailable();
  if (bytesAvailable < 8) return;
  //qDebug()<<"\33[37m[QHyodaTcp::tcpPapi] Papi #"<<bytesAvailable<<"\33[m";
  //qDebug()<<"\33[37m[QHyodaTcp::tcpPapi] Read PAPI input!\33[m";
  // Et on  update le profiling s'il a été initialisé
  if (job->bottomRightTools->papi)
    job->bottomRightTools->papi->update(new QByteArray(tcpServerConnection->read(bytesAvailable)));
  sendAcknowledgePacket();
  state=QHyodaTcp::Sleeping;
}

// ****************************************************************************
// * tcpMatrixIceTHeader
// ****************************************************************************
void QHyodaTcp::tcpMatrixIceTHeader(){
  //qDebug()<<"\33[37m[QHyodaTcp::tcpMatrixIceTHeader]\33[m";
  sendAcknowledgePacket();
  state=QHyodaTcp::MatrixIceTImage;
}

// ****************************************************************************
// * tcpMatrixIceTImage
// ****************************************************************************
void QHyodaTcp::tcpMatrixIceTImage(){
  // Tant qu'on a pas tout reçu, on revient plus tard
  if (tcpServerConnection->bytesAvailable() < tcpImageBytes) return;
  //qDebug()<<"\33[37m[QHyodaTcp::tcpMatrixIceTImage]\33[m";
  // On fait le ménage avant de récupérer la nouvelle image
  if (matrixArray) delete matrixArray;
  matrixArray=new QByteArray(tcpServerConnection->read(tcpImageBytes));
  //qDebug()<<"\33[37m[QHyodaTcp::tcpReadyRead] sending AcknowledgePacket\33[0m";
  sendAcknowledgePacket();
  // On a tout reçu, on peut envoyer une réponse, on va envoyer le POV
  double pov[6]={0.,0.,0.,0.,0.,0.};
  if (job->topLeftTools){
    if (job->topLeftTools->matrix!=NULL){
      job->topLeftTools->matrix->ice->sxyz(&pov[0]);
    }
  }
  //qDebug()<<"\33[37m[QHyodaTcp::tcpReadyRead] sending POV Packet\33[0m";
  sendPacket((char*)&pov[0],8*(4+1+1));
  //qDebug()<<"\33[37m[QHyodaTcp::tcpReadyRead] waiting for recvAcknowledgePacket\33[0m";
  recvAcknowledgePacket();
  // On retourne par défaut en mode doze dès qu'on a reçu une image
  state=QHyodaTcp::Sleeping;
  // On indique que l'on a une matrice à visualiser
  job->matrixButton->setEnabled(true);
  // Tant qu'on a pas demandé le tab QHyodaToolMesh, on ne fait rien
  if (job->topLeftTools->matrix==NULL) return;
  // Pour l'instant, on recréé l'image à chaque fois: cela permet d'éviter le 'shift' visuel
  if (matrix!=NULL) delete matrix;
  matrix=new QImage((uchar*)matrixArray->data(), iceWdth, iceHght, QImage::Format_ARGB32);
  job->topLeftTools->matrix->ice->setImage(matrix);
  // Et on  update le GL
  //job->topLeftTools->matrix->ice->updateGL();
}

// ****************************************************************************
// * tcpReadyRead
// ****************************************************************************
void QHyodaTcp::tcpReadyRead(){
  //qDebug()<<"\33[37m[QHyodaTcp::tcpReadyRead] switch\33[m";
  switch (state){
  case (QHyodaTcp::Sleeping):{tcpSleeping();break;}    
  case (QHyodaTcp::HandShake):{tcpHandShake();break;}
  case (QHyodaTcp::VariableName):{tcpVariableName();break;}
  case (QHyodaTcp::MeshIceTHeader):{tcpMeshIceTHeader();break;}
  case (QHyodaTcp::MeshIceTImage):{tcpMeshIceTImage();break;}
  case (QHyodaTcp::Papi):{tcpPapi();break;}
  case (QHyodaTcp::MatrixIceTHeader):{tcpMatrixIceTHeader();break;}
  case (QHyodaTcp::MatrixIceTImage):{tcpMatrixIceTImage();break;}
  default: qFatal("\33[37m[QHyodaTcp::tcpReadyRead] Unknown state!\33[m");
  }
}

// ****************************************************************************
// * displayError
// ****************************************************************************
void QHyodaTcp::displayError(QAbstractSocket::SocketError socketError){
  if (socketError == QTcpSocket::RemoteHostClosedError) return;
  qDebug() << "\33[37m[QHyodaTcp::displayError] \33[7mSocket error"<<socketError<<"\33[m";
  tcpServer.close();
}

// ****************************************************************************
// * sendPacket
// ****************************************************************************
qint64 QHyodaTcp::sendPacket(const char *data, qint64 maxSize){
  //qDebug() << "\33[7m[QHyodaTcp::sndPacket]\33[m";
  if (tcpServerConnection->write(data,maxSize)!=maxSize)
    qFatal("\33[37m[QHyodaTcp::sendPacket] has not sent maxSize bytes!\33[m");
  return 0;
}


// ****************************************************************************
// * sndAcknowledgePacket
// * BaseForm [Hash[ "Acknowledge", "CRC32"], 16] = 0x3e9ff203
// ****************************************************************************
void QHyodaTcp::sendAcknowledgePacket(void){
  char pkt[4];
  *(unsigned int*)&pkt[0]=0x3e9ff203ul;
  //qDebug() << "\33[37m[QHyodaTcp::sendAcknowledgePacket] \33[m";
  sendPacket((char*)&pkt[0],4);
}


// ****************************************************************************
// * recvAcknowledgePacket
// ****************************************************************************
void QHyodaTcp::recvAcknowledgePacket(void){
  char pktHeader[4];
  if (tcpServerConnection->bytesAvailable() < 4) return;
  //qDebug()<<"\33[37m[QHyodaTcp::recvAcknowledgePacket]\33[m";
  if (tcpServerConnection->read(pktHeader,4)!=4)
    qFatal("\33[37m[QHyodaTcp::recvAcknowledgePacket] Could not fetch packet header!\33[m");
  //qDebug()<<"[QHyodaTcp] Header code="<<*(unsigned int*)&pktHeader[0];
  if (*(unsigned int*)&pktHeader[0]==0x3e9ff203ul) return;
  //qFatal("\33[37m[QHyodaTcp::recvAcknowledgePacket] Not an ACK packet!");
}
