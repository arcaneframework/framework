// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <QtWidgets>
#include <QHyodaSsh.h>
#include <arpa/inet.h>

/******************************************************************************
 * QHyodaSsh
 *****************************************************************************/
QHyodaSsh::QHyodaSsh(QString interactive,
                     QString rankZero,
                     quint32 adrs,
                     quint32 port,
                     bool _singleShot): client_name(QProcessEnvironment::systemEnvironment().value("HOSTNAME")),
                                       inter_name(interactive),
                                       inter_adrs(QString()),
                                       rank_zero(rankZero),
                                       tunnelProcess(new QProcess()),
                                       ceaHostProcess(new QProcess()),
                                       via_interactive_hop(false),
                                       tcpAdrs(adrs),
                                       tcpPort(port),
                                       singleShot(_singleShot){
  qDebug() << "[QHyodaSsh] NEW";
  qDebug() << "[QHyodaSsh] interactive:"<<interactive;
  qDebug() << "[QHyodaSsh] rankZero:"<<rankZero;
    
  via_interactive_hop=(   inter_name.startsWith("germain", Qt::CaseInsensitive)
                       || inter_name.startsWith("cartan",  Qt::CaseInsensitive)
                       || inter_name.startsWith("lascaux", Qt::CaseInsensitive)) ? true : false;

  // Dans le cas d'un single shot, il n'y a pas de hop 
  if (singleShot) via_interactive_hop=false;
  
  // Dans tous les cas, on peut aller chercher le nom du noeud interactif
  ceaHostProcess->setProcessChannelMode(QProcess::MergedChannels);
  connect(ceaHostProcess, SIGNAL(readyReadStandardOutput()), this, SLOT(ceaHostReadyReadStandardOutput()));
  if (!singleShot)
    ceaHostProcess->start(QString("/usr/local/sr/bin/cea_gethost"), QStringList()<<interactive);
  else
    ceaHostProcess->start(QString("/usr/bin/host"), QStringList()<<inter_name);
  
  if (!ceaHostProcess->waitForStarted())
    qFatal("QHyodaSsh::run threadProcess NOT started!");
}


/******************************************************************************
 * ~QHyodaSsh 
 *****************************************************************************/
QHyodaSsh::~QHyodaSsh(){
  qDebug() << "~QHyodaSsh";
  tunnelProcess->close();
  ceaHostProcess->close();
  if (tunnelProcess->state()!=QProcess::NotRunning)
    qFatal("QHyodaSsh tunnelProcess NOT closed!");
  delete tunnelProcess;
}


/******************************************************************************
 * QHyodaSsh::run
 *****************************************************************************/
void QHyodaSsh::run(){
  if (isRunning()) return;

  ceaHostProcess->close();

  // Si on a pas besoin d'un tunnel
  if (!via_interactive_hop){
    // Il reste juste à troquer le localhost 127.0.0.1 du défaut par la station visée
    tcpAdrs=inetAton(inter_adrs.toLatin1().data());
    qDebug()<<"\33[1m[QHyodaSsh::run] tcpAdrs="<<tcpAdrs<<"\33[m";
    // En single shot, on n'a pas fini!, il faut un tunnel
    if (!singleShot) return;
  }
  

  if (singleShot){
    QString command("ssh");
    QStringList args = QStringList() << "-xCTR"
                                     << QString("%1:%2:%1").arg(QString().setNum(tcpPort),inter_adrs)
                                     << rank_zero.toLower()
                                     << "-N";
    //qDebug() << vt100_reverse << command << args << vt100_modesreset;
    // Lancement du process tunnel
    tunnelProcess->setProcessChannelMode(QProcess::MergedChannels);
    connect(tunnelProcess, SIGNAL(readyReadStandardOutput()), this, SLOT(tunnelReadyReadStandardOutput()));
    qDebug()<<"\33[1m[QHyodaSsh::run] singleShot tunnelProcess command="<<command<<", args="<<args<<"\33[m";
    tunnelProcess->start(command, args);
    if (!tunnelProcess->waitForStarted())
      qFatal("QHyodaSsh::run tunnelProcess NOT started!");
    return;
  }
  
  // Tunnel standard
  QString command("ssh");
  QStringList args = QStringList() << "-xCTR" // v
                                   << QString("%1:%2:%1").arg(QString().setNum(tcpPort),client_name)
                                   << inter_adrs
                                   << "ssh"
                                   << "-xCTR"
                                   << QString("%1:%2:%1").arg(QString().setNum(tcpPort),inter_adrs)
                                   << rank_zero.toLower()
                                   << "-N";
  qDebug() << "\33[7m" << command << args << "\33[m";
  // Lancement du process tunnel
  tunnelProcess->setProcessChannelMode(QProcess::MergedChannels);
  connect(tunnelProcess, SIGNAL(readyReadStandardOutput()), this, SLOT(tunnelReadyReadStandardOutput()));
  tunnelProcess->start(command, args);
  if (!tunnelProcess->waitForStarted())
    qFatal("QHyodaSsh::run tunnelProcess NOT started!");
  //Pas d'exec();
}


/******************************************************************************
 * QHyodaSsh::readyReadStandardOutput
 *****************************************************************************/
void QHyodaSsh::tunnelReadyReadStandardOutput(void){
  const QStringList read_lines=QString(tunnelProcess->readAllStandardOutput()).split(QRegExp("\n"));
  for(int i=0,mx=read_lines.size();i<mx;++i){
    const QString line=read_lines.at(i).trimmed();
    if (line.isEmpty()) continue;
    qDebug()<<"\t\t"<<"\33[1m"<<line<<"\33[m";
  }
}


/******************************************************************************
 * QHyodaSsh::readyReadStandardOutput
 *****************************************************************************/
void QHyodaSsh::ceaHostReadyReadStandardOutput(void){
  const QStringList read_lines=QString(ceaHostProcess->readAllStandardOutput()).split(QRegExp("\n"));
  qDebug()<<read_lines;
  for(int i=0,mx=read_lines.size();i<mx;++i){
    const QString line=read_lines.at(i).trimmed();
    if (line.isEmpty()) continue;
    qDebug()<<"\t\t"<<"\33[1m"<<line<<"\33[m";
    QStringList tokens = line.split(QRegExp("\\s"));
    if (singleShot)
      inter_adrs=tokens[3]; //germainXY.c-germain.calcul.bruyeres.t has address ab.c.d.e.f
    else
      inter_adrs=tokens[0];
    qDebug()<<"[\33[1mQHyodaSsh::ceaHostReadyReadStandardOutput] inter_adrs="<<inter_adrs<<"\33[m";
    // Maintenant qu'on a le noeud interactif, on lance le tunneling
    if (!isRunning()) run();
    // On prend que la première adresse
    return;
  }
}



/*
 * Check whether "cp" is a valid ascii representation
 * of an Internet address and convert to a binary address.
 * Returns 1 if the address is valid, 0 if not.
 * This replaces inet_addr, the return value from which
 * cannot distinguish between failure and a local broadcast address.
 */
quint32 QHyodaSsh::inetAton(const char *cp){
	unsigned int val;
	int base, n;
	char c;
	u_int parts[4];
	u_int *pp = parts;

	for (;;){
		// Collect number up to ``.''. Values are specified as for C: 0x=hex, 0=octal, other=decimal.
		val = 0;
		base = 10;
		if (*cp == '0'){
			if (*++cp == 'x' || *cp == 'X')
				base = 16, cp++;
			else
				base = 8;
		}
		while ((c = *cp) != '\0') {
			if (isascii(c) && isdigit(c)){
				val = (val * base) + (c - '0');
				cp++;
				continue;
			}
			if (base == 16 && isascii(c) && isxdigit(c)){
				val = (val << 4) +
				(c + 10 - (islower(c) ? 'a' : 'A'));
				cp++;
				continue;
			}
			break;
		}
		if (*cp == '.'){
			// Internet format: a.b.c.d a.b.c (with c treated as 16-bits) a.b (with b treated as 24 bits) 
			if (pp >= parts + 3 || val > 0xff)
				return (0);
			*pp++ = val, cp++;
		}
		else
			break;
	}

	// Check for trailing characters
	if (*cp && (!isascii(*cp) || !isspace(*cp)))
		return (0);

	// Concoct the address according to the number of parts specified
	n = pp - parts + 1;
	switch (n){
		case 1:                 // a -- 32 bits
			break;
		case 2:                 // a.b -- 8.24 bits
			if (val > 0xffffff)
				return (0);
			val |= parts[0] << 24;
			break;
		case 3:                 // a.b.c -- 8.8.16 bits
			if (val > 0xffff)
				return (0);
			val |= (parts[0] << 24) | (parts[1] << 16);
			break;
		case 4:                 // a.b.c.d -- 8.8.8.8 bits
			if (val > 0xff)
				return (0);
			val |= (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8);
			break;
	}
   return htonl(val);
}

