// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "QHyodaToolCell.h"
#include "QHyodaMachine.h"

QHyodaGdb::QHyodaGdb(QHyodaJob *_job,
                     QHyodaMachine *machine,
                     quint32 adrs,
                     quint32 port,
                     quint32 pyld):job(_job),
                                   tab(machine),
                                   tcpAdrs(adrs),
                                   tcpPort(port),
                                   tcpPyld(pyld),
                                   process(new QProcess()),
                                   state(QHyodaGdb::None),
                                   data_read_memory(QString()){
  qDebug() << "[QHyodaGdb::QHyodaGdb] NEW, tcpAdrs="<<tcpAdrs;
}


/**********************
 * ~QHyodaGdb
 **********************/
QHyodaGdb::~QHyodaGdb(void){
  qDebug() << "QHyodaGdb::QHyodaGdb DELETE";
  process->close();
  delete process;
}


/**********************
 * launch
 **********************/
bool QHyodaGdb::launch(void){
  QString command("ssh");
  QStringList args = QStringList() << "-Tx"
                                   << tab->localHostName
                                   << "/usr/bin/gdb" // "/usr/local/bin/gdb"
                                   << "--nw"             // Do not use a window interface
                                   << "--nx"             // Do not read .gdbinit file
                                   << "--interpreter=mi2"
    // << "--readnever" // Do not read symbol files
    // << "--readnow"   // Fully read symbol files on first access
                                   << "--se"
                                   << cmdline;
  qDebug() << "[QHyodaGdb::launch] command"<<command;
  qDebug() << "[QHyodaGdb::launch] args"<<args;
  process->setProcessChannelMode(QProcess::MergedChannels);
  connect(process, SIGNAL(readyReadStandardOutput()), this, SLOT(gdbmi()));
  process->start(command, args);
  if (!process->waitForStarted())
    qFatal("[QHyodaGdb::launch] NOT started!");
  // Configuration initiale
  enqueue("-gdb-set target-async 1");
  enqueue(QString("-target-select remote %1:3883").arg(job->host));
  // Permet de se configurer vis-à-vis de Papi
  enqueue("-interpreter-exec console \"handle SIG36 nostop noprint pass\"");
  enqueue("-break-insert Arcane::Hyoda::loopbreak");   // break #1
  enqueue("-break-insert Arcane::Hyoda::hook");        // break #2
  enqueue("-break-insert -d Arcane::Hyoda::softbreak");   // break #3 disabled by default
  enqueue("-break-insert -d Arcane::EntryPoint::_getAddressForHyoda");   // break #4 disabled by default
  //enqueue("-break-insert Arcane::Hyoda::softbreak");
  //enqueue("-break-insert Arcane::EntryPoint::_getAddressForHyoda");
  enqueue("-enable-pretty-printing");   
  return true;
}


/**********************
 * enqueue
 **********************/
void QHyodaGdb::enqueue(const QString & t){
  commands.enqueue(t);
  process->write(commands.dequeue().append("\n").toLocal8Bit().constData());
}


/**********************
 * dequeue
 **********************/
void QHyodaGdb::dequeue(void){
  if (!commands.isEmpty())
    process->write(commands.dequeue().append("\n").toLocal8Bit().constData());
}


/******************************************************************************
 * Queued Handler de l'output du process GDB
 *****************************************************************************/
void QHyodaGdb::gdbmi(){
  const QStringList read_lines=QString(process->readAllStandardOutput()).split(QRegExp("\n"));
  //qDebug() << "\n\nQHyodaGdb::gdbmi_parser_slot read_lines"<<read_lines;
   
  for(int i=0; i<read_lines.size(); ++i){
    const QString line=read_lines.at(i).trimmed();
    //qDebug() << "QHyodaGdb::gdbmi filtering"<<line;
    if (line.isEmpty()) continue;
    if (line.startsWith("~")){
      //job->gdbTextEdit->append(line.split(QRegExp("\"")).at(1).remove('\n'));
      continue;
    }
    if (line.startsWith("&")){//log-stream-output
      job->gdbTextEdit->append(line);
      continue;
    }
    if (line.startsWith("=library-loaded")){
      job->gdbTextEdit->append(QString("Loading %1").arg(line.split(QRegExp("\"")).at(1)));
      continue;
    }
    if (line.startsWith("=")){//notify-async-output 
      //job->gdbTextEdit->append("=");//line);
      continue;
    }
    if (line=="^done") {
      //job->gdbTextEdit->append(line);
      continue;
    }
    if (line=="^running") continue;
    if (line.startsWith("*running")) continue;
    if (line=="(gdb)") continue;
    
    outputs.enqueue(line);
  }   

  //qDebug() << "QHyodaGdb::gdbmi outputs: "<<outputs;
  
  while (!outputs.isEmpty()){
    const QString output=outputs.dequeue();
  
    //qDebug() << output;
    //qDebug() << "\tQHyodaGdb::gdbmi pending commands"<<*commands;
    //qDebug() << "\n\tQHyodaGdb::gdbmi state"<<state;

    /** DETACH **/
    if (state==QHyodaGdb::Detach){
      state=QHyodaGdb::None;
      qDebug() << "\t[QHyodaGdb::gdbmi] breaks cleanup, detaching and exit";
      commands.enqueue("-var-assign qhyoda_hooked 0");
      commands.enqueue("-break-disable 1"); // loopbreak
      commands.enqueue("-break-disable 2"); // hook
      commands.enqueue("-break-disable 3"); // softbreak
      commands.enqueue("-break-disable 4"); // entry points
      commands.enqueue("-break-delete 1");
      commands.enqueue("-break-delete 2");
      commands.enqueue("-break-delete 3");
      commands.enqueue("-break-delete 4");
      commands.enqueue("-target-detach");
      commands.enqueue("-gdb-exit");
      continue;
    }

    /** EXIT **/
    if (output.startsWith("^exit")){
      qDebug() << "\t[QHyodaGdb::gdbmi] EXIT";
      //job->quit();
      continue;      
    }
    
    // STOPPED becouse of normal exit
    if (output.startsWith("*stopped,reason=\"exited-normally\"")){
      qDebug() << "\t[QHyodaGdb::gdbmi] Exiting normally!";
//#warning  Exiting normally 
      //job->detach();
      //job->quit();
      //tab->close();
      //QCoreApplication::exit(0);
      //return;
      continue;
    }
    
    
    /*****************************************************************
     * BREAKPOINT #4 == Arcane::EntryPoint::_getAddressForHyoda      *
     * BaseForm[Hash["executeEntryPoint", "CRC32"], 10] = 3782526747 *
     *****************************************************************/
    if (output.startsWith("*stopped,reason=\"breakpoint-hit\",disp=\"keep\",bkptno=\"4\"")){
      int i;
      //qDebug() << "\t[QHyodaGdb::gdbmi] breakpoint-hit getAddressForHyoda:"<<output;
      if (!output.contains("next_entry_point_address")){
        qDebug("\33[7mNo address for Hyoda!\33[m");
        continue;
      }
      QStringList tokens=output.split(QRegExp("\""));
      for(i=0;i<tokens.count();i+=1){
        //qDebug() << "\t[QHyodaGdb::gdbmi] token #"<<i<<", is "<<tokens.at(i);
        if (tokens.at(i).startsWith("next_entry_point_address")) break;
        //if (tokens.at(i).startsWith("0x")) break;
      }
      if (i==tokens.count()) {
        qDebug("File not found from address for Hyoda!");
        while(true);
      }
      QString adrs=tokens.at(i+2);//.split(QRegExp("\"")).at(1);
      qDebug() << "\t[QHyodaGdb::gdbmi] adrs="<<adrs;
      int idxOfEntryPoint=entryPoints.indexOf(adrs);
      qDebug() << "\t[QHyodaGdb::gdbmi] idxOfEntryPoint="<<idxOfEntryPoint;
      // Il y a une phase d'apprentissage où l'on crée la liste des points d'entrées
      if (idxOfEntryPoint==-1){ // no item matched: pas dans la liste
        // On a enfin l'adresse du prochain EntryPoint que l'on va utiliser
        //qDebug() << "\t[QHyodaGdb::gdbmi] " << QString("3782526747-data-disassemble -s %1 -e %1+1 -- 1").arg(adrs);
        // On demande de désassembler l'adresse récupérée
        commands.enqueue(QString("3782526747-data-disassemble -s %1 -e %1+1 -- 1").arg(adrs));
        // Pour rajouter l'adresse à la liste des points d'entrée
        entryPoints.append(adrs);
      }else{
        // Si le point d'entrée est associé à un couple fichier/ligne, on l'affiche
        if (!entryPointsFile.at(idxOfEntryPoint).isEmpty()){
          showViaEmacsClient(entryPointsFile.at(idxOfEntryPoint),entryPointsLine.at(idxOfEntryPoint));
          job->startButton->setEnabled(true);
          job->stepButton->setEnabled(true);
          //qDebug() << "\t[QHyodaGdb::gdbmi] getAddressForHyoda & setEnabled";
        }else
          commands.enqueue("-exec-continue");
      }
      continue;
    }
    if (output.startsWith("3782526747")){
      //qDebug() << "\t3782526748:QHyodaGdb::gdbmi file+line"<<output;
      QString fileName;
      QString lineNumber("+");
      if (output.contains("file=")){
        QStringList tokens=output.split(QRegExp(","));
        //qDebug() << "\t3782526748:QHyodaGdb tokens="<<tokens;
        fileName=tokens.at(2).split(QRegExp("\"")).at(1);
        lineNumber.append(tokens.at(1).split(QRegExp("\"")).at(1));
        qDebug()<<"\33[7m"<<fileName<<lineNumber<<"\33[m";
        //showViaEmacsClient(fileName,lineNumber);
      }
      entryPointsFile.append(fileName);
      entryPointsLine.append(lineNumber);
      commands.enqueue("-exec-continue");
      continue;
    }

    
    /**************
     * SOFT BREAK *
     **************/
    if (output.startsWith("*stopped,reason=\"breakpoint-hit\",disp=\"keep\",bkptno=\"3\"")){
      //qDebug() << "\t[QHyodaGdb::gdbmi] breakpoint-hit softbreak:"<<output;
      QString fileName;
      QString lineNumber("+");
      // Fetching filename and line number
      const QStringList tokens=output.split(QRegExp(","));
      for(int i=0; i<tokens.size(); ++i){
        const QString token=tokens.at(i).trimmed();
        //qDebug() << "QHyodaGdb::gdbmi softbreak filtering"<<token;
        if (token.isEmpty()) continue;
        if (token.startsWith("{name=\"fileName\"")){
          const QStringList values=tokens.at(i+1).trimmed().split(QRegExp("\""));
          //qDebug()<<"fileName="<<values.value(2);
          fileName=values.value(2);
        }
        if (token.startsWith("{name=\"lineNumber\"")){
          const QStringList values=tokens.at(i+1).trimmed().split(QRegExp("\""));
          //qDebug()<<"lineNumber="<<values.value(1);
          lineNumber.append(values.value(1));
        }
      }
      fileName.remove('\\');
      //qDebug()<<"fileName:"<<fileName<<", lineNumber="<<lineNumber;
      
      showViaEmacsClient(fileName,lineNumber);
      job->startButton->setEnabled(true);
      job->stepButton->setEnabled(true);
      continue;
    }

       
    /**************
     * Hook BREAK *
     **************/
    if (output.startsWith("*stopped,reason=\"breakpoint-hit\",disp=\"keep\",bkptno=\"2\"")){
      //qDebug() << "\tQHyodaGdb::gdbmi breakpoint-hit hook";
      // Si on veut changer de cell
      if (state==QHyodaGdb::TargetCell){
        //qDebug() << "\tQHyodaGdb::gdbmi TARGET CELL"<<job->target_cell_id;//tab->targetCellNumLineEdit->text();
        commands.enqueue("-break-enable 1");
        //commands.enqueue("-break-disable 3");
        //commands.enqueue("-break-disable 4");
        commands.enqueue("-exec-continue");
        continue;
      }
      // Sinon, on rafraîchit les variables depuis Arcane
      commands.enqueue(data_read_memory.toLocal8Bit().constData());
      continue;
    }

    /** Résultat, puis ACTION **/
    if (output.startsWith("^done,addr")){
      //qDebug() << "\tQHyodaGdb::gdbmi ^done,addr: "<<output;
      const QStringList splitted_output=output.split(QRegExp("\""));    
      job->refresh_common_variables(splitted_output);

      if (job->bottomRightTools->tabText(job->bottomRightTools->currentIndex())=="Cell")
        job->bottomRightTools->cell->refresh(splitted_output);
      
      // Si un STOP a été demandé
      if (state==QHyodaGdb::Interrupt){
        state=QHyodaGdb::None;
        commands.enqueue("-exec-interrupt");
        job->tackButton->setEnabled(false);
        job->startButton->setEnabled(true);
        job->stepButton->setEnabled(true);
        continue;
      }
  
      // Si un next a été demandé, on ne relance pas
      if (state==QHyodaGdb::Step){
        commands.enqueue("-break-enable 3");
        //commands.enqueue("-break-enable 4");
        job->startButton->setEnabled(true);
        job->stepButton->setEnabled(true);
        commands.enqueue("-exec-continue");
        continue;
      }

      // Si le bouton untack a été poussé
      if (state==QHyodaGdb::Untack){
        state=QHyodaGdb::None;
        //qDebug() << "\tQHyodaGdb::gdbmi UNTACK";
        commands.enqueue("-break-disable 3");
        commands.enqueue("-break-disable 4");
        commands.enqueue("-var-assign qhyoda_hooked 0");
      }
      
      commands.enqueue("-exec-continue");
      continue;
    }

    
    /////////////////////////////////////////////////////////////////
    // Au delà de ce point, c'est de l'opérationnel moins critique //
    /////////////////////////////////////////////////////////////////

    /**  Récupération d'un retour du DUMP **/
    if (output.startsWith("^done,value")){// On récupère l'adresse de la value
      //qDebug() << "\tQHyodaGdb::gdbmi ^done,value";
      // On ne chope que la première adresse (celle de 'data')
      if (!data_read_memory.isNull()) continue;
      data_read_memory.append("-data-read-memory ");
      data_read_memory.append(output.split(QRegExp("\"")).at(1));
      data_read_memory.append(" x 8 1 32"); // sizeof(64bits): struct gdb_data_to_output = 8+8*3 =32
      //qDebug() << "\tQHyodaGdb::gdbmi VALUE, data_read_memory="<<data_read_memory.trimmed();
      continue;
    }
   
  
    /* loopbreak: utile à l'init, lors d'un changement de maille et quand on se réaccroche */
    if (output.startsWith("*stopped,reason=\"breakpoint-hit\",disp=\"keep\",bkptno=\"1\"")){
      //qDebug()<<"\33[7m\tQHyodaGdb::gdbmi *stopped breakpoint-hit @ Arcane::Hyoda::breakpoint, state="<<state<<"\33[m";

      if (state==QHyodaGdb::Retack){
        state=QHyodaGdb::None;
        //qDebug() << "\tQHyodaGdb::gdbmi RETACK, now assigning hyoda";
        commands.enqueue("-var-assign qhyoda_hooked 1");
        commands.enqueue("-break-disable 1");
        commands.enqueue("-break-disable 3");
        commands.enqueue("-break-disable 4");
        commands.enqueue("-exec-continue");
        continue;
      }
      
      // Si on veut changer de cell
      if (state==QHyodaGdb::TargetCell){
        state=QHyodaGdb::None;
        //qDebug() << "\tQHyodaGdb::gdbmi NOW MODIFYING TARGET CELL"<<job->data->target_cell_uid;
        commands.enqueue(QString("-var-assign target_cell_uid %1").arg(job->data->target_cell_uid));
        commands.enqueue("-break-disable 1");
        commands.enqueue("-exec-continue");
        continue;
      }

      // data est la première value que l'on fetch, afin de récupérer son adresse de base
      commands.enqueue("-var-create data * m_data");
      commands.enqueue("-var-set-format data hexadecimal");
      commands.enqueue("-var-evaluate-expression data");
            
      commands.enqueue("-var-create target_cell_uid * m_target_cell_uid");
      commands.enqueue("-var-set-format target_cell_uid hexadecimal");
      commands.enqueue("-var-assign target_cell_uid 0");
      
      commands.enqueue("-var-create qhyoda_hooked * m_qhyoda_hooked");
      commands.enqueue("-var-set-format qhyoda_hooked hexadecimal");

      // Attention, dès qu'on met 'qhyoda_hooked' à 1, ça risque de breaker!
      commands.enqueue("-var-assign qhyoda_hooked 1");

      // Et on relache le breakpoint 1
      commands.enqueue("-break-disable 1");
      // Et on ne veut pas breaker sur le softbreak et les entryPoints
      //commands.enqueue("-break-enable 3");
      //commands.enqueue("-break-enable 4");
      // Si on est en Local, il faut encore faire les dup2 et close
      commands.enqueue("-exec-continue");
      continue;
    }
    
    if (output.startsWith("^done,name=\"target_cell_uid\"")) continue;
    if (output.startsWith("^done,name=\"qhyoda_hooked\"")) continue;
    if (output.startsWith("^done,name=\"data\"")) continue;
    if (output.startsWith("^done,format")) continue;
    if (output.startsWith("^done,bkpt")){
      //qDebug() << "\tQHyodaGdb::gdbmi "<<output;
      job->gdbTextEdit->append(output);
      continue;
    }

    /* Async STOP */
    if (output.startsWith("*stopped")){
      //qDebug() << "\tQHyodaGdb::gdbmi *stopped: "<<output;
      if (state==QHyodaGdb::Retack){
        qDebug() << "\tQHyodaGdb::gdbmi RETACK";
        commands.enqueue("-break-enable 1");
        //commands.enqueue("-break-enable 3");
        //commands.enqueue("-break-enable 4");
        commands.enqueue("-exec-continue");
        continue;
      }
      job->gdbTextEdit->append(output);
      // On tente de continuer
      commands.enqueue("-exec-continue");
      continue;
    }

    /* Connection initiale */
    if (output.startsWith("^connected")){
      state=QHyodaGdb::None;
      qDebug() << "\tQHyodaGdb::gdbmi ^connected";
      // On vient de se connecter, on autorise le clickodrome
      job->arcaneCommonVariablesGroupBox->setEnabled(true);
      job->stopButton->setEnabled(true);
      job->stepButton->setEnabled(true);
      job->startButton->setEnabled(false);
      job->tackButton->setEnabled(true);
      job->cellButton->setEnabled(true);
      job->meshButton->setEnabled(true);
      job->papi_slot();
      job->mesh_slot();
      qDebug() << "ARCANE_HYODA_MATRIX_RENDER:"<<qgetenv("ARCANE_HYODA_MATRIX_RENDER").count();
      if (qgetenv("ARCANE_HYODA_MATRIX_RENDER").count()>0) job->matrix_slot();
      continue;
    }
  
  
    ///////////////////////////////////////////////////////////
    // Au delà de ce point, il devrait rester les end-states //
    //////////////////////////////////////////////////////////
  
    if (output.startsWith("*running")){
      qDebug() << "\tQHyodaGdb::gdbmi gdb_async_running";
      job->gdbTextEdit->append(output);
      continue;
    }
  
    if (output.startsWith("^error")){   
      state=QHyodaGdb::None;
      qDebug() << "\tQHyodaGdb::gdbmi ^error"<<output;
      job->gdbTextEdit->append(output);
      continue;
    }
    
    if (output.startsWith("~")){
      qDebug() << "\tQHyodaGdb::gdbmi console-stream-output:"<<output;
      job->gdbTextEdit->append(output);
      continue;
    }
  
    if (output.startsWith("=")){
      qDebug() << "\tQHyodaGdb::gdbmi notify-async-output:"<<output;
      job->gdbTextEdit->append(output);
      continue;
    }
 
    qDebug() << "\tQHyodaGdb::gdbmi UNFILTERED "<<output;
    job->gdbTextEdit->append(output);
  }
  
  dequeue();
}

void QHyodaGdb::showViaEmacsClient(QString file, QString line){
  if (job->srcButton->isEnabled()) return;
  //qDebug()<<"\33[7mshowViaEmacsClient"<<file<<line<<"\33[m";
  QProcess EmacsClient(this);
  EmacsClient.setProcessChannelMode(QProcess::MergedChannels);
  QString emacsclient("emacsclient");
  QStringList clientArgs=QStringList() << "-s"<< "src" << "-n" << line << file;
  EmacsClient.start(emacsclient, clientArgs);
  if (!EmacsClient.waitForStarted()) qFatal("QProcessEmacs NOT started!");
  if (!EmacsClient.waitForFinished()) qFatal("QProcessEmacs NOT finished!");
  EmacsClient.close();
}
