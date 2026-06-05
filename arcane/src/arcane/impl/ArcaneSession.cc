// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneSession.cc                                            (C) 2000-2020 */
/*                                                                           */
/* Default implementation of a Session.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/ArcaneSession.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/IParallelMng.h"
#include "arcane/IParallelSuperMng.h"
#include "arcane/IParallelReplication.h"
#include "arcane/Directory.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/ISubDomain.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/IApplication.h"
#include "arcane/IRessourceMng.h"
#include "arcane/XmlNode.h"
#include "arcane/CommonVariables.h"
#include "arcane/IIOMng.h"
#include "arcane/ApplicationBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneSession::
ArcaneSession(IApplication* application)
: Session(application)
, m_result_doc(0)
, m_case_name("output")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneSession::
~ArcaneSession()
{
  delete m_result_doc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneSession::
build()
{
  Session::build();

  // If a value is specified in applicationBuildInfo(), use it.
  // Otherwise, the output directory is the case name in the current directory.
  String output_dir_name = application()->applicationBuildInfo().outputDirectory();
  if (output_dir_name.empty()){
    // Determines and creates the directory for exports and listing
    Directory output_base_dir(platform::getCurrentDirectory());
    output_dir_name = output_base_dir.file(m_case_name);
  }
  m_output_directory = Directory(output_dir_name);
  m_listing_directory = Directory(m_output_directory,"listing");

  IParallelSuperMng* sm = application()->parallelSuperMng();
  bool is_master_io = sm->isMasterIO();

  info() << "Output directory is <" << output_dir_name << ">";
  if (is_master_io){
    m_output_directory.createDirectory();
    m_listing_directory.createDirectory();
  }
  // To ensure that the entire output directory is visible to everyone
  sm->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneSession::
_initSubDomain(ISubDomain* sd)
{
  sd->setCaseName(m_case_name);

  sd->setExportDirectory(m_output_directory);
  sd->setStorageDirectory(m_output_directory);
  sd->setListingDirectory(m_listing_directory);

  setLogAndErrorFiles(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Positions the file names for logs and errors.
 *
 * This method is collective across all subdomains and replicas.
 */
void ArcaneSession::
setLogAndErrorFiles(ISubDomain* sd)
{
  bool is_log_disabled = false;
  if (!platform::getEnvironmentVariable("ARCANE_DISABLE_LOG").null()){
    is_log_disabled = true;
  }
  ITraceMng* tm = sd->traceMng();

  IParallelMng* pm = sd->parallelMng();
  IParallelReplication* replication = pm->replication();
  IParallelMng* all_replica_pm = sd->allReplicaParallelMng();
  bool is_master_io = all_replica_pm->isMasterIO();

  Int32 sid = sd->subDomainId();
  Int32 nb_sub_domain = sd->nbSubDomain() * replication->nbReplication();

  Int32 nb_sub_dir = 1;
  Int32 NB_FILE_PER_SUBDIR = 256;

  if (nb_sub_domain>(NB_FILE_PER_SUBDIR)){
    nb_sub_dir = nb_sub_domain / NB_FILE_PER_SUBDIR;
    // if (nb_sub_dir<1) nb_sub_dir = 1; // DEAD CODE
  }

  if (is_master_io){
    m_listing_directory.createDirectory();
    if (nb_sub_dir!=1){
      for( Integer i=0; i<nb_sub_dir; ++i ){
        Directory sub_dir = Directory(m_listing_directory,String::fromNumber(i));
        sub_dir.createDirectory();
      }
    }
  }
  
  // Waits for the master process to create all directories.
  Directory my_directory = m_listing_directory;
  if (nb_sub_dir!=1){
    all_replica_pm->barrier();
    Int32 global_sid = sid  + (sd->nbSubDomain() * replication->replicationRank());
    my_directory = Directory(m_listing_directory,String::fromNumber(global_sid/NB_FILE_PER_SUBDIR));
  }

  // In case of replication, adds the replica number to the end of the files
  // (except for replica 0 to remain compatible with the name without replication)
  String file_suffix = String::fromNumber(sid);
  if (replication->hasReplication()){
    Int32 replication_rank = replication->replicationRank();
    if (replication_rank>0)
      file_suffix  = file_suffix + String("_r") + replication_rank;
  }

  {
    StringBuilder fn("errors.");
    fn += file_suffix;
    String file(my_directory.file(fn));
    info() << "Error output file  '" << file << "'";
    platform::removeFile(file);
    tm->setErrorFileName(file);
  }

  if (is_log_disabled){
    tm->setLogFileName(String());
    tm->info() << "Logs are disabled because environment variable ARCANE_DISABLE_LOG is set";
  }
  else{
    StringBuilder fn("logs.");
    fn += file_suffix;
    String file(my_directory.file(fn));
    platform::removeFile(file);
    tm->setLogFileName(file);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneSession::
_writeExecInfoFile(int ret_val)
{
  IParallelSuperMng* psm = _application()->parallelSuperMng();
  if (psm->commRank()!=0)
    return;

  _checkExecInfoFile();

  XmlNode result_root = m_result_doc->documentNode().documentElement();
  XmlNode execution = result_root.child("execution");
  execution.clear();

  String datetime(platform::getCurrentDateTime());
  // Saves the case information
  {
    execution.createAndAppendElement("return-value",String::fromNumber(ret_val));
    execution.createAndAppendElement("finish-date",datetime);
    execution.createAndAppendElement("date",platform::getCurrentDateTime());
    SubDomainCollection sub_domains(subDomains());
    if (!sub_domains.empty()){
      ISubDomain* sd = sub_domains.front();
      // The subdomain must be correct.
      // This is sometimes not the case if this method is called after an exception
      // thrown during subdomain initialization
      if (sd->isInitialized()){
        ITimeLoopMng* tm = sd->timeLoopMng();
        const CommonVariables& v = sd->commonVariables();
        execution.createAndAppendElement("current-iteration",String::fromNumber(v.globalIteration()));
        execution.createAndAppendElement("current-time",String::fromNumber(v.globalTime()));
        execution.createAndAppendElement("end-time",String::fromNumber(v.globalFinalTime()));
        execution.createAndAppendElement("deltat",String::fromNumber(v.globalDeltaT()));
        execution.createAndAppendElement("cpu-time",String::fromNumber(v.globalCPUTime()));
        const char* is_finished_str = tm->finalTimeReached() ? "1" : "0";
        execution.createAndAppendElement("is-finished",is_finished_str);
      }
    }
  }
  String file(m_listing_directory.file("coderesult.xml"));
  logdate() << "Info output in 'coderesult.xml'";
  IIOMng* io_mng = _application()->ioMng();
  io_mng->writeXmlFile(m_result_doc,file);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneSession::
writeExecInfoFile()
{
  _writeExecInfoFile(-1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneSession::
_checkExecInfoFile()
{
  if (m_result_doc)
    return;

  m_result_doc = _application()->ressourceMng()->createXmlDocument();
  XmlNode doc = m_result_doc->documentNode();
  XmlElement root(doc,"arcane-result");

  // Saves the case configuration information.
  XmlElement config(root,"config");
  XmlElement execution(root,"execution");
    
  config.createAndAppendElement("case-name",m_case_name);
  config.createAndAppendElement("host",platform::getHostName());
  config.createAndAppendElement("pid",String::fromNumber(platform::getProcessId()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneSession::
endSession(int ret_val)
{
  _writeExecInfoFile(ret_val);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneSession::
setCaseName(String casename)
{
  m_case_name = casename;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
