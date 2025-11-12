// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CheckpointMng.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire des protections.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/Directory.h"
#include "arcane/ICheckpointMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/IParallelMng.h"
#include "arcane/IParallelReplication.h"
#include "arcane/IRessourceMng.h"
#include "arcane/IVariableMng.h"
#include "arcane/IIOMng.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/XmlNode.h"
#include "arcane/ICheckpointReader.h"
#include "arcane/ICheckpointWriter.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/IObservable.h"
#include "arcane/CheckpointInfo.h"
#include "arcane/SubDomainBuildInfo.h"
#include "arcane/MeshPartInfo.h"

#include "arcane/VariableCollection.h"
#include "arcane/IVariable.h"
#include "arcane/IMeshModifier.h"
#include "arcane/ItemGroup.h"
#include "arcane/IItemFamily.h"
#include "arcane/IMainFactory.h"
#include "arcane/IPrimaryMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ISubDomain*
arcaneCreateSubDomain(ISession* session,const SubDomainBuildInfo& sdbi);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des protections.
 */
class CheckpointMng
: public TraceAccessor
, public ICheckpointMng
{
 public:

 public:

  CheckpointMng(ISubDomain*);
  ~CheckpointMng() override;

 public:

  void readCheckpoint() override;
  void readDefaultCheckpoint() override;
  CheckpointInfo readDefaultCheckpointInfo() override;
  void readCheckpoint(ICheckpointReader* reader) override;
  void readCheckpoint(ByteConstArrayView infos) override;
  void readCheckpoint(const CheckpointInfo& checkpoint_infos) override;
  CheckpointInfo readCheckpointInfo(Span<const Byte> infos,const String& buf_name) override;

  void writeCheckpoint(ICheckpointWriter* writer) override;
  void writeCheckpoint(ICheckpointWriter* writer,ByteArray& infos) override;
  void writeDefaultCheckpoint(ICheckpointWriter* writer) override;
  IObservable* writeObservable() override { return m_write_observable; }
  IObservable* readObservable() override { return m_read_observable; }

 public:
  
  void build();

 private:

  ISubDomain* m_sub_domain;
  IObservable* m_write_observable; //!< Observable en écriture
  IObservable* m_read_observable; //!< Observable en lecture

 private:

  void _writeCheckpointInfoFile(ICheckpointWriter* checkpoint_writer,ByteArray& infos);
  CheckpointInfo _readCheckpointInfo(Span<const Byte> infos,const String& info_file_name);
  void _readCheckpoint(const CheckpointInfo& checkpoint_info);
  void _readCheckpoint(const CheckpointReadInfo& infos);
  bool _checkChangingNbSubDomain(const CheckpointInfo& ci);
  void _applyNbSubDomainChange(const CheckpointInfo& ci,ICheckpointReader2* reader);
  void _changeItemsOwner(IMesh* mesh,Int32ConstArrayView old_ranks_to_new_ranks);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ICheckpointMng*
arcaneCreateCheckpointMng(ISubDomain* sd)
{
  CheckpointMng* cm = new CheckpointMng(sd);
  cm->build();
  return cm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CheckpointMng::
CheckpointMng(ISubDomain* sd)
: TraceAccessor(sd->traceMng())
, m_sub_domain(sd)
, m_write_observable(IObservable::createDefault())
, m_read_observable(IObservable::createDefault())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CheckpointMng::
~CheckpointMng()
{
  m_read_observable->detachAllObservers();
  m_write_observable->detachAllObservers();

  delete m_read_observable;
  delete m_write_observable;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
build()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
readCheckpoint()
{
  readDefaultCheckpoint();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
readCheckpoint(ICheckpointReader* reader)
{
  m_sub_domain->variableMng()->readCheckpoint(reader);
  m_read_observable->notifyAllObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
_readCheckpoint(const CheckpointReadInfo& infos)
{
  m_sub_domain->variableMng()->readCheckpoint(infos);
  m_read_observable->notifyAllObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CheckpointInfo CheckpointMng::
readDefaultCheckpointInfo()
{
  // Lit le fichier contenant les infos de la précédente exécution pour
  // connaître le service de protection/reprise utilisé
  String info_file_name(m_sub_domain->exportDirectory().file("checkpoint_info.xml"));
  ByteUniqueArray bytes;
  IIOMng* io_mng = m_sub_domain->ioMng();
  io_mng->collectiveRead(info_file_name,bytes);
  CheckpointInfo checkpoint_info = _readCheckpointInfo(bytes,info_file_name);
  return checkpoint_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
readDefaultCheckpoint()
{
  CheckpointInfo checkpoint_info = readDefaultCheckpointInfo();
  _readCheckpoint(checkpoint_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
readCheckpoint(ByteConstArrayView bytes_infos)
{
  CheckpointInfo checkpoint_info = _readCheckpointInfo(bytes_infos,"unknown");
  _readCheckpoint(checkpoint_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CheckpointInfo CheckpointMng::
readCheckpointInfo(Span<const Byte> bytes_infos,const String& buf_name)
{
  String buf_name2 = buf_name;
  if (buf_name2.null())
    buf_name2 = "unknown";
  CheckpointInfo checkpoint_info = _readCheckpointInfo(bytes_infos,buf_name2);
  return checkpoint_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CheckpointInfo CheckpointMng::
_readCheckpointInfo(Span<const Byte> bytes_infos,const String& info_file_name)
{
  CheckpointInfo checkpoint_info;

  // Par défaut, relit les infos en fonction du IParallelMng associé au sous-domaine
  IParallelMng* pm = m_sub_domain->parallelMng();
  Int32 rank = pm->commRank();
  checkpoint_info.setSubDomainRank(rank);
  Int32 replication_rank = pm->replication()->replicationRank();
  checkpoint_info.setReplicationRank(replication_rank);

  ITraceMng* tm = m_sub_domain->traceMng();
  auto xml_doc_ptr = IXmlDocumentHolder::loadFromBuffer(bytes_infos,info_file_name,tm);
  ScopedPtrT<IXmlDocumentHolder> xml_doc(xml_doc_ptr);
  XmlNode doc_node = xml_doc->documentNode();
  if (doc_node.null())
    ARCANE_FATAL("Can not read file '{0}' containing checkpoint/restart informations",
                 info_file_name);

  XmlNode doc_elem = doc_node.documentElement();

  Int32 nb_checkpoint_sub_domain = doc_elem.attr("nb-sub-domain").valueAsInteger();
  checkpoint_info.setNbSubDomain(nb_checkpoint_sub_domain);

  Int32 nb_checkpoint_replication = doc_elem.attr("nb-replication").valueAsInteger();
  checkpoint_info.setNbReplication(nb_checkpoint_replication);

  XmlNode service_elem = doc_elem.child("service");
  String service_name = service_elem.attrValue("name");
  if (service_name.null()){
    ARCANE_THROW(ParallelFatalErrorException,
                 "The file '{0}}' doesn't have "
                 "the name of the protection/restore service used "
                 "(attribute /checkpoint-info/service/@name)", info_file_name);
  }
  checkpoint_info.setServiceName(service_name);
  String service_directory = service_elem.attrValue("directory");
  checkpoint_info.setDirectory(service_directory);

  XmlNode times_node = doc_node.documentElement().child("times");

  XmlNode last_index_attr = times_node.attr("last-index");
  if (last_index_attr.null())
    ARCANE_THROW(IOException,"missing attribute 'last-index'");

  XmlNode last_time_attr = times_node.attr("last-time");
  if (last_time_attr.null())
    ARCANE_THROW(IOException,"missing attribute 'last-time'");
    
  Real last_time = last_time_attr.valueAsReal();
  checkpoint_info.setCheckpointTime(last_time);

  Integer last_index = last_index_attr.valueAsInteger();
  checkpoint_info.setCheckpointIndex(last_index);

  XmlNode meta_data_node = service_elem.child("meta-data");
  if (meta_data_node.null())
    ARCANE_THROW(IOException,"missing tag 'meta-data'");
  checkpoint_info.setReaderMetaData(meta_data_node.value());

  return checkpoint_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
readCheckpoint(const CheckpointInfo& checkpoint_info)
{
  _readCheckpoint(checkpoint_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
_readCheckpoint(const CheckpointInfo& checkpoint_info)
{
  String service_name = checkpoint_info.serviceName();
  if (service_name.null())
    ARCANE_THROW(ParallelFatalErrorException,"Null service");

  String service_directory = checkpoint_info.directory();

  bool has_changing_sub_domain = _checkChangingNbSubDomain(checkpoint_info);

  IApplication* app = m_sub_domain->application();
  // Tente d'utiliser l'interface ICheckpointReader2 si disponible.
  // A noter que le service qui implémente ICheckpointReader2 est un service
  // de l'application alors que pour ICheckpointReader il s'agit d'un service
  // de sous-domaine
  // S'il n'est pas disponible, utilise l'implémentation ICheckpointReader.
  // Avec la nouvelle implémentation, il est possible de traiter le cas où
  // le nombre de sous-domaines change.
  {
    ServiceBuilder<ICheckpointReader2> sb(app);
    Ref<ICheckpointReader2> s(sb.createReference(service_name,SB_AllowNull));
    if (s.get()){
      info() << "Using the checkpoint/restart service"
             << " <" << service_name << "> (implement ICheckpointReader2)";
      if (has_changing_sub_domain)
        _applyNbSubDomainChange(checkpoint_info,s.get());
      else{
        CheckpointReadInfo cri(checkpoint_info);
        IParallelMng* pm = m_sub_domain->parallelMng();
        cri.setParallelMng(pm);
        cri.setReader(s.get());
        _readCheckpoint(cri);
      }
      return;
    }
  }
  // Avec l'ancienne implémentation, il n'est pas possible de changer
  // le nombre de sous-domaines
  if (has_changing_sub_domain)
    ARCANE_FATAL("The number of sub-domains/replica in this run is different "
                 "from the number in checkpoint but the service specified "
                 "for checkpoint {0} does not handle this case",service_name);

  ServiceFinder2T<ICheckpointReader,ISubDomain> sf2(app,m_sub_domain);
  Ref<ICheckpointReader> checkpoint_reader(sf2.createReference(service_name));

  if (!checkpoint_reader.get()){
    ARCANE_FATAL("The service specified for checkpoint/restart ({0}) is not available",
                 service_name);
  }

  info() << "Using the checkpoint/restart service <" << service_name << ">";
  Real last_time = checkpoint_info.checkpointTime();
  Int32 last_index = checkpoint_info.checkpointIndex();
    
  checkpoint_reader->setCurrentTimeAndIndex(last_time,last_index);

  String meta_data = checkpoint_info.readerMetaData();
  checkpoint_reader->setReaderMetaData(meta_data);
  checkpoint_reader->setBaseDirectoryName(service_directory);

  readCheckpoint(checkpoint_reader.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
writeCheckpoint(ICheckpointWriter* writer)
{
  writeDefaultCheckpoint(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
writeDefaultCheckpoint(ICheckpointWriter* writer)
{
  ByteUniqueArray bytes_infos;
  writeCheckpoint(writer,bytes_infos);

  if (m_sub_domain->allReplicaParallelMng()->isMasterIO()){
    Directory export_directory(m_sub_domain->exportDirectory());
    String info_file(export_directory.file("checkpoint_info.xml"));
    std::ofstream ofile(info_file.localstr());
    ofile.write((const char*)bytes_infos.unguardedBasePointer(),bytes_infos.size());
    if (!ofile.good())
      ARCANE_THROW(IOException,"Can not write file '{0}'",info_file);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
writeCheckpoint(ICheckpointWriter* writer,ByteArray& infos)
{
  m_write_observable->notifyAllObservers();
  m_sub_domain->variableMng()->writeCheckpoint(writer);
  _writeCheckpointInfoFile(writer,infos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
_writeCheckpointInfoFile(ICheckpointWriter* checkpoint_writer,ByteArray& infos)
{
  ISubDomain* sd = m_sub_domain;
  IParallelMng* pm = sd->parallelMng();
  IParallelReplication* pr = pm->replication();

  Int32 nb_rank = pm->commSize();
  Int32 nb_replica = pr->nbReplication();

  ScopedPtrT<IXmlDocumentHolder> info_document; //!< Infos sur les protections

  RealConstArrayView checkpoints_time = checkpoint_writer->checkpointTimes();

  IRessourceMng* rm = sd->ressourceMng();
  info_document = rm->createXmlDocument();
  XmlNode doc = info_document->documentNode();
  XmlElement root(doc,"checkpoint-info");

  // Sauvegarde les infos sur le nombre de réplication et de sous-domaines.
  // Cela permettra plus tard de mettre en place les reprises en faisant
  // varier le nombre de sous-domaines ou de réplication.
  root.setAttrValue("nb-sub-domain",String::fromNumber(nb_rank));
  root.setAttrValue("nb-replication",String::fromNumber(nb_replica));

  XmlElement service_info(root,"service");
  String reader_name = checkpoint_writer->readerServiceName();
  service_info.setAttrValue("name",reader_name);
  service_info.setAttrValue("directory",checkpoint_writer->baseDirectoryName());

  String reader_meta_data = checkpoint_writer->readerMetaData();
  XmlElement meta_data_elem(service_info,"meta-data",reader_meta_data);
  XmlElement checkpoints_time_elem(root,"times");

  XmlNode info_root = info_document->documentNode().documentElement();

  {
    // Sauve les informations de la dernière protection
    Integer nb_checkpoint = checkpoints_time.size();
    if (nb_checkpoint>0){
      checkpoints_time_elem.setAttrValue("last-time",String::fromNumber(checkpoints_time[nb_checkpoint-1]));
      checkpoints_time_elem.setAttrValue("last-index",String::fromNumber(nb_checkpoint-1));
    }
  }
  checkpoints_time_elem.clear();
  for( Integer i=0, is=checkpoints_time.size(); i<is; ++i ){
    XmlElement elem(checkpoints_time_elem,"time");
    elem.setAttrValue("value",String::fromNumber(checkpoints_time[i]));
  }

  info_document->save(infos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Regarde si le nombre de sous-domaines a changé entre la
 * protection et l'allocation actuelle.
 */
bool CheckpointMng::
_checkChangingNbSubDomain(const CheckpointInfo& ci)
{
  // Vérifie que le nombre de sous-domaine et de replica est le même entre
  // la protection et l'exécution courante.
  Int32 nb_checkpoint_sub_domain = ci.nbSubDomain();
  Int32 nb_checkpoint_replication = ci.nbReplication();
  // Si on n'a pas les infos (ou qu'elles sont invalides) sur le nombre de
  // sous-domaines ou de replica, on considère qu'on ne change pas le partitionnement.
  // Cela peut arriver si le fichier 'checkpoint_info' est trop ancien
  // ou n'a pas été écrit par ce CheckpointMng.
  if (nb_checkpoint_sub_domain<1 || nb_checkpoint_replication<1){
    info() << "Invalid or missing partitionning info in checkpoint.";
    return false;
  }
  info() << "Reading checkpoint nb_sub_domain=" << nb_checkpoint_sub_domain
         << " nb_replication=" << nb_checkpoint_replication;
  MeshPartInfo current_part_info(makeMeshPartInfoFromParallelMng(m_sub_domain->parallelMng()));

  Int32 nb_rank = current_part_info.nbPart();
  Int32 nb_replication = current_part_info.nbReplication();
  bool has_different_sub_domain = false;
  if (nb_rank!=nb_checkpoint_sub_domain){
    has_different_sub_domain = true;
  }
  // Pour l'instant, one ne supporte pas le changement du nombre de réplica.
  if (nb_replication!=nb_checkpoint_replication){
    ARCANE_THROW(ParallelFatalErrorException,
                 "Bad number of replication ({0} in checkpoint, {1} in this run)",
                 nb_checkpoint_replication,nb_replication);
  }
  return has_different_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
_changeItemsOwner(IMesh* mesh,Int32ConstArrayView old_ranks_to_new_ranks)
{
  Int32 mesh_rank = mesh->meshPartInfo().partRank();
  // Change les propriétaires de toutes les familles
  for( IItemFamily* family : mesh->itemFamilies() ){
    const ItemGroup& all_items = family->allItems();
    // Change les propriétaires pour correspondre au nouveau découpage.
    ENUMERATE_ITEM(iitem,all_items){
      Item item = *iitem;
      Int32 owner = item.owner();
      item.mutableItemBase().setOwner(old_ranks_to_new_ranks[owner],mesh_rank);
    }
    family->notifyItemsOwnerChanged();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointMng::
_applyNbSubDomainChange(const CheckpointInfo& ci,ICheckpointReader2* reader)
{
  ISubDomain* sd1 = m_sub_domain;
  IApplication* app = sd1->application();
  IParallelMng* pm = sd1->parallelMng();
  Int32 nb_old_rank = ci.nbSubDomain();
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  if (nb_rank>nb_old_rank)
    ARCANE_THROW(NotImplementedException,"Increasing number of sub-domains (old={0} new={1})",
                 nb_old_rank,nb_rank);
  UniqueArray<Int32> old_ranks_to_new_ranks(nb_old_rank);
  UniqueArray<Int32> ranks_to_read;
  for( Integer i=0; i<nb_old_rank; ++i ){
    Int32 new_rank = i % nb_rank;
    old_ranks_to_new_ranks[i] = new_rank;
    if (new_rank==my_rank)
      ranks_to_read.add(i);
  }
  info() << "OLD_RANKS_TO_NEW_RANKS=" << old_ranks_to_new_ranks;
  info() << "RANKS_TO_READ=" << ranks_to_read;
  info() << "Apply Changing nb sub domain my_rank=" << my_rank;
  String service_name = ci.serviceName();
  // TODO: faire un nouveau parallelMng() par sous-domaine créé
  // (pour avoir un ITraceMng par sous-domaine)
  IParallelMng* pm2 = pm->sequentialParallelMng();
  UniqueArray<ISubDomain*> sd_to_merge_list2;
  UniqueArray<Byte> case_bytes;
  sd1->fillCaseBytes(case_bytes);

  String message_passing_service = "SequentialParallelMngContainerFactory";
  ServiceBuilder<IParallelMngContainerFactory> sf(app);
  auto pbf = sf.createReference(message_passing_service,SB_AllowNull);
  if (!pbf)
    ARCANE_FATAL("Can not find service '{0}' implementing IParallelMngContainerFactory",message_passing_service);
  Ref<IParallelMngContainer> parallel_builder(pbf->_createParallelMngBuilder(1, pm2->communicator(), pm2->machineCommunicator()));

  for( Int32 i : ranks_to_read ){
    info() << "Reading Part sub_domain index=" << i;
    info() << "Using the checkpoint/restart service"
           << " <" << service_name << "> (implement ICheckpointReader2)";
    CheckpointInfo checkpoint_info2(ci);
    checkpoint_info2.setSubDomainRank(i);
    CheckpointReadInfo cri(checkpoint_info2);
    cri.setReader(reader);
    cri.setParallelMng(pm2);
    bool is_first = (i==my_rank);
    ISubDomain* sd2 = nullptr;
    if (is_first){
      sd2 = sd1;
    }
    else {
      String file_suffix = String::format("s_{0}_{1}",my_rank,i);
      ITraceMng* tm = app->createAndInitializeTraceMng(sd1->traceMng(),file_suffix);
      Ref<IParallelMng> sub_pm = parallel_builder->_createParallelMng(0,tm);
      SubDomainBuildInfo sdbi(sub_pm,i);
      sdbi.setCaseFileName(sd1->caseFullFileName());
      sdbi.setCaseBytes(case_bytes);

      // TODO: protéger arcaneCreateSubDomain()
      // dans une section critique.
      // On utilise directement arcaneCreateSubDomain() pour éviter
      // d'ajouter le sous-domaine créé à la liste des sous-domaines
      // de la session (cela peut poser problème car ensuite on ne
      // saura pas vraiment le détruire)
      sd2 = arcaneCreateSubDomain(sd1->session(),sdbi);
      sd2->initialize();
      sd2->readCaseMeshes();

      sd_to_merge_list2.add(sd2);
      sd2->setIsContinue();
      sd2->allocateMeshes();
    }
    sd2->variableMng()->readCheckpoint(cri);
    sd2->checkpointMng()->readObservable()->notifyAllObservers();
    // Il faut indiquer que les variables sont utilisées sinon
    // elles ne seront pas transférées. Les sous-domaines additionnels
    // ne créent pas les modules donc il est possible que le sous-domaine
    // sd1 ait plus de variables que les autres. C'est le cas avec les
    // variable NoDump qui n'existent pas chez les autres sous-domaines.
    // Il ne faut donc pas les initialiser.
    // TODO: il serait préférable de prendre les variables du
    // maillage communes à tout les maillages qu'on va fusionner.
    VariableCollection vars = sd2->variableMng()->variables();
    for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
      IVariable* var = *ivar;
      if (var->isUsed())
        continue;
      if ((var->property() & IVariable::PNoDump)!=0)
        continue;
      // Ne traite pas les variables qui ne sont pas sur des familles.
      if (var->itemFamilyName().null())
        continue;
      var->setUsed(true);
      info() << "LIST_VAR name=" << var->fullName();
    }
  }
  UniqueArray<IMesh*> meshes_to_merge;
  for( ISubDomain* sd_to_merge : sd_to_merge_list2 ){
    meshes_to_merge.add(sd_to_merge->defaultMesh());
  }

  // Change les propriétaires des maillages pour qu'ils référencent les
  // nouveaux rangs.
  _changeItemsOwner(sd1->defaultMesh(),old_ranks_to_new_ranks);
  for( IMesh* mesh : meshes_to_merge )
    _changeItemsOwner(mesh,old_ranks_to_new_ranks);

  {
    IMesh* mesh = sd1->defaultMesh();
    // Procède à la fusion des maillages
    mesh->modifier()->mergeMeshes(meshes_to_merge);
    // Met à jour IMesh::meshPartInfo() car
    // le nombre de parties des maillage a changé.
    MeshPartInfo p(makeMeshPartInfoFromParallelMng(mesh->parallelMng()));
    mesh->toPrimaryMesh()->setMeshPartInfo(p);
  }

  // TODO: détruire les sous-domaines créés
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
