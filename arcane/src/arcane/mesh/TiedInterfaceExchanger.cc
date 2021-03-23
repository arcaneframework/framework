// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedInterfaceExchanger.cc                                   (C) 2000-2016 */
/*                                                                           */
/* Echangeur entre sous-domaines des interfaces liées.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/mesh/TiedInterfaceExchanger.h"
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/TiedInterface.h"
#include "arcane/mesh/FaceFamily.h"

#include "arcane/SerializeBuffer.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemPrinter.h"
#include "arcane/ItemFamilySerializeArgs.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TiedInterfaceExchanger::OneSubDomainInfo
: public TraceAccessor
{
 public:
  OneSubDomainInfo(ITraceMng* tm,Int32 rank)
  : TraceAccessor(tm), m_rank(rank){}
 public:
  void addOne(Face face,ConstArrayView<TiedNode> tied_nodes,
              ConstArrayView<TiedFace> tied_faces)
  {
    //info() << "ADD_ONE_TIED_FACE face=" << ItemPrinter(face) << " rank=" << m_rank;
    uids.add(face.uniqueId());
    Integer nb_tied_node = tied_nodes.size();
    Integer nb_tied_face = tied_faces.size();
    nb_items.add(nb_tied_node);
    nb_items.add(nb_tied_face);
    for( Integer z=0; z<nb_tied_node; ++z ){
      const TiedNode& tn = tied_nodes[z];
      slaves_node_uid.add(tn.node().uniqueId());
      Real2 iso = tn.isoCoordinates();
      //info() << "ADD NODE node=" << ItemPrinter(tn.node()) << " iso=" << iso;
      isos.add(iso.x);
      isos.add(iso.y);
    }
    for( Integer z=0; z<nb_tied_face; ++z ){
      const TiedFace& tf = tied_faces[z];
      slaves_face_uid.add(tf.face().uniqueId());
      //info() << "ADD FACE face=" << ItemPrinter(tf.face());
    }
  }
  void serializeReserve(ISerializer* buf)
  {
    buf->reserveArray(nb_items);
    buf->reserveArray(isos);
    buf->reserveArray(uids);
    buf->reserveArray(slaves_node_uid);
    buf->reserveArray(slaves_face_uid);
  }

  void serializePut(ISerializer* buf)
  {
    buf->putArray(uids);
    buf->putArray(nb_items);
    buf->putArray(isos);
    buf->putArray(slaves_node_uid);
    buf->putArray(slaves_face_uid);
  }

  void deserialize(ISerializer* buf)
  {
    buf->getArray(uids);
    info() << "DESERIALIZE_INFO rank=" << m_rank << " nb_face=" << uids.largeSize();
    buf->getArray(nb_items);
    buf->getArray(isos);
    buf->getArray(slaves_node_uid);
    info() << "NODES: " << slaves_node_uid;
    buf->getArray(slaves_face_uid);
    info() << "FACES: " << slaves_face_uid;
  }

 public:
  Int32 m_rank;
  Int64UniqueArray uids;
  Int64UniqueArray slaves_node_uid;
  Int64UniqueArray slaves_face_uid;
  IntegerUniqueArray nb_items;
  RealUniqueArray isos;
  //! Uniquement après désérialisation, contient le localId() de chaque entité de \a slaves_node_uid.
  Int32UniqueArray slaves_node_local_id;
  //! Uniquement après désérialisation, contient le localId() de chaque entité de \a slaves_face_uid.
  Int32UniqueArray slaves_face_local_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TiedInterfaceExchanger::DeserializedInfo
: public TraceAccessor
, public ITiedInterfaceRebuilder
{
 public:
  DeserializedInfo(IMesh* mesh,ITraceMng* tm)
  : TraceAccessor(tm), m_face_family(mesh->faceFamily()), m_node_family(mesh->nodeFamily()){}
 public:
  struct FaceInfo
  {
    //! Instance contenant l'info
    OneSubDomainInfo* sd_info;
    //! Indice de la face dans le tableau \a uid.
    Integer face_index;
    //! Indice dans \a slaves_face_uid de la première face esclave.
    Integer slave_face_index;
    //! Indice dans \a isos et \a slaves_node_uid du première noeud esclave.
    Integer slave_node_index;

    FaceInfo(OneSubDomainInfo* _sd_info,
             Integer _face_index,
             Integer _slave_face_index,
             Integer _slave_node_index)
    {
      sd_info = _sd_info;
      face_index = _face_index;
      slave_face_index = _slave_face_index;
      slave_node_index = _slave_node_index;
    }
  };

  /*!
   * \brief Construit les infos après déserialisation.
   */
  void buildAfterDeserializeInfo(OneSubDomainInfo* sdi)
  {
    Integer nb_uid = sdi->uids.size();
    Integer face_index = 0;
    Integer node_index = 0;
    info(4) << "BUILD_AFTER_DESERIALIZE_INFO rank=" << sdi->m_rank << " nb_face=" << nb_uid;
    for( Integer i=0; i<nb_uid; ++i ){
      Int64 uid = sdi->uids[i];
      Integer nb_slave_node = sdi->nb_items[i*2];
      Integer nb_slave_face = sdi->nb_items[(i*2)+1];
      //info() << "ADD_FACE uid=" << uid << " nb_node=" << nb_slave_node << " nb_face=" << nb_slave_face;
      m_index_map.insert(std::make_pair(uid,FaceInfo(sdi,i,face_index,node_index)));
      face_index += nb_slave_face;
      node_index += nb_slave_node;
    }
    m_sdi_array.add(sdi);
  }

  void convertUniqueIds()
  {
    for(Integer z=0, n=m_sdi_array.size(); z<n; ++ z){
      OneSubDomainInfo* sdi = m_sdi_array[z];

      Integer nb_total_slave_face = sdi->slaves_face_uid.size();
      info(4) << "CONVERT_UID nb_slave_face=" << nb_total_slave_face;
      sdi->slaves_face_local_id.resize(sdi->slaves_face_uid.size());
      m_face_family->itemsUniqueIdToLocalId(sdi->slaves_face_local_id,sdi->slaves_face_uid);

      Integer nb_total_slave_node = sdi->slaves_node_uid.size();
      info(4) << "CONVERT_UID nb_slave_node=" << nb_total_slave_node;
      sdi->slaves_node_local_id.resize(sdi->slaves_node_uid.size());
      m_node_family->itemsUniqueIdToLocalId(sdi->slaves_node_local_id,sdi->slaves_node_uid);
    }
  }

  const FaceInfo& _getInfo(Int64 uid)
  {
    std::map<Int64,FaceInfo>::const_iterator iter = m_index_map.find(uid);
    if (iter==m_index_map.end())
      throw ArgumentException(A_FUNCINFO,String::format("Can not find uid '{0}'",uid));
    return iter->second;
  }

  void getNbSlave(Int64 uid,Integer* nb_node,Integer* nb_face)
  {
    const FaceInfo& fi = _getInfo(uid);
    Integer face_index = fi.face_index;
    *nb_node = fi.sd_info->nb_items[face_index*2];
    *nb_face = fi.sd_info->nb_items[(face_index*2)+1];
  }
  
  virtual void fillTiedInfos(Face face,
                             Int32ArrayView tied_nodes_lid,
                             Real2ArrayView tied_nodes_isos,
                             Int32ArrayView tied_faces_lid)
  {
    const FaceInfo& fi = _getInfo(face.uniqueId());
    OneSubDomainInfo* sdi = fi.sd_info;
    Int32ConstArrayView slaves_face_local_id = sdi->slaves_face_local_id.constView();
    Int32ConstArrayView slaves_node_local_id = sdi->slaves_node_local_id.constView();
    Integer face_index = fi.face_index;
    Integer nb_node = sdi->nb_items[face_index*2];
    Integer nb_face = sdi->nb_items[(face_index*2)+1];
    RealConstArrayView isos = sdi->isos.subConstView(fi.slave_node_index*2,nb_node*2);
    Int32ConstArrayView nodes_lid = slaves_node_local_id.subView(fi.slave_node_index,nb_node);
    Int32ConstArrayView faces_lid = slaves_face_local_id.subView(fi.slave_face_index,nb_face);
    tied_nodes_lid.copy(nodes_lid);
    tied_faces_lid.copy(faces_lid);
    for( Integer z=0; z<nb_node; ++z )
      tied_nodes_isos[z] = Real2(isos[(z*2)],isos[(z*2)+1]);
  }

 private:

  IItemFamily* m_face_family;
  IItemFamily* m_node_family;
  std::map<Int64,FaceInfo> m_index_map;
  UniqueArray<OneSubDomainInfo*> m_sdi_array;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TiedInterfaceExchanger::
TiedInterfaceExchanger(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_sub_domain(mesh->subDomain())
, m_deserialized_info(new DeserializedInfo(mesh,traceMng()))
, m_my_rank(mesh->parallelMng()->commRank())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TiedInterfaceExchanger::
~TiedInterfaceExchanger()
{
  for( SubDomainInfoMap::const_iterator iter(m_infos.begin()); iter!=m_infos.end(); ++iter )
    delete iter->second;
  delete m_deserialized_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline TiedInterfaceExchanger::OneSubDomainInfo* TiedInterfaceExchanger::
_getInfo(Int32 rank)
{
  SubDomainInfoMap::const_iterator iter = m_infos.find(rank);
  if (iter!=m_infos.end())
    return iter->second;
  OneSubDomainInfo* sdi = new OneSubDomainInfo(traceMng(),rank);
  m_infos.insert(std::make_pair(rank,sdi));
  return sdi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceExchanger::
initialize()
{
  // precond:
  // - cet appel doit se faire avant face.owner() new contiennent
  // les nouveaux propriétaires car dans ce cas le ENUMERATE_FACE
  // ne fonctionnera pas correctement puisqu'il s'agit d'une groupe
  // de mailles propres et il est remis à jour lorsque les propriétaires
  // changent.
  // - les faces esclaves et leur face maitre doivent avoir le même propriétaire

  // Parcours l'ensemble des interfaces liées et:
  // 1- sauve les infos sur les faces liées qui seront conservées
  // 2- prépare les infos pour chaque sous-domaine.
  // Chaque face liée contient soit la face maître associée (s'il s'agit
  // d'une face esclave), soit la liste des faces esclaves (s'il s'agit
  // d'une face maître).
  // 3- Supprime les infos de connectivité de chaque face.

  ConstArrayView<TiedInterface*> tied_interfaces(m_mesh->trueTiedInterfaces());
  FaceFamily& face_family = m_mesh->trueFaceFamily();
  VariableItemInt32& new_owners(face_family.itemsNewOwner());
  for( Integer i=0, n=tied_interfaces.size(); i<n; ++i ){
    TiedInterface* ti = tied_interfaces[i];
    FaceGroup master_group = ti->masterInterface();
    TiedInterfaceNodeList tied_nodes = ti->tiedNodes();
    TiedInterfaceFaceList tied_faces = ti->tiedFaces();
    Integer index = 0;
    ENUMERATE_FACE(iface,master_group){
      Face face = *iface;
      Int32 owner = new_owners[iface];
      OneSubDomainInfo* sdi = _getInfo(owner);
      ConstArrayView<TiedNode> face_tied_nodes = tied_nodes[index];
      ConstArrayView<TiedFace> face_tied_faces = tied_faces[index];
      //info() << "ADD_TIED_INFOS: face=" << ItemPrinter(face) << " n1=" << face_tied_nodes.size() << " n2=" << face_tied_faces.size();
      sdi->addOne(face,face_tied_nodes,face_tied_faces);
      ++index;
    }
    face_family.removeTiedInterface(ti);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sérialise les faces dans le tampon \a buf.
 *
 * Les faces sont créés en même temps que les mailles.
 * Cette sérialisation n'a donc pas besoin de les créer. On se contente
 * donc de gérer uniquement les données concernant les interfaces liées. 
 * Il faut faire bien attention de n'envoyer que les faces qui vont
 * appartenir au sous-domaine de destination et pas les faces fantômes
 */
void TiedInterfaceExchanger::
serialize(const ItemFamilySerializeArgs& args)
{
  ISerializer* buf = args.serializer();
  Int32 rank = args.rank();
  OneSubDomainInfo* sdi = _getInfo(rank);
  switch(buf->mode()){
  case ISerializer::ModeReserve:
    sdi->serializeReserve(buf);
    break;
  case ISerializer::ModePut:
    sdi->serializePut(buf);
    break;
  case ISerializer::ModeGet:
    sdi->deserialize(buf);
    m_deserialized_info->buildAfterDeserializeInfo(sdi);
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceExchanger::
finalize()
{
  // NOTE: il faut être certain que cette méthode soit appelée
  // une fois que les localId() des entités n'évoluent plus
  // (c.a.d après un compactage et un tri)
  
  // Il faut éventuellement reconstruire les infos après désérialisation
  // de notre propre sous-domaine car celui-ci n'a pas été désérialisé.
  OneSubDomainInfo* sdi = _getInfo(m_my_rank);
  m_deserialized_info->buildAfterDeserializeInfo(sdi);

  // Il faut convertir les uniqueId en localId.
  // Il ne faut surtout pas le faire avant d'être ici car les
  // localId() peuvent changer durant un repartitonnement.
  m_deserialized_info->convertUniqueIds();
  
  // Toutes les nouvelles entités et les groupes
  // ont été mis à jour. Il reste maintenant à récupérer les
  // informations de chaque face liée, à savoir la liste des noeuds
  // et face esclave ainsi que les coordonnées iso.
  ConstArrayView<TiedInterface*> tied_interfaces(m_mesh->trueTiedInterfaces());
  IntegerUniqueArray nb_slave_nodes;
  IntegerUniqueArray nb_slave_faces;
  FaceFamily& face_family = m_mesh->trueFaceFamily();
  for( Integer i=0, n=tied_interfaces.size(); i<n; ++i ){
    TiedInterface* ti = tied_interfaces[i];
    FaceGroup master_group = ti->masterInterface();
    Integer nb_master_face = master_group.size();
    nb_slave_nodes.resize(nb_master_face);
    nb_slave_faces.resize(nb_master_face);
    Integer index = 0;
    ENUMERATE_FACE(iface,master_group){
      Face face = *iface;
      Integer nb_node = 0;
      Integer nb_face = 0;
      //info() << "TRY GET INFO face=" << ItemPrinter(face) << " owner=" << face.owner()
      m_deserialized_info->getNbSlave(face.uniqueId(),&nb_node,&nb_face);
      nb_slave_nodes[index] = nb_node;
      nb_slave_faces[index] = nb_face;
      ++index;
    }
    ti->rebuild(m_deserialized_info,nb_slave_nodes,nb_slave_faces);
    face_family.applyTiedInterface(ti);
    ti->checkValid();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* TiedInterfaceExchanger::
family() const
{
  return m_mesh->faceFamily();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
