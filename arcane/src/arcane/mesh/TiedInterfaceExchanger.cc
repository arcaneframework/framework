// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedInterfaceExchanger.cc                                   (C) 2000-2016 */
/*                                                                           */
/* Exchanger between sub-domains of tied interfaces.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/mesh/TiedInterfaceExchanger.h"
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/TiedInterface.h"
#include "arcane/mesh/FaceFamily.h"

#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/ItemFamilySerializeArgs.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TiedInterfaceExchanger::OneSubDomainInfo
: public TraceAccessor
{
 public:

  OneSubDomainInfo(ITraceMng* tm, Int32 rank)
  : TraceAccessor(tm)
  , m_rank(rank)
  {}

 public:

  void addOne(Face face, ConstArrayView<TiedNode> tied_nodes,
              ConstArrayView<TiedFace> tied_faces)
  {
    //info() << "ADD_ONE_TIED_FACE face=" << ItemPrinter(face) << " rank=" << m_rank;
    uids.add(face.uniqueId());
    Integer nb_tied_node = tied_nodes.size();
    Integer nb_tied_face = tied_faces.size();
    nb_items.add(nb_tied_node);
    nb_items.add(nb_tied_face);
    for (Integer z = 0; z < nb_tied_node; ++z) {
      const TiedNode& tn = tied_nodes[z];
      slaves_node_uid.add(tn.node().uniqueId());
      Real2 iso = tn.isoCoordinates();
      //info() << "ADD NODE node=" << ItemPrinter(tn.node()) << " iso=" << iso;
      isos.add(iso.x);
      isos.add(iso.y);
    }
    for (Integer z = 0; z < nb_tied_face; ++z) {
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
  //! Only after deserialization, contains the localId() of each entity in \a slaves_node_uid.
  Int32UniqueArray slaves_node_local_id;
  //! Only after deserialization, contains the localId() of each entity in \a slaves_face_uid.
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

  DeserializedInfo(IMesh* mesh, ITraceMng* tm)
  : TraceAccessor(tm)
  , m_face_family(mesh->faceFamily())
  , m_node_family(mesh->nodeFamily())
  {}

 public:

  struct FaceInfo
  {
    //! Instance containing the info
    OneSubDomainInfo* sd_info;
    //! Index of the face in the \a uid array.
    Integer face_index;
    //! Index in \a slaves_face_uid of the first slave face.
    Integer slave_face_index;
    //! Index in \a isos and \a slaves_node_uid of the first slave node.
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
   * \brief Constructs the info after deserialization.
   */
  void buildAfterDeserializeInfo(OneSubDomainInfo* sdi)
  {
    Integer nb_uid = sdi->uids.size();
    Integer face_index = 0;
    Integer node_index = 0;
    info(4) << "BUILD_AFTER_DESERIALIZE_INFO rank=" << sdi->m_rank << " nb_face=" << nb_uid;
    for (Integer i = 0; i < nb_uid; ++i) {
      Int64 uid = sdi->uids[i];
      Integer nb_slave_node = sdi->nb_items[i * 2];
      Integer nb_slave_face = sdi->nb_items[(i * 2) + 1];
      //info() << "ADD_FACE uid=" << uid << " nb_node=" << nb_slave_node << " nb_face=" << nb_slave_face;
      m_index_map.insert(std::make_pair(uid, FaceInfo(sdi, i, face_index, node_index)));
      face_index += nb_slave_face;
      node_index += nb_slave_node;
    }
    m_sdi_array.add(sdi);
  }

  void convertUniqueIds()
  {
    for (Integer z = 0, n = m_sdi_array.size(); z < n; ++z) {
      OneSubDomainInfo* sdi = m_sdi_array[z];

      Integer nb_total_slave_face = sdi->slaves_face_uid.size();
      info(4) << "CONVERT_UID nb_slave_face=" << nb_total_slave_face;
      sdi->slaves_face_local_id.resize(sdi->slaves_face_uid.size());
      m_face_family->itemsUniqueIdToLocalId(sdi->slaves_face_local_id, sdi->slaves_face_uid);

      Integer nb_total_slave_node = sdi->slaves_node_uid.size();
      info(4) << "CONVERT_UID nb_slave_node=" << nb_total_slave_node;
      sdi->slaves_node_local_id.resize(sdi->slaves_node_uid.size());
      m_node_family->itemsUniqueIdToLocalId(sdi->slaves_node_local_id, sdi->slaves_node_uid);
    }
  }

  const FaceInfo& _getInfo(Int64 uid)
  {
    std::map<Int64, FaceInfo>::const_iterator iter = m_index_map.find(uid);
    if (iter == m_index_map.end())
      throw ArgumentException(A_FUNCINFO, String::format("Can not find uid '{0}'", uid));
    return iter->second;
  }

  void getNbSlave(Int64 uid, Integer* nb_node, Integer* nb_face)
  {
    const FaceInfo& fi = _getInfo(uid);
    Integer face_index = fi.face_index;
    *nb_node = fi.sd_info->nb_items[face_index * 2];
    *nb_face = fi.sd_info->nb_items[(face_index * 2) + 1];
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
    Integer nb_node = sdi->nb_items[face_index * 2];
    Integer nb_face = sdi->nb_items[(face_index * 2) + 1];
    RealConstArrayView isos = sdi->isos.subConstView(fi.slave_node_index * 2, nb_node * 2);
    Int32ConstArrayView nodes_lid = slaves_node_local_id.subView(fi.slave_node_index, nb_node);
    Int32ConstArrayView faces_lid = slaves_face_local_id.subView(fi.slave_face_index, nb_face);
    tied_nodes_lid.copy(nodes_lid);
    tied_faces_lid.copy(faces_lid);
    for (Integer z = 0; z < nb_node; ++z)
      tied_nodes_isos[z] = Real2(isos[(z * 2)], isos[(z * 2) + 1]);
  }

 private:

  IItemFamily* m_face_family;
  IItemFamily* m_node_family;
  std::map<Int64, FaceInfo> m_index_map;
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
, m_deserialized_info(new DeserializedInfo(mesh, traceMng()))
, m_my_rank(mesh->parallelMng()->commRank())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TiedInterfaceExchanger::
~TiedInterfaceExchanger()
{
  for (SubDomainInfoMap::const_iterator iter(m_infos.begin()); iter != m_infos.end(); ++iter)
    delete iter->second;
  delete m_deserialized_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline TiedInterfaceExchanger::OneSubDomainInfo* TiedInterfaceExchanger::
_getInfo(Int32 rank)
{
  SubDomainInfoMap::const_iterator iter = m_infos.find(rank);
  if (iter != m_infos.end())
    return iter->second;
  OneSubDomainInfo* sdi = new OneSubDomainInfo(traceMng(), rank);
  m_infos.insert(std::make_pair(rank, sdi));
  return sdi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceExchanger::
initialize()
{
  // precond:
  // - this call must be made before face.owner() new contains
  // the new owners because in this case ENUMERATE_FACE
  // will not work correctly since it is a group
  // of own cells and it is updated when owners
  // change.
  // - slave faces and their master face must have the same owner

  // Iterates through all tied interfaces and:
  // 1- saves the information about the tied faces that will be kept
  // 2- prepares the information for each sub-domain.
  // Each tied face contains either the associated master face (if it is
  // a slave face), or the list of slave faces (if it is
  // a master face).
  // 3- Deletes the connectivity information for each face.

  ConstArrayView<TiedInterface*> tied_interfaces(m_mesh->trueTiedInterfaces());
  FaceFamily& face_family = m_mesh->trueFaceFamily();
  VariableItemInt32& new_owners(face_family.itemsNewOwner());
  for (Integer i = 0, n = tied_interfaces.size(); i < n; ++i) {
    TiedInterface* ti = tied_interfaces[i];
    FaceGroup master_group = ti->masterInterface();
    TiedInterfaceNodeList tied_nodes = ti->tiedNodes();
    TiedInterfaceFaceList tied_faces = ti->tiedFaces();
    Integer index = 0;
    ENUMERATE_FACE (iface, master_group) {
      Face face = *iface;
      Int32 owner = new_owners[iface];
      OneSubDomainInfo* sdi = _getInfo(owner);
      ConstArrayView<TiedNode> face_tied_nodes = tied_nodes[index];
      ConstArrayView<TiedFace> face_tied_faces = tied_faces[index];
      //info() << "ADD_TIED_INFOS: face=" << ItemPrinter(face) << " n1=" << face_tied_nodes.size() << " n2=" << face_tied_faces.size();
      sdi->addOne(face, face_tied_nodes, face_tied_faces);
      ++index;
    }
    face_family.removeTiedInterface(ti);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Serializes the faces into the buffer \a buf.
 *
 * The faces are created at the same time as the cells.
 * This serialization therefore does not need to create them. We only manage
 * the data concerning the tied interfaces.
 * It is important to be careful to only send the faces that will
 * belong to the destination sub-domain and not the ghost faces.
 */
void TiedInterfaceExchanger::
serialize(const ItemFamilySerializeArgs& args)
{
  ISerializer* buf = args.serializer();
  Int32 rank = args.rank();
  OneSubDomainInfo* sdi = _getInfo(rank);
  switch (buf->mode()) {
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
  // NOTE: it must be certain that this method is called
  // once the localId() of the entities no longer changes
  // (i.e., after compaction and sorting)

  // It may be necessary to rebuild the info after deserialization
  // of our own sub-domain because it was not deserialized.
  OneSubDomainInfo* sdi = _getInfo(m_my_rank);
  m_deserialized_info->buildAfterDeserializeInfo(sdi);

  // It is necessary to convert the uniqueId to localId.
  // It must absolutely not be done before reaching here because the
  // localId() can change during a partitioning.
  m_deserialized_info->convertUniqueIds();

  // All new entities and groups
  // have been updated. Now it remains to retrieve the
  // information for each tied face, namely the list of nodes
  // and slave faces as well as the iso coordinates.
  ConstArrayView<TiedInterface*> tied_interfaces(m_mesh->trueTiedInterfaces());
  IntegerUniqueArray nb_slave_nodes;
  IntegerUniqueArray nb_slave_faces;
  FaceFamily& face_family = m_mesh->trueFaceFamily();
  for (Integer i = 0, n = tied_interfaces.size(); i < n; ++i) {
    TiedInterface* ti = tied_interfaces[i];
    FaceGroup master_group = ti->masterInterface();
    Integer nb_master_face = master_group.size();
    nb_slave_nodes.resize(nb_master_face);
    nb_slave_faces.resize(nb_master_face);
    Integer index = 0;
    ENUMERATE_FACE (iface, master_group) {
      Face face = *iface;
      Integer nb_node = 0;
      Integer nb_face = 0;
      //info() << "TRY GET INFO face=" << ItemPrinter(face) << " owner=" << face.owner()
      m_deserialized_info->getNbSlave(face.uniqueId(), &nb_node, &nb_face);
      nb_slave_nodes[index] = nb_node;
      nb_slave_faces[index] = nb_face;
      ++index;
    }
    ti->rebuild(m_deserialized_info, nb_slave_nodes, nb_slave_faces);
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

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
